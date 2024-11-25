import os
import itertools
import random
from matplotlib.pyplot import title
# import imageio.v2 as imageio
import torch
# import cv2
import numpy as np
import torch.distributed as dist
from torch._six import inf
import scipy.optimize
# from pyinstrument import Profiler
from PIL import Image
import torch.nn as nn
# profiler = Profiler(interval=0.0001)
import torch.nn.functional as F
import torchvision.transforms as T
# import cv2

import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import zoom
import torch

transform = T.ToPILImage()

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_miou, miou, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_miou,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    # breakpoint()
    if miou > max_miou :
        print(epoch)
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_best.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")
    
    if epoch%10 == 0:
        print("ass")
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


###spherical distortion #########################
def compute_fov(f, xi, width):
    return 2 * np.arccos((xi + np.sqrt(1 + (1 - xi**2) * (width/2/f)**2)) / ((width/2/f)**2 + 1) - xi)

# compute focal length from field of view and xi
def compute_focal(fov, xi, width):
    return width / 2 * (xi + np.cos(fov/2)) / np.sin(fov/2)


def linspace(start, stop, num):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out

def get_inverse_distortion(num_points, D, mag=1.0):
    focal = lambda x: 1/(x.reshape(1, x.shape[0]).repeat_interleave(D.shape[1], 0).flatten() * (torch.outer(D[0], x**0).flatten() + torch.outer(D[1], x**1).flatten() + torch.outer(D[2], x**2).flatten() +torch.outer(D[3], x**3).flatten()))
    fov = 1.48806
    theta_d_max = torch.tensor(fov).reshape(1).cuda()
    f = focal(theta_d_max).reshape(1, D.shape[1])
    dist_func = lambda x: f* x * (D[0] * x**0 + D[1] * x**1 + D[2] * x**2 + D[3] * x**3)
    # theta_d = torch.linspace(0, torch.tan(torch.tensor(fov)), num_points+1).reshape(1, num_points+1).repeat_interleave(D.shape[1], 0).transpose(1,0).cuda()
    theta_d = torch.linspace(0, fov, num_points+1).reshape(1, num_points+1).repeat_interleave(D.shape[1], 0).transpose(1,0).cuda()
    delta = float(torch.diff(theta_d, axis=0)[0][0])
    a = np.random.uniform(0, 1)
    err = np.random.uniform(0, delta/2)
    if  a > 0.5:
        err = np.random.uniform(0, delta/2)
        theta_d = theta_d + err
        # theta_d[-1] = torch.tan(torch.tensor(fov))
        theta_d[-1] = torch.tensor(fov)
    elif a < 0.5:
        theta_d = theta_d - err
        theta_d[0] = 0.0
    # r_list = dist_func(torch.arctan(theta_d))

    r_list = dist_func(theta_d)
    # return r_list, torch.tan(torch.tensor(fov))
    return r_list, fov

def f(x):
    a = 1.38
    return a - a*torch.exp(-x)

# Define a function to numerically compute the inverse of f(x, a)
def inverse_f(y, x_init=0.0, lr=1e-2, steps=1000):
    # Convert y and a to tensors
    y = torch.tensor(y, dtype=torch.float32)
    
    # Initialize x as a tensor with requires_grad=True
    x = torch.tensor([x_init], dtype=torch.float32, requires_grad=True)
    
    # Set up the optimizer (Adam in this case)
    optimizer = torch.optim.Adam([x], lr=lr)
    
    # Perform gradient descent to minimize the difference f(x, a) - y
    for _ in range(steps):
        optimizer.zero_grad()
        
        # Compute f(x, a)
        fx = f(x)
        fx = fx.cuda()
        
        # Compute the loss as the difference between f(x, a) and y
        # import pdb;pdb.set_trace()
        loss = (fx - y) ** 2
        
        # Backpropagate to compute gradients
        loss.backward()
        
        # Update x
        optimizer.step()
    
    # Return the computed inverse (the value of x that makes f(x, a) close to y)
    return x.item()


def get_inverse_dist_spherical(num_points, xi, fov):

    theta_d_max = fov/2

    def rad(xi, theta):
#         funct = g_inv(theta, theta_d_max, num_points=num_points).cuda()
        funct = f(theta).cuda()
        funct = funct.reshape(num_points + 1, 1)
        # breakpoint()
        radius = ((torch.cos(funct[-1]) + xi)/torch.sin(funct[-1]))*torch.sin(funct)/(torch.cos(funct) + xi)
        # radius = ((torch.cos(theta_max) + xi)/torch.sin(theta_max))*torch.sin(funct)/(torch.cos(funct) + xi)
        return radius

#     theta_d = torch.linspace(0, g(theta_d_max), num_points).cuda()
    theta_d = torch.linspace(0, inverse_f(theta_d_max), num_points+1).cuda()
    # print(f"The inverse of f(x) = y for y = {theta_d_max} is approximately x = {theta_d[-1]}")
    r_list = rad(xi, theta_d)
    return r_list, theta_d

def get_sample_params_from_subdiv(subdiv, distortion_model, img_size, D=torch.tensor(np.array([0.5, 0.5, 0.5, 0.5]).reshape(4,1)).cuda()):
    """Generate the required parameters to sample every patch based on the subdivison
    Args:
        subdiv (tuple[int, int]): the number of subdivisions for which we need to create the 
                                  samples. The format is (radius_subdiv, azimuth_subdiv)
        n_radius (int): number of radius samples
        n_azimuth (int): number of azimuth samples
        img_size (tuple): the size of the image
    Returns:
        list[dict]: the list of parameters to sample every patch
    """
    max_radius = min(img_size)/2
    width = img_size[1]
    # D_min = get_inverse_distortion(subdiv[0], D, max_radius)
    if distortion_model == 'spherical': # in case of spherical distortion pass the 
        fov = D[2][0]
        f  = D[1]
        xi = D[0]
        D_min, theta_max = get_inverse_dist_spherical(subdiv[0], xi, fov)
        D_min = D_min*max_radius
        # breakpoint()
        import pdb;pdb.set_trace()
    elif distortion_model == 'polynomial' or distortion_model == 'polynomial_woodsc':
        # 
        D_min, theta_max = get_inverse_distortion(subdiv[0], D, 1.0)
        D_min = D_min*max_radius
    # breakpoint()
    D_min = D_min[1:, :]
    alpha = 2*torch.tensor(np.pi).cuda() / subdiv[1]
    D_min = D_min.reshape(subdiv[0], 1, D.shape[1]).repeat_interleave(subdiv[1], 1).cuda()
    phi_start = 0
    phi_end = 2*torch.tensor(np.pi)
    phi_step = alpha
    phi_list = torch.arange(phi_start, phi_end, phi_step)
    phi_list = phi_list.reshape(1, subdiv[1]).repeat_interleave(subdiv[0], 0).reshape(subdiv[0], subdiv[1], 1).repeat_interleave(D.shape[1], 2).cuda()
    # phi = p.transpose(1,0).flatten().cuda()
    phi_list_cos  = torch.cos(phi_list) 
    phi_list_sine = torch.sin(phi_list) 
    x = D_min * phi_list_cos    # takes time the cosine and multiplication function 
    y = D_min * phi_list_sine
    return x.transpose(1, 2).transpose(0,1), y.transpose(1, 2).transpose(0,1), theta_max


def concentric_dic_sampling_origin(subdiv, distortion_model, img_size, D=torch.tensor(np.array([0.5, 0.5, 0.5, 0.5]).reshape(4,1)).cuda()):
    """Generate the required parameters to sample every patch based on the subdivison
    Args:
        subdiv (tuple[int, int]): the number of subdivisions for which we need to create the 
                                  samples. The format is (radius_subdiv, azimuth_subdiv)
        n_radius (int): number of radius samples
        n_azimuth (int): number of azimuth samples
        img_size (tuple): the size of the image
    Returns:
        list[dict]: the list of parameters to sample every patch
    """
    max_radius = min(img_size)/2
    width = subdiv[0]*2 
    # D_min = get_inverse_distortion(subdiv[0], D, max_radius)
    if distortion_model == 'spherical': # in case of spherical distortion pass the 
        fov = D[2][0]
        f  = D[1]
        xi = D[0]
        D_min, theta_max = get_inverse_dist_spherical(subdiv[0], xi, fov)
        D_min = D_min.transpose(0, 1)
        # D_min = D_min.reshape((subdiv[0], D.shape[1]))
        D_min = D_min*max_radius
    # breakpoint()
    # import pdb;pdb.set_trace()
    # flip = -torch.flip(D_min[:, 1:], (0,1))
    flip = -torch.flip(D_min[:, 1:], (1, 0))
    r_ = torch.cat((torch.flip(flip, [0]), D_min[:, 1:]), axis = 1).transpose(0, 1)
    # r_ = torch.cat((-torch.flip(D_min, (0,1))[:, :-1], D_min[:, 1:]), axis = 1).transpose(0, 1)
    # A_ = r_1.reshape(D.shape[1], width, 1).repeat_interleave(width, 2)
    # B_ = r_1.reshape(D.shape[1], 1, width).repeat_interleave(width, 1)
    R1 = torch.linspace(0.0, 1, width).cuda()
    R2 = torch.linspace(0.0, 1, width).cuda()
    a = 2*R1 - 1
    b = 2*R2 - 1
    radius = torch.zeros((D.shape[1], width, width), dtype=torch.float32).cuda()
    phi = torch.zeros((D.shape[1],width, width), dtype=torch.float32).cuda()

    # Create meshgrid for a and b
    # A, B = torch.meshgrid(a, b,indexing='ij')

    # Vectorized conditions
    # for i in range(D.shape[1]):
    #     # Create meshgrid for a and b
    #     A_, B_ = torch.meshgrid(r_[:, i], r_[:, i], indexing='ij')

    #     # Vectorized conditions
    #     condition1 = (A == 0) & (B == 0)
    #     condition2 = A * A > B * B
    #     condition3 = ~condition1 & ~condition2

    #     # Calculate radius and phi for condition1
    #     radius[i, condition1] = 0
    #     phi[i, condition1] = 0

    #     # Calculate radius and phi for condition2
    #     radius[i, condition2] = A_[condition2]
    #     phi[i, condition2] = (np.pi / 4) * (B[condition2] / A[condition2])

    #     # Calculate radius and phi for condition3 (the remaining cases)
    #     radius[i, condition3] = B_[condition3]
    #     phi[i, condition3] = np.pi / 2 - (np.pi / 4) * (A[condition3] / B[condition3])

    # Assuming a, b, r_ are already defined tensors
    # a = torch.rand(128)  # Replace with actual a
    # b = torch.rand(128)  # Replace with actual b
    # r_ = torch.rand(128, B)  # Assuming r_ is provided

    # Create meshgrid for a and b
    # breakpoint()
    B  = D.shape[1]
    A, B_mesh = torch.meshgrid(a, b)  # Shape: (H, H)

    # Stack the meshgrid to handle the batch size in one step
    A_stack = A.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, H, H)
    B_stack = B_mesh.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, H, H)

    # Create meshgrid for r_[:, i] for each i in the batch without loop
    A_ = torch.stack([torch.meshgrid(r_[:, i], r_[:, i])[0] for i in range(B)])
    B_ = torch.stack([torch.meshgrid(r_[:, i], r_[:, i])[1] for i in range(B)])

    # Vectorized conditions across the batch
    condition1 = (A_stack == 0) & (B_stack == 0)
    condition2 = A_stack * A_stack > B_stack * B_stack
    condition3 = ~condition1 & ~condition2

    # Initialize radius and phi tensors
    radius = torch.zeros((B, width, width)).cuda()
    phi = torch.zeros((B, width, width)).cuda()

    # Calculate radius and phi for all conditions in one go across the batch
    radius = torch.where(condition1, torch.tensor(0.0).cuda(), torch.where(condition2, A_, B_))
    phi = torch.where(
        condition1, 
        torch.tensor(0.0).cuda(), 
        torch.where(condition2, (np.pi / 4) * (B_stack / A_stack), np.pi / 2 - (np.pi / 4) * (A_stack / B_stack))
    )

    # Now radius and phi have shape (B, H, H)

    # breakpoint()
    x = radius*torch.cos(phi)
    y = radius*torch.sin(phi)
    return x, y, theta_max

# def get_optimal_buffers(subdiv, n_radius, n_azimuth, img_size):
#     """Get the optimal radius and azimuth buffers for a given subdivision

#     Args:
#         subdiv (int or tuple[int, int]): the number of subdivisions for which we need to create the samples.
#                                          If specified as a tuple, the format is (radius_subdiv, azimuth_subdiv)
#         n_radius (int): number of radius samples
#         n_azimuth (int): number of azimuth samples
#         img_size (tuple): the size of the image

#     Returns:
#         tuple[int, int]: the optimal radius and azimuth buffers
#     """

#     # Get the optimal buffers
#     if isinstance(subdiv, int):
#         radius_buffer = img_size[0] / (2**(subdiv+1)*n_radius)
#         azimuth_buffer = 2*np.pi / (2**(subdiv+2)*n_azimuth)
#     elif isinstance(subdiv, tuple) and len(subdiv) == 2:
#         radius_buffer = img_size[0] / (radius_subdiv*n_radius*2*2)
#         azimuth_buffer = 2*np.pi / (azimuth_subdiv*n_azimuth*2)
#     else:
#         raise ValueError("Invalid subdivision")
   
#     return radius_buffer, azimuth_buffer


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes



class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


if __name__=='__main__':
    # profiler.start()
    # import matplotlib.pyplot as plt
    # _, ax = plt.subplots(figsize=(8, 8))
    # ax.set_title("Sampling locations")
    # colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']

    # subdiv = 3
    radius_subdiv = 2
    azimuth_subdiv = 8
    subdiv = (radius_subdiv, azimuth_subdiv)
    # subdiv = 3
    n_radius = 8
    n_azimuth = 8
    img_size = (64, 64)
    # radius_buffer, azimuth_buffer = get_optimal_buffers(subdiv, n_radius, n_azimuth, img_size)
    radius_buffer = azimuth_buffer = 0

    D = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0]).reshape(1,4).transpose(1,0)).cuda()
    # 

    x, y, theta = get_sample_params_from_subdiv(
        subdiv=subdiv,
        img_size=img_size,
        n_radius=n_radius,
        D = D,
        n_azimuth=n_azimuth,
        radius_buffer=radius_buffer,
        azimuth_buffer=azimuth_buffer
    )

    # profiler.stop()
    # profiler.print()
