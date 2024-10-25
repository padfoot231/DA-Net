import os
import itertools
import random
from matplotlib.pyplot import title
import imageio.v2 as imageio
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
from envmap import EnvironmentMap
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


def distort_batch(x, alpha, D, shift=(0.0, 0.0), phi=0.0) :
    """Distort a batch of images (in-place) using a fisheye distortion model (same as distort_image but for a batch of images)
    Args:
        x (torch.Tensor): the batch to distort
        alpha (float): fov angle (radians)
        D (list[float]): a list containing the k1, k2, k3 and k4 parameters
        shift (tuple[float, float]): x and y shift (respectively)
        phi (float): the rotation angle (radians)
    Returns:
        torch.Tensor: the distorted batch
    """
    arr = x.moveaxis(1, -1).numpy()
    for i in range(arr.shape[0]):
        arr[i] = distort_image(arr[i], alpha, D, shift, phi)
    return x

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

def get_inverse_distortion(num_points, D, fov, mag=1.0):
    
    theta_d_max = fov/2
    # xi = 0

    m = 2.288135593220339
    n = 5.0
    a = theta_d_max
    b = 8.31578947368421
    c =  0.3333333333333333

    def f(x, n, a, b):
        return b*torch.pow(x/a, n)
    def h(x, m, a):
        return -torch.pow(-x/a + 1, m) + 1
    def g(x, m, n, a, b, c):
        return c*f(x, n, a, b) + (1-c)*h(x, m, a)
    def g_inv(y, m, n, a, b, c):
        test_x = torch.linspace(0, theta_d_max, 100000).cuda()
        test_y = g(test_x, m, n, a, b, c).cuda()
        x = torch.zeros(num_points + 1).cuda()
        for i in range(num_points + 1):
            lower_idx = test_y[test_y <= y[i]].argmax()
            x[i] = test_x[lower_idx]
        return x

    def rad(D, theta):
        focal = lambda x: 1/(x * (D[0] * x**0 + D[1] * x**1 + D[2] * x**2 + D[3] * x**3))
        f = focal(theta_d_max).reshape(1, D.shape[1])
        funct = g_inv(theta, m = m, n = n, a = a, b = b, c = c)
        funct = funct.reshape(num_points + 1, 1)
        # breakpoint()
        radius = f*funct * (D[0] * funct**0 + D[1] * funct**1 + D[2] * funct**2 + D[3] * funct**3)
        # radius = ((torch.cos(theta_max) + xi)/torch.sin(theta_max))*torch.sin(funct)/(torch.cos(funct) + xi)
        return radius
    # print("ass")
    # theta_d_max = torch.tan(fov/2)
    # theta_d = linspace(torch.tensor([0]).cuda(), g(theta_d_max), num_points+1).cuda()
    theta_d = torch.linspace(0, g(theta_d_max, m = m, n = n, a = a, b = b, c = c), num_points + 1).cuda()

    r_list = rad(D, theta_d)
    # print("theta")
    # r_lin = rad(theta_d_num)
    # r_d = rad(theta_d_num1)
    return r_list, theta_d_max


def get_inverse_dist_spherical(num_points, xi, fov, new_f):
    
    theta_d_max = fov/2
    # import pdb;pdb.set_trace()
    # xi = 0

    m = 2.288135593220339
    n = 5.0
    a = theta_d_max
    b = 8.31578947368421
    c =  0.3333333333333333

    def f(x, n, a, b):
        return b*torch.pow(x/a, n)
    def h(x, m, a):
        return -torch.pow(-x/a + 1, m) + 1
    def g(x, m, n, a, b, c):
        return c*f(x, n, a, b) + (1-c)*h(x, m, a)
    def g_inv(y, m, n, a, b, c):
        test_x = torch.linspace(0, theta_d_max, 100000).cuda()
        test_y = g(test_x, m, n, a, b, c).cuda()
        x = torch.zeros(num_points + 1).cuda()
        for i in range(num_points + 1):
            lower_idx = test_y[test_y <= y[i]].argmax()
            x[i] = test_x[lower_idx]
        return x

    def rad(xi, theta):
        funct = g_inv(theta, m = m, n = n, a = a, b = b, c = c)
        funct = funct.reshape(num_points + 1, 1)
        # breakpoint()
        radius = ((torch.cos(funct[-1]) + xi)/torch.sin(funct[-1]))*torch.sin(funct)/(torch.cos(funct) + xi)
        # radius = ((torch.cos(theta_max) + xi)/torch.sin(theta_max))*torch.sin(funct)/(torch.cos(funct) + xi)
        return radius
    # print("ass")
    # theta_d_max = torch.tan(fov/2)
    # theta_d = linspace(torch.tensor([0]).cuda(), g(theta_d_max), num_points+1).cuda()
    theta_d = torch.linspace(0, g(theta_d_max, m = m, n = n, a = a, b = b, c = c), num_points + 1).cuda()

    r_list = rad(xi, theta_d)
    # print("theta")
    # r_lin = rad(theta_d_num)
    # r_d = rad(theta_d_num1)
    return r_list, theta_d_max

def get_sample_params_from_subdiv(subdiv, distortion_model, img_size, D=torch.tensor(np.array([0.5, 0.5, 0.5, 0.5]).reshape(4,1)).cuda(), radius_buffer=0, azimuth_buffer=0):
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
        D_min, theta_max = get_inverse_dist_spherical(subdiv[0], xi, fov, f)
        D_min = D_min
    elif distortion_model == 'polynomial' or distortion_model == 'polynomial_woodsc':
        # 
        D_min, theta_max = get_inverse_distortion(subdiv[0], D, 1.0)
        # D_min = D_min*max_radius
    alpha = 2*torch.tensor(np.pi).cuda() / subdiv[1]
    # import pdb;pdb.set_trace()
    D_min = D_min[1:, :]
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
    # import pdb;pdb.set_trace()
    u = x*torch.sqrt(1- torch.pow(y, 2)/2)
    v = y*torch.sqrt(1- torch.pow(x, 2)/2)
    return u.transpose(1, 2).transpose(0,1), v.transpose(1, 2).transpose(0,1), theta_max


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
        D_min, theta_max = get_inverse_dist_spherical(subdiv[0], xi, fov, f)
        D_min = D_min.transpose(0, 1)
        # import pdb;pdb.set_trace()
        # D_min = D_min.reshape((subdiv[0], D.shape[1]))
        D_min = D_min*max_radius
    elif distortion_model == 'polynomial' or distortion_model == 'polynomial_woodsc':
        fov = torch.tensor(3.31613).cuda()
        D_min, theta_max = get_inverse_distortion(subdiv[0], D, fov, 1.0)
        D_min = D_min.transpose(0, 1)
        D_min = D_min*max_radius
    # import pdb;pdb.set_trace() .
    flip = -torch.flip(D_min[:, 1:], (1, 0))
    r_ = torch.cat((torch.flip(flip, [0]), D_min[:, 1:]), axis = 1).transpose(0, 1)
    # A_ = r_1.reshape(D.shape[1], width, 1).repeat_interleave(width, 2)
    # B_ = r_1.reshape(D.shape[1], 1, width).repeat_interleave(width, 1)
    R1 = torch.linspace(0.0, 1, width).cuda()
    R2 = torch.linspace(0.0, 1, width).cuda()
    a = 2*R1 - 1
    b = 2*R2 - 1
    radius = torch.zeros((D.shape[1], width, width), dtype=torch.float32).cuda()
    phi = torch.zeros((D.shape[1],width, width), dtype=torch.float32).cuda()

    
    B  = D.shape[1]
    A, B_mesh = torch.meshgrid(a, b, indexing='ij')  # Shape: (H, H)

    # Stack the meshgrid to handle the batch size in one step
    A_stack = A.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, H, H)
    B_stack = B_mesh.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, H, H)

    # Create meshgrid for r_[:, i] for each i in the batch without loop
    A_ = torch.stack([torch.meshgrid(r_[:, i], r_[:, i], indexing='ij')[0] for i in range(B)])
    B_ = torch.stack([torch.meshgrid(r_[:, i], r_[:, i], indexing='ij')[1] for i in range(B)])

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
    return x, y, r_


def Eliptical_mapping(subdiv, distortion_model, img_size, D=torch.tensor(np.array([0.5, 0.5, 0.5, 0.5]).reshape(4,1)).cuda()):
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
        D_min, theta_max = get_inverse_dist_spherical(subdiv[0], xi, fov, f)
        D_min = D_min.transpose(0, 1)
        # import pdb;pdb.set_trace()
        # D_min = D_min.reshape((subdiv[0], D.shape[1]))
        D_min = D_min
    elif distortion_model == 'polynomial' or distortion_model == 'polynomial_woodsc':
        fov = torch.tensor(3.31613).cuda()
        D_min, theta_max = get_inverse_distortion(subdiv[0], D, fov, 1.0)
        D_min = D_min.transpose(0, 1)
        D_min = D_min
    # import pdb;pdb.set_trace() 
    r = D_min[:, 1:]
    B, N = r.shape
    size = 3 + (N - 1)*2
    x = torch.zeros((B, size, size)).cuda()
    y = torch.zeros((B, size, size)).cuda()
    for i in range(N):
        # import pdb;pdb.set_trace()
        step = 3 + i*2
        start = -r[:, i]
        end = r[:, i]
        base_linspace = torch.linspace(0, 1, step).unsqueeze(0).expand(B, -1).cuda()  # Shape: (B, steps)
        # Scale the base linspace to the batch-specific start and end values
        result = start.unsqueeze(1) + base_linspace * (end.unsqueeze(1) - start.unsqueeze(1))
        
    #     import pdb;pdb.set_trace()
        a = torch.cat((-r[:, i].reshape(1,r[:, i].shape[0]) , r[:, i].reshape(1,r[:, i].shape[0])), dim = 0)
    #     import pdb;pdb.set_trace()
        steps_x = result.shape[1]  # Number of x points
        steps_y = a.T.shape[1]  # Number of y points

        # Create fixed meshgrid structure
    #     import pdb;pdb.set_trace()
        xx = result.unsqueeze(1).expand(B, steps_y, steps_x).type(torch.float).transpose(1, 2)  # Shape: (B, steps_y, steps_x)
        yy = a.T.unsqueeze(2).expand(B, steps_y, steps_x).type(torch.float).transpose(1, 2)  # Shape: (B, steps_y, steps_x)

        xx_ , yy_ = yy.transpose(1, 2).type(torch.float), xx.transpose(1, 2).type(torch.float)
    #     import pdb;pdb.set_trace()
        split = (size - step)//2
        
        if split ==0 :
    #         import pdb;pdb.set_trace()
            x[:, [0,-1],:] = xx_
            x[:, :, [0,-1]] = xx
            y[:, [0,-1],:] = yy_
            y[:, :, [0,-1]] = yy
    #         import pdb;pdb.set_trace()
        else:
    #         import pdb;pdb.set_trace()
            x[:, [split,-split-1],split:-split] = xx_
            x[:, split:-split, [split,-split-1]] = xx
            y[:, [split,-split-1],split:-split] = yy_
            y[:, split:-split, [split,-split-1]] = yy
    u = x * torch.sqrt(1 - (y**2 / 2))*max_radius
    v = y * torch.sqrt(1 - (x**2 / 2))*max_radius
    return u, v, r

def cubemap(image, n_rad, fov, xi, h, order):


    # fov = 175
    # xi = 0.2
    # h = 128

    img_size= h #64
    bg= 0.3
    f = compute_focal(np.deg2rad(fov), xi, h)


    #here correct (not nan)
    e = EnvironmentMap(image, format_='fisheye', dist=[f/(h/img_size),xi])
    #breakpoint()
    basedim= n_rad*h//2 #always make it the half of H
    cube = e.copy().convertTo('cube',4*basedim, order=order)
    #print(cube.data.shape)

    all_cube= cube.data

    all_cube[basedim//2:2*basedim+basedim//2, basedim//2 :2*basedim+basedim//2]=0 + bg
    #test with removing the nan
    all_cube[np.isnan(all_cube)]=0 + bg

    #plt.imsave('cube.png', all_cube)

    #print(np.where(np.isnan(all_cube)))

    top = all_cube[0:basedim, basedim:2*basedim]
    front = all_cube[basedim:2*basedim, basedim:2*basedim]
    right= np.fliplr(np.flipud(all_cube[basedim:2*basedim, 2*basedim:3*basedim]))
    back = all_cube[3*basedim:4*basedim, basedim:2*basedim] 
    left= np.fliplr(np.flipud(all_cube[1*basedim:2*basedim, 0:basedim]))
    bottom= all_cube[2*basedim:3*basedim, basedim:2*basedim]

    cubemap= np.zeros((3*basedim,3*basedim,image.ndim))+bg
    cubemap[0:basedim, basedim:2*basedim]= bottom
    cubemap[basedim:2*basedim,0:basedim]= left
    cubemap[basedim:2*basedim,basedim:2*basedim]= back
    cubemap[basedim:2*basedim,2*basedim:3*basedim]= right
    cubemap[2*basedim:3*basedim,basedim:2*basedim] = top
    # print(cubemap.shape)
    x,y= np.where(np.any(cubemap!=bg, axis=-1))
    min_x, max_x, min_y, max_y = x.min(), x.max(), y.min(),y.max()
    # print(min_x, min_y, max_x, max_y)
    # plt.imsave(os.path.join(out_folder, 'before_cubemap_{}.png'.format(xi)), cubemap)

    new_cubemap= cubemap[min_y:max_y+1,min_x:max_x+1]

    return new_cubemap
# print("after removing borders ", new_cubemap.shape)
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
