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
from numpy import logical_and as land, logical_or as lor

# import cv2

import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import zoom
import torch

# grid_size = 128



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


def f(x, n=5.0, a=3.48, b=8.31578947368421):
    return b*torch.pow(x/a, n)
def h(x, m=2.288135593220339, a=3.48):
    return -torch.pow(-x/a + 1, m) + 1
def g(x, m=2.288135593220339, n= 5.0, a=3.48, b=8.31578947368421, c=0.3333333333333333):
    return c*f(x, a = a) + (1-c)*h(x, a=a)
# def g_inv(y, theta_d_max):
#     # Create test_x values
#     test_x = torch.linspace(0, theta_d_max, 100000)
#     test_y = g(test_x, a=theta_d_max)
    
#     # Expand dimensions to allow broadcasting for batch processing
#     y_expanded = y.unsqueeze(-1)  # Shape: (batch_size, num_points, 1)
#     test_y_expanded = test_y.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 100000)
    
#     # Find indices where test_y is just below y
#     diff = test_y_expanded - y_expanded  # Shape: (batch_size, num_points, 100000)
#     lower_idx = (diff <= 0).sum(dim=-1) - 1  # Find the last index where test_y <= y
#     lower_idx = torch.clamp(lower_idx, min=0)  # Ensure no negative indices
    
#     # Get the corresponding values of test_x for each index in the batch
#     x = test_x[lower_idx]  # Shape: (batch_size, num_points)
#     return x

def g_inv(y, theta_d_max):
    # Create test_x values
    test_x = torch.linspace(0, theta_d_max, 1000)  # Reduced number of points
    test_y = g(test_x, a=theta_d_max)

    # Use torch.searchsorted for efficient binary search
    indices = torch.searchsorted(test_y, y, right=True)  # Find insertion points
    indices = torch.clamp(indices, 1, test_x.size(0) - 1)  # Clamp to valid index range

    # Get the two closest points around each y value for interpolation
    x0, x1 = test_x[indices - 1], test_x[indices]
    y0, y1 = test_y[indices - 1], test_y[indices]

    # Linear interpolation
    x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)

    return x


def polynomial_distoriton(num_points, D, fov, mag=1.0):
    theta_d_max = fov/2
    def rad(D, theta):
        focal = lambda x: 1/(x * (D[0] * x**0 + D[1] * x**1 + D[2] * x**2 + D[3] * x**3))
        f = focal(theta_d_max).reshape(1, D.shape[1])
        funct = g_inv(theta, theta_d_max)
        # funct = funct.reshape(num_points + 1, 1)
        funct = funct.reshape(1, num_points + 1)
        funct.repeat_interleave(D.shape[1], 0)
        # import pdb;pdb.set_trace()
        # radius = f*funct * (D[0] * funct**0 + D[1] * funct**1 + D[2] * funct**2 + D[3] * funct**3)
        funct_powers = torch.stack([funct**1, funct**2, funct**3, funct**4], dim=0)
        radius = torch.sum(D[:, :, None] * funct_powers, dim=0)  # shape (2, 65)
        # radius = ((torch.cos(theta_max) + xi)/torch.sin(theta_max))*torch.sin(funct)/(torch.cos(funct) + xi)
        # import pdb;pdb.set_trace()
        return f.T*radius
    # print("ass")
    # theta_d_max = torch.tan(fov/2)
    # theta_d = linspace(torch.tensor([0]).cuda(), g(theta_d_max), num_points+1).cuda()
    theta_d = torch.linspace(0, g(theta_d_max, a=theta_d_max), num_points + 1).cuda()

    r_list = rad(D, theta_d)

    return r_list

def spherical_distoriton(num_points, xi, fov, new_f):
    
    theta_d_max = fov/2

    def rad(xi, theta):        
        k_d = ((torch.cos(theta_d_max) + xi)/torch.sin(theta_d_max))
        num_points = theta.shape[1]
        # print(num_points)
        funct = g_inv(theta, theta_d_max)
    #     import pdb;pdb.set_trace()
    #     funct = funct.reshape(theta.shape[0], num_points)
        radius = k_d*torch.sin(funct)/(torch.cos(funct) + xi)
        return radius
    # print("ass")
    # theta_d_max = torch.tan(fov/2)
    # theta_d = linspace(torch.tensor([0]).cuda(), g(theta_d_max), num_points+1).cuda()
    theta_d = torch.linspace(0, g(theta_d_max, a=theta_d_max), num_points + 1).reshape(1, num_points + 1)
    # import pdb;pdb.set_trace()
    xi = xi.reshape(xi.shape[0], 1)
    r_list = rad(xi, theta_d)
    # print("theta")
    # r_lin = rad(theta_d_num)
    # r_d = rad(theta_d_num1)
    return r_list, theta_d_max



def concentric_dic_sampling(subdiv, distortion_model, img_size, D=torch.tensor(np.array([0.5, 0.5, 0.5, 0.5]).reshape(4,1)).cuda()):
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
        fov = D[2][0].cpu()
        f  = D[1].cpu()
        xi = D[0].cpu()
        # import pdb;pdb.set_trace()
        D_min, theta_max = spherical_distoriton(subdiv[0], xi, fov, f)
        # D_min = D_min.transpose(0, 1)
        # import pdb;pdb.set_trace()
        # D_min = D_min.reshape((subdiv[0], D.shape[1]))
        D_min = D_min
    elif distortion_model == 'polynomial' or distortion_model == 'polynomial_woodsc':
        fov = torch.tensor(3.31613).cuda()
        D_min = polynomial_distoriton(subdiv[0], D, fov, 1.0)
        # D_min = D_min.transpose(0, 1)
        D_min = D_min
      # profiler.print()
    # import pdb;pdb.set_trace()
    u, v = DA_grid(D_min, (subdiv[0]*2,subdiv[1]*2))

    return u*max_radius, v*max_radius


def DA_grid(Dmin, grid_size):
    # Set the tolerance and calculate x_major
    # import pdb;pdb.set_trace()
    grid_size = grid_size[0] + 1
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    phi = torch.atan2(yy, xx)
    tolerance = 1e-8

    x_major = torch.isclose(xx**2, yy**2, atol=tolerance) | (xx**2 > yy**2)

    ##############################
    # Create meshgrid of indices for a single grid (same for all batches)
    i = torch.arange(grid_size).unsqueeze(1).repeat(1, grid_size)
    j = torch.arange(grid_size).unsqueeze(0).repeat(grid_size, 1)
    batch_size = Dmin.size(0)
    # grid_size = 7

    # Add batch dimension to `i` and `j` by unsqueezing and repeating for all batches
    i = i.unsqueeze(0).repeat(batch_size, 1, 1)
    j = j.unsqueeze(0).repeat(batch_size, 1, 1)

    # Calculate the absolute distances from the center for i and j
    center = grid_size // 2
    i_dist = torch.abs(i - center)
    j_dist = torch.abs(j - center)

    # Create the output grid for the batch, ensure it's a float tensor to match theta's dtype
    rad_fin = torch.zeros(batch_size, grid_size, grid_size, dtype=torch.float32)
    # Condition for the major axis
    condition = (i_dist**2 > j_dist**2)


    # Assign values from `theta` based on the condition
    batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)
    rad_fin[condition] = Dmin[batch_indices.expand_as(i_dist), i_dist.long()][condition]
    rad_fin[~condition] = Dmin[batch_indices.expand_as(j_dist), j_dist.long()][~condition]


    ###################
    # import pdb;pdb.set_trace()
    # Vectorized computation of output_x and output_y
    # Extend the dimensions of xx, yy, and phi to handle the batch size
    xx_ = xx.unsqueeze(0).expand(batch_size, -1, -1)
    yy_ = yy.unsqueeze(0).expand(batch_size, -1, -1)
    phi_ = phi.unsqueeze(0).expand(batch_size, -1, -1)

    # import pdb;pdb.set_trace()

    # Compute output_x and output_y for the entire batch
    output_x = torch.where(x_major.unsqueeze(0).expand(batch_size, -1, -1),
                           torch.sign(xx_) * rad_fin,
                           torch.sign(xx_) * rad_fin / torch.abs(torch.tan(phi_ + 2 * np.pi)))
    
    output_y = torch.where(x_major.unsqueeze(0).expand(batch_size, -1, -1),
                           torch.sign(yy_) * rad_fin * torch.abs(torch.tan(phi_ + 2 * np.pi)),
                           torch.sign(yy_) * rad_fin)

    # import pdb;pdb.set_trace()
    
    u, v = square_to_disc(output_x, output_y)

    return u, v


#eliptical mapping

def square_to_disc(x, y):
    u = x * torch.sqrt(1 - (y**2 / 2))
    v = y * torch.sqrt(1 - (x**2 / 2))
    return u, v

def disc_to_square(u, v):
    # square = u*u - v*v
#     import pdb;pdb.set_trace()
    square = u * u - v * v
    # inf = torch.tensor(torch.inf)
    # Compute the terms
    term1 = 2 + square + u * 2 * torch.sqrt(torch.tensor(2.0))
    term2 = 2 + square - u * 2 * torch.sqrt(torch.tensor(2.0))
    term3 = 2 - square + v * 2 * torch.sqrt(torch.tensor(2.0))
    term4 = 2 - square - v * 2 * torch.sqrt(torch.tensor(2.0))

    # Replace negative values with 1 before applying the square root
    sqrt_term1 = torch.where(term1 >= 0, torch.sqrt(term1), torch.tensor(0.0))
    sqrt_term2 = torch.where(term2 >= 0, torch.sqrt(term2), torch.tensor(0.0))
    sqrt_term3 = torch.where(term3 >= 0, torch.sqrt(term3), torch.tensor(0.0))
    sqrt_term4 = torch.where(term4 >= 0, torch.sqrt(term4), torch.tensor(0.0))

    # Compute the final results
    x = 0.5 * (sqrt_term1 - sqrt_term2)
    y = 0.5 * (sqrt_term3 - sqrt_term4)
    return x, y




def theta_from_radius(radius, theta_d_max, D, distortion_model):


    def rad_poly(D, theta):
        # import pdb;pdb.set_trace()
        focal = lambda x: 1/(x * (D[0] * x**0 + D[1] * x**1 + D[2] * x**2 + D[3] * x**3))
        f = focal(theta_d_max).reshape(1, D.shape[1])
        funct = g_inv(theta, theta_d_max)
        # funct = funct.reshape(num_points + 1, 1)
        funct = funct.reshape(1, num_points)
        funct.repeat_interleave(D.shape[1], 0)
        # import pdb;pdb.set_trace()
        # radius = f*funct * (D[0] * funct**0 + D[1] * funct**1 + D[2] * funct**2 + D[3] * funct**3)
        funct_powers = torch.stack([funct**1, funct**2, funct**3, funct**4], dim=0)
        radius = torch.sum(D[:, :, None] * funct_powers, dim=0)  # shape (2, 65)
        # radius = ((torch.cos(theta_max) + xi)/torch.sin(theta_max))*torch.sin(funct)/(torch.cos(funct) + xi)
        # import pdb;pdb.set_trace()
        return f.T*radius

    def rad_sphere(xi, theta):
        
        # k_d = ((torch.cos(torch.tensor(theta_d_max)) + xi)/torch.sin(torch.tensor(theta_d_max))).cuda()
        num_points = theta.shape[1]
        # print(num_points)
        funct = g_inv(theta, theta_d_max)
        radius = ((torch.cos(theta_d_max) + xi)/torch.sin(theta_d_max))*torch.sin(funct)/(torch.cos(funct) + xi)
        return radius
    
    num_points = 10000
    radius = radius.unsqueeze(0)
    if distortion_model == 'spherical':
        # import pdb;pdb.set_trace()
        fov = D[2][0]
        f  = D[1]
        xi = D[0].cpu()
        B = xi.shape[0]
        xi = xi.reshape(B, 1)
        # import pdb;pdb.set_trace()
        theta_values = torch.linspace(0, g(theta_d_max, a=theta_d_max), num_points)
        theta_values = theta_values.unsqueeze(0)
        
        radius_values = rad_sphere(xi, theta_values) ## (5, 1)
    #     radius_values = torch.cat((radius_values, radius_values), dim=1)
        

    elif distortion_model == 'polynomial':
        # import pdb;pdb.set_trace()
        B = D.shape[1]
        theta_values = torch.linspace(0, g(theta_d_max, a=theta_d_max), num_points)
        radius_values = rad_poly(D, theta_values)  ## (5, 1)
        theta_values = theta_values.unsqueeze(0)

    theta_ = torch.zeros(B, radius.shape[1])
    for i in range(B):
        lower_indices = (radius_values[i].unsqueeze(1).unsqueeze(1) <= radius.unsqueeze(0)).sum(dim=0) - 1

        # Get theta values corresponding to the found indices
        theta_result = theta_values[0][lower_indices]
        theta_[i] = theta_result
#     import pdb;pdb.set_trace()
    
    return theta_

# def DA_grid_inv(D, img_size, distortion_model):

#     if distortion_model == 'spherical':
#         fov = D[2][0]
#     elif distortion_model == 'polynomial':
#         fov = torch.tensor(3.31613).cuda()
    
#     theta_d_max = fov/2
#     k = 1 / torch.tensor(3.4386)

#     x = torch.linspace(-1, 1, img_size)
#     y = torch.linspace(-1, 1, img_size)
#     xx, yy = torch.meshgrid(x, y)

#     xx, yy = xx, yy

#     # u_, v_ = square_to_disc(u, v)
#     xx, yy = disc_to_square(xx, yy)

#     xx, yy = xx.cuda(), yy.cuda()

#     tolerance = 1e-8
#     x_major = torch.isclose(xx**2, yy**2, atol=tolerance) | (xx**2 > yy**2)    
    
#     rad = torch.where(x_major, xx, yy)


#     phi = torch.atan2(yy, xx)
#     batch_size  = D.shape[1]
#     # xi = xi.reshape(xi.shape[0], 1)
    
#     rad = rad[:-xx.shape[0]//2, xx.shape[0]//2:]
#     rad = rad.reshape(-1)

#     import pdb;pdb.set_trace()

    
#     fq = theta_from_radius(torch.abs(rad), theta_d_max, D, distortion_model)


#     fq = fq.reshape(batch_size, xx.shape[0]//2, xx.shape[1]//2)   
#     sq =  torch.flip(torch.flip(fq, dims=[0]), (0, 2)) #Q2
#     fh = torch.cat((sq, fq), dim=2)
#     sh = torch.flip(torch.flip(fh, [0, 1]), [0])
#     theta = torch.cat((fh, sh), dim=1)
#     # import pdb;pdb.set_trace()
#     #     import pdb;pdb.set_trace()
#     xx_ = xx.unsqueeze(0).expand(batch_size, -1, -1)
#     yy_ = yy.unsqueeze(0).expand(batch_size, -1, -1)
#     phi_ = phi.unsqueeze(0).expand(batch_size, -1, -1)

#     x_out = torch.where(x_major.unsqueeze(0).expand(batch_size, -1, -1), 
#                         torch.sign(xx_)*k*theta, 
#                         torch.sign(xx_)*k*theta/torch.abs(torch.tan(phi_+2*np.pi)))
#     y_out = torch.where(x_major.unsqueeze(0).expand(batch_size, -1, -1), 
#                         torch.sign(yy_)*k*theta*torch.abs(torch.tan(phi_+2*np.pi)), 
#                         torch.sign(yy_)*k*theta)
    
#     return x_out, y_out

def DA_grid_inv(D, img_size, distortion_model):

    grid_size = img_size + 1
    if distortion_model == 'spherical':
        fov = D[2][0].cpu()
    elif distortion_model == 'polynomial':
        fov = torch.tensor(3.31613).cuda()
    
    theta_d_max = fov/2
    k = 1 / torch.tensor(3.4386)

    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # u_, v_ = square_to_disc(u, v)
    xx, yy = disc_to_square(xx, yy)

    tolerance = 1e-8
    x_major = torch.isclose(xx**2, yy**2, atol=tolerance) | (xx**2 > yy**2)    
    
    rad = torch.where(x_major, xx, yy)


    phi = torch.atan2(yy, xx)
    batch_size  = D.shape[1]
    # xi = xi.reshape(xi.shape[0], 1)
    # import pdb;pdb.set_trace()
    

    rad = rad[xx.shape[0]//2-1:-xx.shape[0]//2, xx.shape[0]//2:]

    # im[:, 3:-4, 4:]
    rad = rad.reshape(-1)
    # rad = torch.cat((torch.tensor([0]).cuda(), rad), dim=0)
    rad[0] = 0

    
    theta = theta_from_radius(torch.abs(rad), theta_d_max, D, distortion_model)


    ##############################
    grid_size = img_size + 1
    # Create meshgrid of indices for a single grid (same for all batches)
    i = torch.arange(grid_size).unsqueeze(1).repeat(1, grid_size)
    j = torch.arange(grid_size).unsqueeze(0).repeat(grid_size, 1)
    batch_size = theta.size(0)
    # grid_size = 7

    # Add batch dimension to `i` and `j` by unsqueezing and repeating for all batches
    i = i.unsqueeze(0).repeat(batch_size, 1, 1)
    j = j.unsqueeze(0).repeat(batch_size, 1, 1)

    # Calculate the absolute distances from the center for i and j
    center = grid_size // 2
    i_dist = torch.abs(i - center)
    j_dist = torch.abs(j - center)

    # Create the output grid for the batch, ensure it's a float tensor to match theta's dtype
    theta_fin = torch.zeros(batch_size, grid_size, grid_size, dtype=torch.float32)

    # Condition for the major axis
    condition = (i_dist**2 > j_dist**2)

    # Assign values from `theta` based on the condition
    batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)
    theta_fin[condition] = theta[batch_indices.expand_as(i_dist), i_dist.long()][condition]
    theta_fin[~condition] = theta[batch_indices.expand_as(j_dist), j_dist.long()][~condition]


    ###################

    
    xx_ = xx.unsqueeze(0).expand(batch_size, -1, -1)
    yy_ = yy.unsqueeze(0).expand(batch_size, -1, -1)
    phi_ = phi.unsqueeze(0).expand(batch_size, -1, -1)

    # import pdb;pdb.set._trace()
    x_out = torch.where(x_major.unsqueeze(0).expand(batch_size, -1, -1), 
                        torch.sign(xx_)*k*theta_fin, 
                        torch.sign(xx_)*k*theta_fin/torch.abs(torch.tan(phi_+2*np.pi)))
    y_out = torch.where(x_major.unsqueeze(0).expand(batch_size, -1, -1), 
                        torch.sign(yy_)*k*theta_fin*torch.abs(torch.tan(phi_+2*np.pi)), 
                        torch.sign(yy_)*k*theta_fin)
    
    x_out, y_out = x_out.cuda(), y_out.cuda()
    
    return x_out, y_out


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


def fish2world(u,v, dist):

    f, xi, H = dist

    r= np.sqrt((u-0.5)**2+(v-0.5)**2)
    valid = r <= 1 #np.ones(r.shape, dtype='bool') #r <= 1

    u0, v0 = H//2, H//2
    u = (u*H - u0) / f
    v = (v*H - v0) / f

    omega= (xi + np.sqrt(1+(1-xi**2)*(u**2+v**2))) / (u**2+ v**2 +1)
    x = omega * u
    y= omega * v 
    z= omega - xi


    #breakpoint()
    return x,y,z

def world2cube(x, y, z):
    # world -> cube
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))
    z = np.atleast_1d(np.asarray(z))
    u = np.zeros(x.shape)
    v = np.zeros(x.shape)

    # forward
    indForward = np.nonzero(
        land(land(z <= 0, z <= -np.abs(x)), z <= -np.abs(y)))
    u[indForward] = 1.5 - 0.5 * x[indForward] / z[indForward]
    v[indForward] = 1.5 + 0.5 * y[indForward] / z[indForward]

    # backward
    indBackward = np.nonzero(
        land(land(z >= 0,  z >= np.abs(x)),  z >= np.abs(y)))
    u[indBackward] = 1.5 + 0.5 * x[indBackward] / z[indBackward]
    v[indBackward] = 3.5 + 0.5 * y[indBackward] / z[indBackward]

    # down
    indDown = np.nonzero(
        land(land(y <= 0,  y <= -np.abs(x)),  y <= -np.abs(z)))
    u[indDown] = 1.5 - 0.5 * x[indDown] / y[indDown]
    v[indDown] = 2.5 - 0.5 * z[indDown] / y[indDown]

    # up
    indUp = np.nonzero(land(land(y >= 0,  y >= np.abs(x)),  y >= np.abs(z)))
    u[indUp] = 1.5 + 0.5 * x[indUp] / y[indUp]
    v[indUp] = 0.5 - 0.5 * z[indUp] / y[indUp]

    # left
    indLeft = np.nonzero(
        land(land(x <= 0,  x <= -np.abs(y)),  x <= -np.abs(z)))
    u[indLeft] = 0.5 + 0.5 * z[indLeft] / x[indLeft]
    v[indLeft] = 1.5 + 0.5 * y[indLeft] / x[indLeft]

    # right
    indRight = np.nonzero(land(land(x >= 0,  x >= np.abs(y)),  x >= np.abs(z)))
    u[indRight] = 2.5 + 0.5 * z[indRight] / x[indRight]
    v[indRight] = 1.5 - 0.5 * y[indRight] / x[indRight]

    # bring back in the [0,1] intervals
    u = u / 3.
    v = v / 4.

    if u.size == 1:
        return u.item(), v.item()

    return u, v
def imageCoordinates(h, w):
    """Returns the (u, v) coordinates for each pixel center."""
    cols = np.linspace(0, 1, w*2 + 1)
    rows = np.linspace(0, 1, h*2 + 1)

    cols = cols[1::2]

    rows = rows[1::2]

    return [d.astype('float32') for d in np.meshgrid(cols, rows)]

def interpolate(h, w, u, v):
    u = u.clone()
    v = v.clone()

    # To avoid displacement due to the padding
    u += 0.5 / w
    v += 0.5 / h

    # Normalize u and v to [-1, 1] for grid_sample
    u = 2 * u - 1
    v = 2 * v - 1

    grid = torch.stack((u, v), dim=-1)

    return grid

def cube_inv_grid(B, basedim, dist):

    h = 4*basedim
    w = 3*basedim

    fov = dist[2][0]
    f  = dist[1]
    xi = dist[0]
    grid_ = torch.zeros(B, h, w, 2)
    B = xi.shape[0]
    for i in range(B):
        # import pdb;pdb.set_trace()
        f_ = f[i].cpu().numpy()
        xi_ = xi[i].cpu().numpy()
        grid_x, grid_y = imageCoordinates(h, w)
        dx, dy, dz = fish2world(grid_x, grid_y, [f_,xi_, 128])
        u, v = world2cube(dx, dy, dz)
        grid = interpolate(h, w, torch.tensor(u), torch.tensor(v))
        grid_[i] = grid
    
    return grid_


def cube_inv(cube, dist):


# Assume cube is a tensor with shape (2, 128, 128, 3)
    # import pdb;pdb.set_trace()
    B, C, H, W = cube.shape
    basedim = cube.shape[-1]//2

    

    # Create tensors for re_cubemap and new_cube
    re_cubemap = torch.zeros((B, C, 4*basedim, 3*basedim), dtype=cube.dtype).cuda()
    cubemap = torch.zeros((B, C, 3*basedim, 3*basedim), dtype=cube.dtype).cuda()

    # Assign cube to the center of cubemap
    cubemap[:, :, basedim//2:-basedim//2, basedim//2:-basedim//2] = cube

    # Bottom face
    re_cubemap[:, :, 2*basedim:3*basedim, basedim:2*basedim] = cubemap[:, :, 0:basedim, basedim:2*basedim]

    # Left face (flipping upside-down and left-right)
    re_cubemap[:, :, basedim:2*basedim, 0:basedim] = torch.flip(cubemap[:, :, basedim:2*basedim, 0:basedim], dims=[2, 3])

    # Back face
    re_cubemap[:, :, 3*basedim:4*basedim, basedim:2*basedim] = cubemap[:, :, basedim:2*basedim, basedim:2*basedim]

    # Right face (flipping upside-down and left-right)
    re_cubemap[:, :, basedim:2*basedim, 2*basedim:3*basedim] = torch.flip(cubemap[:, :, basedim:2*basedim, 2*basedim:3*basedim], dims=[2, 3])

    # Front face
    re_cubemap[:, :, 0:basedim, basedim:2*basedim] = cubemap[:, :, 2*basedim:3*basedim, basedim:2*basedim]

    # im = interpolate(new_cube[0], torch.tensor(u), torch.tensor(v))
    grid_ = cube_inv_grid(B, basedim, dist)
    grid_ = grid_.cuda()

    # breakpoint()
    interpolated_img = F.grid_sample(re_cubemap, grid_, mode='bilinear', align_corners=True)

    resized_tensor = F.interpolate(interpolated_img, size=(H, W), mode='bilinear', align_corners=True)

    return resized_tensor
