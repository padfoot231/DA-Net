# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import itertools
from matplotlib.pyplot import title
import torch
import cv2
import numpy as np
import torch.distributed as dist
from torch._six import inf


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


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

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


def distort_image(img: np.ndarray, D: list[float], shift: tuple[float, float]=(0.0, 0.0)) -> np.ndarray:
    """Distort an image using a fisheye distortion model
    Args:
        img (np.ndarray): the image to distort
        alpha (float): fov angle (radians)
        D (list[float]): a list containing the k1, k2, k3 and k4 parameters
        shift (tuple[float, float]): x and y shift (respectively)

    Returns:
        np.ndarray: the distorted image
    """
    height, width, _ = img.shape
    center = [height//2, width//2]

    # Image coordinates
    map_x, map_y = np.mgrid[0:height, 0:width].astype(np.float32)

    # Center coordinate system
    if height % 2 == 0:
        center[0] -= 0.5
    if width % 2 == 0:
        center[1] -= 0.5

    map_x -= center[0]
    map_y -= center[1]

    # (shift and) convert to polar coordinates
    r = np.sqrt((map_x + shift[0])**2 + (map_y + shift[1])**2)
    theta = (r * (np.pi / 2)) / height

    # Compute fisheye distortion with equidistant projection
    theta_d = theta * (1 + D[0]*theta**2 + D[1]*theta**4 + D[2]*theta**6 + D[3]*theta**8)

    # Scale so that image always fits the original size
    f = map_y.max() / theta_d[int(center[0]), 0]
    r_d = f * theta_d

    # Compute distorted map and rotate
    map_xd = (r_d / r) * map_x + center[0]
    map_yd = (r_d / r) * map_y + center[1]

    # Distort
    distorted_image = cv2.remap(
        img, map_yd, map_xd,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return distorted_image


def distort_batch(x: torch.Tensor, alpha: float, D: list[float], shift: tuple[float, float]=(0.0, 0.0), phi: float=0.0) -> torch.Tensor:
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


def get_sample_locations(alpha, phi, dmin, ds, n_azimuth, n_radius, img_size, radius_buffer=0, azimuth_buffer=0):
    """Get the sample locations in a given radius and azimuth range
    
    Args:
        alpha (float): width of the azimuth range (radians)
        phi (float): phase shift of the azimuth range  (radians)
        dmin (float): minimum radius of the patch (pixels)
        ds (float): distance between the inner and outer arcs of the patch (pixels)
        n_azimuth (int): number of azimuth samples
        n_radius (int): number of radius samples
        img_size (tuple): the size of the image (width, height)
        radius_buffer (int, optional): radius buffer (pixels). Defaults to 0.
        azimuth_buffer (int, optional): azimuth buffer (radians). Defaults to 0.
    
    Returns:
        list[tuple[int, int]]: list of sample locations
    """

    # Compute center of the image to shift the samples later
    center = [img_size[0]//2, img_size[1]//2]
    if img_size[0] % 2 == 0:
        center[0] -= 0.5
    if img_size[1] % 2 == 0:
        center[1] -= 0.5

    # Sweep start and end
    r_start = dmin + ds - radius_buffer
    r_end = dmin + radius_buffer
    alpha_start = phi + azimuth_buffer
    alpha_end = alpha + phi - azimuth_buffer

    # Get the sample locations
    sample_points = []
    for radius in np.linspace(r_start, r_end, n_radius):
        for angle in np.linspace(alpha_start, alpha_end, n_azimuth):
            x = radius * np.cos(angle) + center[0]
            y = radius * np.sin(angle) + center[1]
            sample_points.append((x, y))
    
    return sample_points


def get_sample_params_from_subdiv(subdiv, n_radius, n_azimuth, img_size, radius_buffer=0, azimuth_buffer=0):
    """Generate the required parameters to sample every patch based on the subdivison

    Args:
        subdiv (int): the number of subdivisions for which we need to create the samples
        n_radius (int): number of radius samples
        n_azimuth (int): number of azimuth samples
        img_size (tuple): the size of the image

    Returns:
        list[dict]: the list of parameters to sample every patch
    """
    max_radius = img_size[0]//2
    angle_width = 2*np.pi / 2**(subdiv+1)

    alpha = angle_width
    ds = max_radius / 2**(subdiv-1)
    
    dmin_step = -max_radius / 2**(subdiv-1)
    dmin_start = max_radius + dmin_step
    dmin_end = 0

    phi_step = -angle_width
    phi_start = 0
    phi_end = -2*np.pi
    
    dmin_list = np.arange(dmin_start, dmin_end, dmin_step)
    phi_list = np.arange(phi_start, phi_end, phi_step)

    dmin_list = np.append(dmin_list, 0)

    return [{
        "alpha": alpha, "phi": phi, "dmin": dmin, "ds": ds, "n_azimuth": n_azimuth,
        "n_radius": n_radius, "img_size": img_size, "radius_buffer": radius_buffer,
        "azimuth_buffer": azimuth_buffer 
        } for phi, dmin in itertools.product(phi_list, dmin_list)]


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
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Sampling locations")
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']

    subdiv = 3
    n_radius = 8
    n_azimuth = 8
    img_size = (64, 64)
    radius_buffer = img_size[0] / (n_radius * (2**(subdiv-1)) * 2 * 2)
    azimuth_buffer = 2*np.pi / (n_azimuth * (2**(subdiv+1)) * 2)

    params = get_sample_params_from_subdiv(
        subdiv=subdiv,
        img_size=img_size,
        n_radius=n_radius,
        n_azimuth=n_azimuth,
        radius_buffer=radius_buffer,
        azimuth_buffer=azimuth_buffer
    )

    for i in range(len(params)):
        sample_locations = get_sample_locations(**params[i])
        ax.scatter(*zip(*sample_locations), color=colors[i%len(colors)], s=6)
    plt.show()