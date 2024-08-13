# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_angular_denoiser_unet import swin_transformer_angular_denoiser_unet
from .swin_unet import SwinTransformerSys
from .swin_unet_disc import Radial_curve_cds
from .swin_transformer_angular_denoiser_upsample import swin_transformer_angular_denoiser_upsample
from .darswin_swin_unet import DarSwin_unet



class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        model_type = config.MODEL.TYPE
        # breakpoint()
        if model_type == 'swin_upsample':
            self.swin_unet = swin_transformer_angular_denoiser_upsample(img_size=config.DATA.IMG_SIZE,
                                    radius_cuts=config.MODEL.SWIN.RADIUS_CUTS, 
                                    azimuth_cuts=config.MODEL.SWIN.AZIMUTH_CUTS,
                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                    depths=config.MODEL.SWIN.DEPTHS,
                                    num_heads=config.MODEL.SWIN.NUM_HEADS,
                                    window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                    qk_scale=config.MODEL.SWIN.QK_SCALE,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWIN.APE,
                                    patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                    distortion_model = config.MODEL.DISTORTION, 
                                    n_radius = config.MODEL.NRADIUS, 
                                    n_azimuth = config.MODEL.NAZIMUTH) 
        elif model_type == 'swin_ang_den_unet':
            self.swin_unet = swin_transformer_angular_denoiser_unet(img_size=config.DATA.IMG_SIZE,
                                    radius_cuts=config.MODEL.SWIN.RADIUS_CUTS, 
                                    azimuth_cuts=config.MODEL.SWIN.AZIMUTH_CUTS,
                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                    depths=config.MODEL.SWIN.DEPTHS,
                                    num_heads=config.MODEL.SWIN.NUM_HEADS,
                                    window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                    qk_scale=config.MODEL.SWIN.QK_SCALE,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWIN.APE,
                                    patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                    distortion_model = config.MODEL.DISTORTION, 
                                    n_radius = config.MODEL.NRADIUS, 
                                    n_azimuth = config.MODEL.NAZIMUTH) 
        elif model_type == 'swin_unet':
            self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN.EMBED_DIM,
                        depths=config.MODEL.SWIN.DEPTHS,
                        num_heads=config.MODEL.SWIN.NUM_HEADS,
                        window_size=config.MODEL.SWIN.WINDOW_SIZE_GRID,
                        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                        qk_scale=config.MODEL.SWIN.QK_SCALE,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN.APE,
                        patch_norm=config.MODEL.SWIN.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        elif model_type == 'swin_unet_disc':
            self.swin_unet = Radial_curve_cds(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN.PATCH_SIZE,
                        n_radius = config.MODEL.NRADIUS, 
                        in_chans=config.MODEL.SWIN.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN.EMBED_DIM,
                        depths=config.MODEL.SWIN.DEPTHS,
                        num_heads=config.MODEL.SWIN.NUM_HEADS,
                        window_size=config.MODEL.SWIN.WINDOW_SIZE_GRID,
                        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                        qk_scale=config.MODEL.SWIN.QK_SCALE,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN.APE,
                        patch_norm=config.MODEL.SWIN.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        elif model_type == 'darswin_unet':
            self.swin_unet = DarSwin_unet(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN.EMBED_DIM,
                        depths=config.MODEL.SWIN.DEPTHS,
                        num_heads=config.MODEL.SWIN.NUM_HEADS,
                        window_size=config.MODEL.SWIN.WINDOW_SIZE_GRID,
                        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                        qk_scale=config.MODEL.SWIN.QK_SCALE,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN.APE,
                        patch_norm=config.MODEL.SWIN.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        distortion_model = config.MODEL.DISTORTION, 
                        radius_cuts=config.MODEL.SWIN.RADIUS_CUTS, 
                        azimuth_cuts=config.MODEL.SWIN.AZIMUTH_CUTS,
                        n_radius = config.MODEL.NRADIUS, 
                        n_azimuth = config.MODEL.NAZIMUTH) 
        else:
            raise NotImplementedError(f"Unkown model: {model_type}")

    def forward(self, x, dist,class_batch):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x,dist,class_batch)
        return logits

    def load_from(self, config):
        # breakpoint()
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")
            # breakpoint()
            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]
            # breakpoint()
            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
