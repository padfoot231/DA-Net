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
from .swin_unet import SwinTransformerSys
from .swin_unet_disc import Radial_curve_cds
from .swin_unet_disc_label import Radial_curve_cds_lab
from .darswin_swin_unet import DarSwin_unet
from .unet_model import UNet
from .unet_model_cube import UNet_cube
from .dar_unet_model import DarUNet
from .swin_unet_cube import SwinTransformerSys_Cube
from .ddr_net_cube import DDRNet_cube
from .ddr_net_da import DDRNet_DA
from .ddr_net import DDRNet


class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        model_type = config.MODEL.TYPE
        if model_type == 'swin_unet':
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
        elif model_type == 'swin_unet_cube':
            self.swin_unet = SwinTransformerSys_Cube(img_size=config.DATA.IMG_SIZE,
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
            self.swin_unet = Radial_curve_cds(img_size=config.DATA.IMG_SIZE_WOOD,
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
        elif model_type == 'Unet':
            self.swin_unet = UNet(n_channels=config.MODEL.SWIN.IN_CHANS, n_classes=config.MODEL.NUM_CLASSES)
        elif model_type == 'Unet_cube':
            self.swin_unet = UNet_cube(n_channels=config.MODEL.SWIN.IN_CHANS, n_classes=config.MODEL.NUM_CLASSES)
        elif model_type == 'DarUnet':
            self.swin_unet = DarUNet(n_rad=config.MODEL.NRADIUS, n_channels=config.MODEL.SWIN.IN_CHANS, n_classes=config.MODEL.NUM_CLASSES)
        elif model_type == 'swin_unet_disc_label':
            self.swin_unet = Radial_curve_cds_lab(img_size=config.DATA.IMG_SIZE,
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
        elif model_type == 'ddr_net_dar':
            self.swin_unet = DDRNet_DA(pretrained=False, 
            num_classes=config.MODEL.NUM_CLASSES, 
            n_rad = config.MODEL.NRADIUS, img_size=config.DATA.IMG_SIZE)
        elif model_type == 'ddr_net_cube':
            self.swin_unet = DDRNet_cube(pretrained=False, num_classes=config.MODEL.NUM_CLASSES)
        elif model_type == 'ddr_net':
            self.swin_unet = DDRNet(pretrained=False, num_classes=config.MODEL.NUM_CLASSES)
        else:
            raise NotImplementedError(f"Unkown model: {model_type}")

    def forward(self, x, dist):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x,dist)
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
