#!/bin/bash 
export WANDB_MODE="disabled"


python -m torch.distributed.launch \
--nproc_per_node 3 \
--master_port 12345  main.py \
--cfg configs/swin/woodscapes_pretrain.yaml \
--output /home-local2/akath.extra.nobkp/Radial-unet \
--data-path /home-local2/akath.extra.nobkp/cityscapes \
--batch-size 8
