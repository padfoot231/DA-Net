#!/bin/bash 
export WANDB_MODE="disabled"


python -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12345  main.py --eval \
--cfg configs/swin/DDM.yaml \
--output /home-local2/akath.extra.nobkp/Radial-unet \
--data-path /home-local2/akath.extra.nobkp/CVRG-Pano \
--resume /home-local2/akath.extra.nobkp/Radial-unet/DDM/default/ckpt_epoch_670.pth \
--fov 170.0 \
--xi 0.5 \
--batch-size 1