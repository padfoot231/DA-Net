#!/bin/bash 
export WANDB_MODE="disabled"
# export NCCL_BLOCKING_WAIT=1 
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1


python -m torch.distributed.launch \
--nproc_per_node 3 \
--master_port 12345  main.py \
--cfg configs/swin/stanford_3.yaml \
--output /home-local2/akath.extra.nobkp/rad_3 \
--data-path  /home-local2/akath.extra.nobkp/semantic2d3d \
--fov 175.0 \
--xi 0.0 \
--batch-size 4


# --resume /home/prongs/scratch/Radial-unet_stan/CVRG_25_4/default/ckpt_epoch_80.pth \
# --cfg configs/swin/CVRG_den_unet.yaml \
