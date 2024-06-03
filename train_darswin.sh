#!/bin/bash 
export WANDB_MODE="disabled"
# export NCCL_BLOCKING_WAIT=1 
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1


python -m torch.distributed.launch \
--nproc_per_node 4 \
--master_port 12345  main.py \
--cfg configs/swin/CVRG_25_4.yaml \
--output /home/prongs/scratch/Radial-unet \
--data-path  $SLURM_TMPDIR/data/CVRG-Pano \
--fov 175.0 \
--xi 0.0 \
--batch-size 8

# --resume /home/prongs/scratch/Radial-unet_stan/CVRG_25_4/default/ckpt_epoch_80.pth \
# --cfg configs/swin/CVRG_den_unet.yaml \
