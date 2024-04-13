#!/bin/bash 
export WANDB_MODE="disabled"
export NCCL_BLOCKING_WAIT=1 
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

python -m torch.distributed.launch \
--nproc_per_node 4 \
--master_port 12345  main.py \
--cfg configs/swin/woodscape_den_unet_sgd_angular_r_theta.yaml \
--output /home/prongs/scratch/Radial-unet \
--data-path $SLURM_TMPDIR/data/woodscape \
--fov 170.0 \
--xi 0.0 \
--batch-size 8