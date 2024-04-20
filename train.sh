#!/bin/bash 
export WANDB_MODE="disabled"
export NCCL_BLOCKING_WAIT=1 
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1


python -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12345  main.py \
--cfg configs/swin/CVRG_20_5.yaml \
--output /home/prongs/scratch/Radial-unet \
--data-path $SLURM_TMPDIR/data/CVRG-Pano \
--fov 175.0 \
--xi 0.0 \
--batch-size 8


# --cfg configs/swin/CVRG_den_unet.yaml \
