#!/bin/bash 
export WANDB_MODE="disabled"
# export NCCL_BLOCKING_WAIT=1 
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1


python -m torch.distributed.launch \
--nproc_per_node 4 \
--master_port 12346  main.py \
--cfg configs/swin/stanford_unet_cube.yaml \
--output /home/prongs/scratch/Unet_pe_up \
--data-path  $SLURM_TMPDIR/data_new/semantic2d3d \
--fov 175.0 \
--xi 0.0 \
--batch-size 8

# --resume /home/prongs/scratch/Unet/Unet_cube_new/default/ckpt_epoch_best_new.pth \



# 
# --cfg configs/swin/CVRG_den_unet.yaml \
