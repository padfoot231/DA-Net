#!/bin/bash 
export WANDB_MODE="disabled"


python -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12345  main.py --eval \
--cfg configs/swin/CVRG_25_4_upsample.yaml \
--resume /home/prongs/scratch/Radial-unet_sem/CVRG_25_4_upsample_curve/default/ckpt_epoch_60.pth \
--data-path  $SLURM_TMPDIR/data/semantic2d3d \
--fov 175.0 \
--xi 0.0 \
--batch-size 8