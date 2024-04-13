#!/bin/bash 
export WANDB_MODE="disabled"


python -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12345  main.py --eval \
--cfg configs/swin/CVRG_swin.yaml \
--resume /home/prongs/scratch/Radial-unet/swin/default/ckpt_epoch_1810.pth \
--data-path  $SLURM_TMPDIR/data/CVRG-Pano \
--fov 170.0 \
--xi 0.0 \
--batch-size 8