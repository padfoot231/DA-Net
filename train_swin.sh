#!/bin/bash 
export WANDB_MODE="disabled"


python -m torch.distributed.launch \
--nproc_per_node 4 \
--master_port 12345  main.py \
--cfg configs/swin/CVRG_swin.yaml \
--output /home/prongs/scratch/Radial-unet \
--data-path  $SLURM_TMPDIR/data/CVRG-Pano \
--fov 175.0 \
--xi 0.0 \
--batch-size 8
