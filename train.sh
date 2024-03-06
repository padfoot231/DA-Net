#!/bin/bash 
export WANDB_MODE="disabled"


python -m torch.distributed.launch \
--nproc_per_node 4 \
--master_port 12345  main.py \
--cfg configs/swin/woodscape_den_unet.yaml \
--output /home/prongs/scratch/Radial-unet \
--data-path $SLURM_TMPDIR/data/woodscapes \
--fov 90.0 \
--xi 0.0 \
--batch-size 4