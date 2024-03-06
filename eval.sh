#!/bin/bash 
export WANDB_MODE="disabled"

for fov in $(seq 90.0 20.0 180)
do 
    echo $fov
    python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12345  main.py  --eval  \
    --cfg configs/swin/woodscapes_pretrain1.yaml \
    --output /home/prongs/scratch/Radial-unet \
    --resume /home/prongs/scratch/Radial-unet/woodscapes_pretrain1/default/ckpt_epoch_1020.pth \
    --data-path /home/prongs/scratch/CVRG-Pano \
    --fov $fov \
    --xi 0.0 \
    --batch-size 8
done
