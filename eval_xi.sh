#!/bin/bash 
export WANDB_MODE="disabled"

for i in $(seq 0.0 0.1 0.9)
do 
    echo $i
    python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12345  main.py --eval \
    --cfg configs/swin/CVRG_den_unet.yaml \
    --data-path  $SLURM_TMPDIR/data/CVRG-Pano \
    --resume /home/prongs/scratch/Radial-unet/woodscape_den_unet_20_5_more/default/ckpt_epoch_1600.pth \
    --output /home/prongs/scratch/Radial-eval \
    --fov 170.0 \
    --xi $i \
    --batch-size 1
done


# /home/prongs/scratch/Radial-unet/swin/default/ckpt_epoch_1800.pth
