#!/bin/bash 
export WANDB_MODE="disabled"

for i in $(seq 0.0 0.05 0.9)
do 
    echo $i
    python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12345  main.py --eval \
    --cfg configs/swin/stanford_3_darunet.yaml \
    --data-path  $SLURM_TMPDIR/data_new/semantic2d3d \
    --resume /home/prongs/scratch/Unet/DarUnet_3/default/ckpt_epoch_best.pth \
    --output /home/prongs/scratch/DarUnet_3/ \
    --fov 175.0 \
    --xi $i \
    --batch-size 4
done


# /home/prongs/scratch/Radial-unet/swin/default/ckpt_epoch_1800.pth
