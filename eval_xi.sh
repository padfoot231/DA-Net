#!/bin/bash 
export WANDB_MODE="disabled"

for i in $(seq 0.0 0.1 0.9)
do 
    echo $i
    python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --master_port 12345  main.py --eval \
    --cfg configs/swin/CVRG_25_4.yaml \
    --data-path  $SLURM_TMPDIR/data/semantic2d3d \
    --resume  /home/prongs/scratch/Radial-unet_stan/CVRG_25_4/default/ckpt_epoch_80.pth \
    --output /home/prongs/scratch/Radial-eval \
    --fov 170.0 \
    --xi $i \
    --batch-size 8
done


# /home/prongs/scratch/Radial-unet/swin/default/ckpt_epoch_1800.pth
