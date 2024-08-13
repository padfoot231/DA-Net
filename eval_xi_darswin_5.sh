#!/bin/bash 
export WANDB_MODE="disabled"

for i in $(seq 0.0 0.05 0.9)
do 
    echo $i
    python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12345  main.py --eval \
    --cfg configs/swin/stanford_5.yaml \
    --data-path  $SLURM_TMPDIR/data_new/semantic2d3d \
    --resume /home/prongs/scratch/Radial-unet-stanford-recon-5-unet-test/stanford/default/ckpt_epoch_400.pth \
    --output /home/prongs/scratch/Radial-unet-stanford-recon/ \
    --fov 175.0 \
    --xi $i \
    --batch-size 4
done


# /home/prongs/scratch/Radial-unet/swin/default/ckpt_epoch_1800.pth
