#!/bin/bash 
export WANDB_MODE="disabled"
# export NCCL_BLOCKING_WAIT=1 
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

for i in $(seq 0.0 0.05 0.9)
do 
    python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12346  main_cubemap.py \
    --cfg configs/swin/stanford_unet_cubemap.yaml \
    --output /home/prongs/scratch/Unet \
    --data-path  $SLURM_TMPDIR/data_new/semantic2d3d \
    --resume /home/prongs/scratch/Unet/Unet/default/ckpt_epoch_best.pth \
    --fov 175.0 \
    --xi $i \
    --batch-size 8
done



