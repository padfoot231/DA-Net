#!/bin/bash 
export WANDB_MODE="disabled"
export NCCL_BLOCKING_WAIT=1 
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

for i in $(seq 0.0 0.05 0.9)
do 
    python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12345  main.py --eval \
    --cfg configs/swin/CVRG_swin.yaml \
    --data-path  $SLURM_TMPDIR/data_new/semantic2d3d \
    --resume /home/prongs/scratch/Swin-unet-disc-sem-new/swin_new/default/ckpt_epoch_400.pth \
    --output /home/prongs/scratch/Swin-unet-disc_sem  \
    --fov 175.0 \
    --xi $i \
    --batch-size 4
done



