#!/bin/bash 

python -m torch.distributed.launch \
--nproc_per_node 3 \
--master_port 12345  main.py \
--cfg configs/swin/all_distorted_swin_small_patch2_window4_64.yaml \
--data-path /home-local2/akath.extra.nobkp/all-distorted-full-imagenet \
--batch-size 128
