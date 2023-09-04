CONFIG=one_distortion_swin_small_patch2_window4_64_gp2.yaml

export WANDB_MODE="disabled"

python -m torch.distributed.launch --nproc_per_node=3 main.py --eval    \
--resume /home-local2/akath.extra.nobkp/output/gp2_10_5/ckpt_epoch_90.pth \
--cfg configs/swin/$CONFIG                                              \
--data-path /home-local2/akath.extra.nobkp/imagenet_2010      \
--batch-size 128
