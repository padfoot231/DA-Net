CONFIG=one_distortion_swin_small_patch2_window4_64_gp1.yaml

export WANDB_MODE="disabled"

python -m torch.distributed.launch --nproc_per_node=1 main.py --eval    \
--resume output/one_distortion_swin_small_patch2_window4_64_gp1/default/ckpt_epoch_35.pth \
--cfg configs/swin/$CONFIG                                              \
--data-path /home-local2/akath.extra.nobkp/imagenet_2010 \                                                      \
--batch-size 128

