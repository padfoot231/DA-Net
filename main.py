# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np
# import wandb
import pickle as pkl

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms as T
from utils import DiceLoss, Evaluator
from models.build import SwinUnet as ViT_seg
from torch import optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt 
from torchmetrics.classification import MulticlassJaccardIndex, MultilabelJaccardIndex


# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.nn.modules.loss import CrossEntropyLoss

from timm.utils import accuracy, AverageMeter

from config import get_config
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

transform = T.ToPILImage()
data_dic = {}

# wandb.init(project="semantic segmantation", entity='padfoot')
# run_name = wandb.run.name
def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--img_size', type=int, help="image size")
    parser.add_argument('--img_size_wood', type=tuple, help="image size")
    parser.add_argument('--num_classes', type=int, help="number of segmentation classes")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--base_lr', type=float , help="base learning rate")
    parser.add_argument('--fov', type=float, default=90.0, help="field of view")
    parser.add_argument('--grp', type=str, default="vlow", help="group of distortion")
    parser.add_argument('--xi', type=float, default=0.0, help="distortion parameter of the image")

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    base_lr = config.TRAIN.BASE_LR
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes) 
    logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model
    # breakpoint()
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    num_classes = config.MODEL.NUM_CLASSES
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    evaluator = Evaluator(config.MODEL.NUM_CLASSES)

    max_miou = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')


    if config.MODEL.RESUME:
        max_miou = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        miou, loss = validate(config, ce_loss, dice_loss, evaluator, data_loader_val, model)
        logger.info(f"Mean iou of the network on the {len(dataset_val)} test images: {miou:.4f}")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        miou, loss = validate(config, ce_loss, dice_loss, evaluator, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {miou:.4f}")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, ce_loss, dice_loss, evaluator, data_loader_train, optimizer, epoch, mixup_fn,lr_scheduler,
                        loss_scaler)

        miou, loss = validate(config, ce_loss, dice_loss, evaluator, data_loader_val, model)
        
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_miou, miou, optimizer, lr_scheduler, loss_scaler,
                            logger)
        # acc1_test, acc5_test, loss_test = test(config, data_loader_test, model)
        logger.info(f"Mean IOU of the network on the {len(dataset_val)} test images: {miou:.4f}%")
        max_miou = max(max_miou, miou)
        logger.info(f'Max miou: {max_miou:.4f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, ce_loss, dice_loss, evaluator, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    metric_perclass = MulticlassJaccardIndex(num_classes=config.MODEL.NUM_CLASSES, average=None).cuda()
    metric = MulticlassJaccardIndex(num_classes=config.MODEL.NUM_CLASSES).cuda()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    miou_meter = AverageMeter()
    iter_num = 0
    max_iterations = config.TRAIN.EPOCHS*num_steps
    start = time.time()
    end = time.time()
    for idx, (samples, targets, grid, mask, one_hot) in enumerate(data_loader):

        ###############
        # breakpoint()
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        grid = grid.cuda(non_blocking=True)   
        mask = mask.cuda(non_blocking=True)
        one_hot = one_hot.cuda(non_blocking=True)
        # breakpoint()
        outputs = model(samples, grid)
        B, _, _, _ = samples.shape
        # breakpoint()
        one_hot = one_hot.transpose(2, 3).transpose(1, 2)
        outputs[:, :, mask[0, 0] == 0] = one_hot[:, :, mask[0, 0] == 0]
        # breakpoint()
        loss_ce = ce_loss(outputs, targets[:].long())
        loss_dice = dice_loss(outputs, targets, softmax=True)
        loss = 0.4 * loss_ce + 0.6 * loss_dice
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        tar = targets.cpu().numpy()
        pred = outputs.argmax(1).cpu().numpy()
        # pred = output.data.cpu().numpy()
        evaluator.add_batch(tar, pred)
        # breakpoint()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        # lr = config.TRAIN.BASE_LR * (1.0 - iter_num / max_iterations) ** 0.9
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        # miou_meter.update(miou.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()
        if (idx) % 100 == 0:
            # image= images[0,...].permute(1,2,0)
            # image*= torch.tensor(std).cuda(cuda_id)
            # image+= torch.tensor(mean).cuda(cuda_id)
            # plt.imsave(save_path+ '/val_img_{}.png'.format(epoch_num), np.clip(image.cpu().numpy(),0,1) )
            label = targets[0].detach().cpu().numpy()
            plt.imsave(config.OUTPUT+ '/train_label_{}.png'.format(idx), label.astype(np.uint8))
            pred= outputs.argmax(1)[0].cpu().detach().numpy()
            plt.imsave(config.OUTPUT+ '/train_pred_{}.png'.format(idx), pred)
        # wandb.log({"loss_train" : loss.item(), "loss_ce":loss_ce.item(), "epoch" : epoch, "grad":norm_meter.val })

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'miou {evaluator.Mean_Intersection_over_Union():.4f} ({miou_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
        # wandb.log({"loss_train" : loss.item(), "loss_ce":loss_ce.item(), "epoch" : epoch, "grad":norm_meter.val, 'lr':lr })

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, ce_loss, dice_loss, evaluator, data_loader, model):
    
    model.eval()

    mean=[0.2151, 0.2235, 0.2283]
    std=[0.2300, 0.2334, 0.2419]

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    miou_meter = AverageMeter()
    # metric_perclass = MulticlassJaccardIndex(num_classes=config.MODEL.NUM_CLASSES, average=None).cuda()
    # metric = MultilabelJaccardIndex(num_classes=config.MODEL.NUM_CLASSES).cuda()
    evaluator.reset()
    end = time.time()
    running_loss = 0
    running_acc1 = 0
    running_acc5 = 0
    for idx, (images, target, grid, mask, one_hot) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        grid = grid.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        one_hot = one_hot.cuda(non_blocking=True)
        # compute output
        # breakpoint()
        # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        output = model(images, grid)
        # breakpoint()
        B, _, _, _ = images.shape
        one_hot = one_hot.transpose(2, 3).transpose(1, 2)
        # import pdb;pdb.set_trace()
        output[:, :, mask[0, 0] == 0] = one_hot[:, :, mask[0, 0] == 0]
        # plt.imshow(output.argmax(1)[0].cpu().numpy())
        # plt.savefig("lab.png")
        # measure accuracy and record loss
        loss_ce = ce_loss(output, target[:].long())
        loss_dice = dice_loss(output, target, softmax=True)
        loss = 0.4 * loss_ce + 0.6 * loss_dice
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # breakpoint() .
        tar = target.cpu().numpy()
        pred = output.argmax(1).cpu().numpy()
        # pred = output.data.cpu().numpy()
        evaluator.add_batch(tar, pred)
        # miou = metric(pred, target)
        # breakpoint()
        # acc1 = reduce_tensor(acc1)
        # acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        # miou_meter.update(miou.item(), target.size(0))
        # acc5_meter.update(acc5.item(), target.size(0))
        # wandb.log({"loss_val" : loss.item(), 
        #             "Acc1" : acc1, 
        #             "Acc5" : acc5})

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if False:
            image= images[0,...].permute(1,2,0)
            image*= torch.tensor(std).cuda()
            image+= torch.tensor(mean).cuda()
            plt.imsave(config.OUTPUT+ '/' + str(args.fov) + '/' + str(args.xi) + '/' + 'val_img_{}.png'.format(idx), np.clip(image.cpu().numpy(),0,1) )
            # + str(config.DATA.XI) + '/' +  
            label = target[0].detach().cpu().numpy()
            plt.imsave(config.OUTPUT+ '/' + str(args.fov) + '/' + str(args.xi) + '/' + 'val_label_{}.png'.format(idx), label.astype(np.uint8))
            pred= output.argmax(1)[0].cpu().detach().numpy()
            plt.imsave(config.OUTPUT+ '/' + str(args.fov) + '/' + str(args.xi) + '/' + 'val_pred_{}.png'.format(idx), pred)
        if False:
            image= images[0,...].permute(1,2,0)
            image*= torch.tensor(std).cuda()
            image+= torch.tensor(mean).cuda()
            plt.imsave(config.OUTPUT+ '/' + str(args.xi) + '/' + 'val_img_{}.png'.format(idx), np.clip(image.cpu().numpy(),0,1) )
            # + str(config.DATA.XI) + '/' +  
            label = target[0].detach().cpu().numpy()
            plt.imsave(config.OUTPUT+ '/' + str(args.xi) + '/' + 'val_label_{}.png'.format(idx), label.astype(np.uint8))
            pred= output.argmax(1)[0].cpu().detach().numpy()
            plt.imsave(config.OUTPUT+ '/' + str(args.xi) + '/' + 'val_pred_{}.png'.format(idx), pred)

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'miou {evaluator.Mean_Intersection_over_Union():.4f} ({miou_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')
    # with open('/home-local2/akath.extra.nobkp/rad_gp4_gp4.pkl', 'wb') as f:
    #     pkl.dump(data_dic, f)
    # logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    # breakpoint()
    miou = evaluator.Mean_Intersection_over_Union()
    # wandb.log({"loss_val" :loss_meter.avg, 
    #         "Acc1" : miou})

    return miou, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _, dist) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images, dist)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images, dist)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()




    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    # linear scale the learning rate according to total batch size, may not be optimal
    # if args.batch_size != 24 and args.batch_size % 6 == 0:
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0 ## change it based on the scheduler performance
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = config.TRAIN.BASE_LR * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    
    config.defrost()
    config.TRAIN.BASE_LR = config.TRAIN.BASE_LR
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
