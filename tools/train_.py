# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
# from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss  # 기존 loss
from utils.criterion import BondaryLoss # 삭제 X
from utils.function import train, validate  # train, validate 함수는 그대로 사용
from utils.utils import create_logger, FullModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/nail_config.yaml",  # nail_config.yaml 사용
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0

    imgnet = 'imagenet' in config.MODEL.PRETRAINED  # ImageNet pretrained 여부
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)

    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # Data Loaders
    train_dataset = datasets.NailDataset(config, is_train=True, is_extra=False)
    extra_train_dataset = datasets.NailDataset(config, is_train=True, is_extra=True) # unlabeled dataset
    test_dataset = datasets.NailDataset(config, is_train=False, is_extra=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False, # pin_memory=True
        drop_last=True)
    
    extra_train_loader = torch.utils.data.DataLoader( # unlabeled data loader
        extra_train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,  # pin_memory=True
        drop_last=True)


    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False) # pin_memory=True

    # Loss function
    # criterion = nn.CrossEntropyLoss(ignore_index=config.TRAIN.IGNORE_LABEL)  # 기존 loss
    criterion = nn.BCEWithLogitsLoss()  # 2진 분류 Loss
    bd_criterion = BondaryLoss()

    # model = FullModel(model, criterion)  # Loss와 모델을 묶음. -> P, D branch사용 x
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # Optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV)
    elif config.TRAIN.OPTIMIZER == 'adam':  # Adam optimizer 사용
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.TRAIN.LR,
                                     weight_decay=config.TRAIN.WD)
    else:
        raise ValueError('Only Support SGD and Adam optimizer')

    epoch_iters = int(train_dataset.__len__() / batch_size)  # 배치가 drop_last=True이므로 재계산

    best_mIoU = 0  # best_mIoU 변수 초기화
    last_epoch = 0
    if config.TRAIN.RESUME:  # Resume 기능
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = end_epoch * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        # Train
        train(config, epoch, end_epoch, epoch_iters, config.TRAIN.LR, num_iters,
              train_loader, extra_train_loader, optimizer, model, criterion, writer_dict)  # 인자 수정

        # Validation (5 epoch마다 또는 마지막 100 epoch)
        if (epoch % 5 == 0) or (epoch >= end_epoch - 100):
            valid_loss, mean_IoU, IoU_array = validate(config, test_loader, model, writer_dict)

            # Save checkpoint
            logger.info('=> saving checkpoint to {}'.format(final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch + 1,
                'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

            # Save best model
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'best.pt')) # best.pt

            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(valid_loss, mean_IoU, best_mIoU)
            logging.info(msg)
            logging.info(IoU_array)


    # 최종 모델 저장
    torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'final_state.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % np.int((end - start) / 3600))
    logger.info('Done')


if __name__ == '__main__':
    main()