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
import torch.nn.functional as F  # For interpolation
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import BondaryLoss  # BoundaryLoss는 그대로 사용
from utils.function import validate  # validate 함수는 그대로 사용
from utils.utils import create_logger
import yaml


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
    return args


def create_teacher(config, imgnet_pretrained):
    """Create a teacher model with the same architecture."""
    teacher_model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet_pretrained)
    return teacher_model

def update_ema_variables(model, ema_model, alpha, global_step):
    """Update teacher model parameters via EMA."""
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha= 1 - alpha)

def get_threshold(pseudo_labels, num_classes):
    """Calculate the threshold for confidence masking."""
    hist = torch.histc(pseudo_labels, bins=num_classes, min=0, max=num_classes - 1)  # 각 class별 개수
    threshold = (hist.sum() / num_classes).int()  # 클래스당 평균 개수
    return threshold, hist

def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # config file load
    with open(args.cfg, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    from yacs.config import CfgNode as CN
    config = CN(config_dict)

    update_config(config, args)

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

    imgnet = 'imagenet' in config.MODEL.PRETRAINED
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)
    # Teacher model initialization
    teacher_model1 = create_teacher(config, imgnet_pretrained=imgnet).cuda()
    teacher_model2 = create_teacher(config, imgnet_pretrained=imgnet).cuda()  # 두번째 teacher

    # Freeze teacher model
    for p in teacher_model1.parameters():
        p.requires_grad = False
    for p in teacher_model2.parameters():  # 두 번째 teacher도 freeze
      p.requires_grad = False


    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # Data Loaders
    train_dataset = datasets.Nail(config, is_train=True, is_extra=False)
    extra_train_dataset = datasets.Nail(config, is_train=True, is_extra=True)
    test_dataset = datasets.Nail(config, is_train=False, is_extra=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)

    extra_train_loader = torch.utils.data.DataLoader(
        extra_train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()  # 2진 분류 Loss
    bd_criterion = BondaryLoss()

    model = nn.DataParallel(model, device_ids=gpus).cuda()
    teacher_model1 = nn.DataParallel(teacher_model1, device_ids=gpus) # teacher도 DataParallel
    teacher_model2 = nn.DataParallel(teacher_model2, device_ids=gpus) # teacher도 DataParallel

    # Optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV)
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=config.TRAIN.LR,
                                    weight_decay=config.TRAIN.WD)
    else:
        raise ValueError('Only Support SGD and Adam optimizer')

    epoch_iters = int(train_dataset.__len__() / batch_size)

    best_mIoU = 0
    last_epoch = 0
    if config.TRAIN.RESUME:
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

    # EMA alpha
    ema_alpha = config.TRAIN.EMA_ALPHA  # config에 EMA_ALPHA 추가해야 함.


    for epoch in range(last_epoch, end_epoch):
        model.train()  # student model train
        teacher_model1.eval()  # teacher model eval
        teacher_model2.eval()  # teacher model eval

        # Teacher Selection: 1 에폭마다 teacher 번갈아 선택
        current_teacher = teacher_model1 if epoch % 2 == 0 else teacher_model2
        other_teacher = teacher_model2 if epoch % 2 == 0 else teacher_model1



        for i, (input, target, _) in enumerate(train_loader):  # Labeled data

            # input = input.cuda() # 이미 nail.py에서 cuda()처리
            # target = target.cuda()
            target = target.unsqueeze(1).float()

            labeled_output = model(input)  # Forward pass labeled data
            labeled_loss = criterion(labeled_output, target)  # Supervised loss
            loss = labeled_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update student -> current_teacher
            update_ema_variables(model, current_teacher, ema_alpha, writer_dict['train_global_steps'])



        # Unlabeled data, teacher-student training
        for i, (input_unlabeled, _, _) in enumerate(extra_train_loader):
            # input_unlabeled = input_unlabeled.cuda()

            with torch.no_grad():
                # Teacher model prediction
                teacher_output = current_teacher(input_unlabeled)
                teacher_output = torch.sigmoid(teacher_output) # 확률값으로

                # Teacher Thresholding , confidence masking
                max_prob, max_idx = torch.max(teacher_output, dim=1) # unqueeze 불필요
                threshold, hist = get_threshold(max_idx.flatten(), config.DATASET.NUM_CLASSES) # 클래스 개수
                mask = max_prob.ge(0.5).float()  # threshold = 0.5 사용

                # 필터링 된 pseudo label 생성.
                filtered_pseudo_labels = max_idx.unsqueeze(1) * mask  # (B, 1, H, W)


            # Student model prediction.
            student_output = model(input_unlabeled)
            # Unsupervised loss (using pseudo labels)
            unlabeled_loss = criterion(student_output, filtered_pseudo_labels.float())  * mask.mean() # 마스크 평균 곱해줌.

            loss = unlabeled_loss
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()  # student 파라미터 업데이트
            # EMA update student -> current_teacher
            update_ema_variables(model, current_teacher, ema_alpha, writer_dict['train_global_steps'])
             # EMA update student -> other_teacher
            update_ema_variables(model, other_teacher, ema_alpha, writer_dict['train_global_steps'])

            writer_dict['train_global_steps'] += 1  # global step 증가


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
                torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'best.pt'))

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