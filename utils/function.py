# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F
import torch.nn as nn

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix, compute_miou
from utils.utils import adjust_learning_rate
from utils.utils import AverageMeter, inter_and_union


def train(cfg, epoch, num_epoch, epoch_iters, base_lr, num_iters,
          train_loader, extra_train_loader, optimizer, model, criterion, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()  # labels.long().cuda() -> labels.cuda()

        output = model(images)
        if cfg.MODEL.NUM_OUTPUTS == 1:
            output = torch.squeeze(output)
        loss = criterion(output, labels)

        # Accuracy (Binary Classification)
        pred = (output > 0.5).float()
        acc = (pred == labels).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % cfg.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}'.format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average())
            logging.info(msg)
    # Unlabeled data (Self-training, 예시)
    if extra_train_loader:
      model.eval()
      with torch.no_grad():
        for j, (input_extra, _) in enumerate(extra_train_loader):
          input_extra = input_extra.cuda(non_blocking=True)
          output_extra = model(input_extra)
          if cfg.MODEL.NUM_OUTPUTS == 1:  # single output일 때,
            output_extra = torch.squeeze(output_extra)

          # Pseudo-labeling (thresholding)
          pseudo_labels = (output_extra > 0.5).float()

                # Pseudo label을 사용한 추가적인 loss계산 (optional).
                # loss_extra = criterion(output_extra, pseudo_labels)
                # loss_extra.backward()
                # optimizer.step()

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def inter_and_union(pred, mask, K):
    # K는 클래스 개수 (배경 포함)
    pred = pred.view(-1)  # Flatten
    mask = mask.view(-1)  # Flatten
    
    # 교집합(intersection) 계산
    intersection = np.zeros(K)
    for k in range(K):
      intersection[k] = ((pred == k) & (mask == k)).sum() # True False로 계산.

    # 합집합(union) 계산
    union = np.zeros(K)
    for k in range(K):
      union[k] = ((pred == k) | (mask == k)).sum()

    return intersection, union


def validate(cfg, test_loader, model, writer_dict):

    model.eval()
    ave_loss = AverageMeter()
    intersection_sum = np.zeros(cfg.DATASET.NUM_CLASSES)
    union_sum = np.zeros(cfg.DATASET.NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss 사용

    with torch.no_grad():
        for i, (input, target,_) in enumerate(test_loader):

            input = input.cuda()
            target = target.cuda()
            output = model(input)
            # 만약 output이 리스트 형태라면 첫 번째 요소만 사용
            if isinstance(output, list):
                output = output[0]

            # output size에 맞게 target을 resize할 필요 없음!

            loss = criterion(output, target)  # CrossEntropyLoss 사용

            # prediction
            pred = output.argmax(dim=1)  # CrossEntropyLoss를 위한 prediction

            # binary segmentation에서는 0,1 두 class의 교집합, 합집합 계산
            intersection, union = inter_and_union(pred.cpu().numpy(), target.cpu().numpy(), K=cfg.DATASET.NUM_CLASSES)
            intersection_sum += intersection
            union_sum += union

            ave_loss.update(loss.item())

    iou = intersection_sum / (union_sum + 1e-10)
    mean_iou = iou.mean()

    writer_dict['valid_global_steps'] += 1

    return ave_loss.average(), mean_iou, iou

# testval 함수 삭제
# def testval(config, test_dataset, testloader, model,
#             sv_dir='./', sv_pred=False):
#     ...

def test(config, test_dataset, testloader, model,
         sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            # size = size[0]  # 배치 차원 제거 -> size를 image.shape에서 가져옴.
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())
            
            # if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:  # 주석 처리 또는 삭제
            #     pred = F.interpolate(
            #         pred, size[-2:],
            #         mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            #     )

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name) # 이 함수 확인