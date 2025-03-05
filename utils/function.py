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

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix, compute_miou
from utils.utils import adjust_learning_rate



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

def validate(cfg, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    confusion_matrix = np.zeros((cfg.DATASET.NUM_CLASSES, cfg.DATASET.NUM_CLASSES))
    criterion = nn.BCEWithLogitsLoss() # validate에도 criterion추가.
    with torch.no_grad():
        for idx, (image, label) in enumerate(testloader): # bd_gts 삭제
            # size = label.size() # 삭제
            image = image.cuda()
            label = label.cuda() # label.long().cuda() -> label.cuda()
            # bd_gts = bd_gts.float().cuda() # 삭제

            # losses, pred, _, _ = model(image, label, bd_gts) # 기존
            output = model(image) # 수정
            if cfg.MODEL.NUM_OUTPUTS == 1:
              output = torch.squeeze(output)
            loss = criterion(output, label) # loss 계산
            
            # if not isinstance(pred, (list, tuple)): # 삭제
            #     pred = [pred]
            # for i, x in enumerate(pred):
            #     x = F.interpolate(
            #         input=x, size=size[-2:],
            #         mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            #     )

            #     confusion_matrix[..., i] += get_confusion_matrix(
            #         label,
            #         x,
            #         size,
            #         config.DATASET.NUM_CLASSES,
            #         config.TRAIN.IGNORE_LABEL
            #     )

            # if idx % 10 == 0: # 삭제
            #     print(idx)

            # loss = losses.mean() # 삭제
            ave_loss.update(loss.item())

            # for i in range(nums): # 삭제
            #     pos = confusion_matrix[..., i].sum(1)
            #     res = confusion_matrix[..., i].sum(0)
            #     tp = np.diag(confusion_matrix[..., i])
            #     IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            #     mean_IoU = IoU_array.mean()

            #     logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
            pred = (output > 0.5).float()  # 예측 (0 또는 1)
            confusion_matrix += get_confusion_matrix(
                label.cpu().numpy(),
                pred.cpu().numpy(),
                size = None,  # 이제 size 필요 없음
                num_classes=cfg.DATASET.NUM_CLASSES,
                ignore_label=cfg.TRAIN.IGNORE_LABEL
            )

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array

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