# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config

# class FullModel(nn.Module): # 삭제

#   def __init__(self, model, sem_loss, bd_loss):
#     super(FullModel, self).__init__()
#     self.model = model
#     self.sem_loss = sem_loss
#     self.bd_loss = bd_loss

#   def pixel_acc(self, pred, label):
#     _, preds = torch.max(pred, dim=1)
#     valid = (label >= 0).long()
#     acc_sum = torch.sum(valid * (preds == label).long())
#     pixel_sum = torch.sum(valid)
#     acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
#     return acc

#   def forward(self, inputs, labels, bd_gt, *args, **kwargs):

#     outputs = self.model(inputs, *args, **kwargs)

#     h, w = labels.size(1), labels.size(2)
#     ph, pw = outputs[0].size(2), outputs[0].size(3)
#     if ph != h or pw != w:
#         for i in range(len(outputs)):
#             outputs[i] = F.interpolate(outputs[i], size=(
#                 h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

#     acc  = self.pixel_acc(outputs[-2], labels)
#     loss_s = self.sem_loss(outputs[:-1], labels)
#     loss_b = self.bd_loss(outputs[-1], bd_gt)

#     filler = torch.ones_like(labels) * config.TRAIN.IGNORE_LABEL
#     bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:])>0.8, labels, filler)
#     loss_sb = self.sem_loss(outputs[-2], bd_label)
#     loss = loss_s + loss_b + loss_sb

#     return torch.unsqueeze(loss,0), outputs[:-1], acc, [loss_s, loss_b]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

# inter_and_union 함수 추가
def inter_and_union(pred, mask, K):
    # K는 클래스 개수 (배경 포함)
    pred = pred.view(-1)  # Flatten
    mask = mask.view(-1)  # Flatten

    # 교집합(intersection) 계산
    intersection = np.zeros(K)  # 클래스 개수만큼 0으로 초기화
    for k in range(K):
        intersection[k] = ((pred == k) & (mask == k)).sum() # True, False로 계산

    # 합집합(union) 계산
    union = np.zeros(K)  # 클래스 개수만큼 0으로 초기화
    for k in range(K):
      union[k] = ((pred == k) | (mask == k)).sum()


    return intersection, union

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    # output = pred.cpu().numpy().transpose(0, 2, 3, 1) # 삭제
    # seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8) # 삭제
    # seg_gt = np.asarray(
    # label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int) # 삭제

    seg_gt = label.flatten() # 수정
    seg_pred = pred.flatten() # 수정
    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters,
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr

def compute_miou(preds, targets, num_classes, ignore_index=255):

    confusion_matrix = np.zeros((num_classes, num_classes))
    for pred, target in zip(preds, targets):
        pred = pred.flatten()
        target = target.flatten()
        mask = (target != ignore_index)
        pred = pred[mask]
        target = target[mask]
        index = (target * num_classes + pred).astype('int32')
        label_count = np.bincount(index)
        for i_label in range(num_classes):
            for i_pred in range(num_classes):
                cur_index = i_label * num_classes + i_pred
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred] = label_count[cur_index]
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc

def collate_fn(batch):
    images, targets, names = zip(*batch)  # Unpack the batch

    # Find the maximum height and width
    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)

    # Pad the images and targets
    padded_images = []
    padded_targets = []
    for img, tar in zip(images, targets):
        pad_height = max_height - img.shape[1]
        pad_width = max_width - img.shape[2]

        # Pad image (padding value 0, BGR format)
        padded_img = np.pad(img, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
        padded_images.append(torch.from_numpy(padded_img))


        padded_tar = np.pad(tar, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0) # 수정된 부분
        padded_targets.append(torch.from_numpy(padded_tar))

    # Stack images and targets
    images = torch.stack(padded_images, dim=0)
    targets = torch.stack(padded_targets, dim=0)

    return images, targets, list(names)