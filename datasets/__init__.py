# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cityscapes import Cityscapes as cityscapes
from .camvid import CamVid as camvid
from .nail import Nail  # Nail 클래스를 직접 import (as nail 제거)

DATASETS = {
    'cityscapes': cityscapes,
    'camvid': camvid,
    'nail': Nail,  # 추가
}

def get_dataset(cfg, **kwargs):
    return DATASETS[cfg.DATASET.DATASET](cfg, **kwargs)