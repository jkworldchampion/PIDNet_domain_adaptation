# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cityscapes import Cityscapes as cityscapes
from .camvid import CamVid as camvid
from .nail import Nail as nail  # 추가

DATASETS = {
    'cityscapes': cityscapes,
    'camvid': camvid,
    'nail': nail,  # 추가
}

def get_dataset(cfg, **kwargs):
    return DATASETS[cfg.DATASET.DATASET](cfg, **kwargs)