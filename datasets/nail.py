import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class Nail(Dataset):
    def __init__(self, cfg, is_train=True, is_extra=False):
        self.root = cfg.DATASET.ROOT  # "nail"
        self.is_train = is_train
        self.is_extra = is_extra
        self.input_size = cfg.TRAIN.IMAGE_SIZE if is_train else cfg.TEST.IMAGE_SIZE
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet mean
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # ImageNet std

        if self.is_extra:  # Unlabeled data
            self.images_dir = os.path.join(self.root, 'non-label')
            self.image_paths = [os.path.join(self.images_dir, img_id) for img_id in sorted(os.listdir(self.images_dir))]

        else:  # Labeled data (train, val, test)
            if self.is_train:
                data_type = 'train'
            else:
                data_type = 'test' if cfg.DATASET.TEST_SET else 'val'  # Use 'test' or 'val'

            self.images_dir = os.path.join(self.root, 'raw', f'{data_type}')
            self.masks_dir = os.path.join(self.root, 'raw', f'{data_type}_label')
            self.image_paths = [os.path.join(self.images_dir, img_id)
                                for img_id in sorted(os.listdir(self.images_dir))]
            self.mask_paths = [os.path.join(self.masks_dir, img_id)
                               for img_id in sorted(os.listdir(self.masks_dir))]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.is_extra:
            # Unlabeled data: only image
            img_path = self.image_paths[idx]
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)

            image = image.astype(np.float32) / 255.0
            image = (image - self.mean) / self.std
            image = image.transpose((2, 0, 1))

            return torch.from_numpy(image.copy()), np.array(0, dtype=np.int64)  # Dummy label


        else:
            # Labeled data: image and mask
            img_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
            mask = cv2.resize(mask, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)  # Binarize

            image = image.astype(np.float32) / 255.0
            image = (image - self.mean) / self.std
            image = image.transpose((2, 0, 1))

            return torch.from_numpy(image.copy()), torch.from_numpy(mask.copy()).float()