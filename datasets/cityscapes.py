
import numpy as np
import os
from PIL import Image
import sys
import torchvision
import torch
from datasets import base
from platforms.platform import get_platform
import torch.nn.functional as F

import random


def data_augmentation(img, gt, dataset='nyuv2', scale_list=[1.0, 1.2, 1.5]):
    if dataset == 'nyuv2':
        h, w = img.shape[1], img.shape[2]
        h_crop, w_crop = 288, 384
        i_crop = random.randint(0, h - h_crop)
        j_crop = random.randint(0, w - w_crop)
    elif dataset == 'city':
        h, w = img.shape[1], img.shape[2]
        h_crop, w_crop = 512, 512
        i_crop = random.randint(0, h - h_crop)
        j_crop = random.randint(0, w - w_crop)
    img = img[:, i_crop:i_crop + h_crop, j_crop:j_crop + w_crop]

    if 'semantic' in gt:
        gt['semantic'] = gt['semantic'][i_crop:i_crop + h_crop, j_crop:j_crop + w_crop]
    if 'depth' in gt:
        gt['depth'] = gt['depth'][:, i_crop:i_crop + h_crop, j_crop:j_crop + w_crop]

    scale = scale_list[random.randint(0, len(scale_list) - 1)]
    if scale != 1:
        h, w = img.shape[1], img.shape[2]
        h_new, w_new = int(h / scale), int(w / scale)
        i = random.randint(0, h - h_new)
        j = random.randint(0, w - w_new)
        img = F.interpolate(img[None, :, i:i + h_new, j:j + w_new], size=(h, w), mode='bilinear',
                            align_corners=True).squeeze(0)
        if 'semantic' in gt:
            gt['semantic'] = F.interpolate(gt['semantic'][None, None, i:i + h_new, j:j + w_new], size=(h, w),
                                           mode='nearest').squeeze(0).squeeze(0)
        if 'depth' in gt:
            gt['depth'] = F.interpolate(gt['depth'][None, :, i:i + h_new, j:j + w_new], size=(h, w), mode='bilinear',
                                        align_corners=True).squeeze(0)
            gt['depth'] = gt['depth'] / scale

    if torch.rand(1) < 0.5:
        img = torch.flip(img, dims=[2])
        if 'semantic' in gt:
            gt['semantic'] = torch.flip(gt['semantic'], dims=[1])
        if 'depth' in gt:
            gt['depth'] = torch.flip(gt['depth'], dims=[2])

    return img, gt


class Dataset(base.ImageDataset):
    """Cityscapes"""

    def __init__(self, tasks=['semantic', 'depth'], is_training=True, augmentation=False):
        self.tasks = tasks
        self.is_training = is_training
        self.augmentation = augmentation

        if self.is_training:
            self.data_len = 2975
        else:
            self.data_len = 500

    def __getitem__(self, index):
        # load data from the pre-processed .npy files
        gt = {}  # only load ground truth necessary for training
        data_path = './dataset/city'
        if self.is_training:
            data_suffix = 'train_'
        else:
            data_suffix = 'valid_'

        img = torch.from_numpy(np.load(data_path + '/image/{}{:d}.npy'.format(data_suffix, index)))
        img = img.cuda(non_blocking=True)
        img = img.float()
        img /= 255

        for task in self.tasks:
            gt[task] = torch.from_numpy(np.load(data_path + '/{}/{}{:d}.npy'.format(task, data_suffix, index)))
            gt[task] = gt[task].cuda(non_blocking=True)
            gt[task] = gt[task].float()

        # apply data augmentation if required
        if self.augmentation:
            img, gt = data_augmentation(img, gt, dataset='city')

        if 'semantic' in gt:
            gt['semantic'] = gt['semantic'].long()
        if 'depth' in gt:
            gt['depth'] = gt['depth'] / 10  # balance the depth scale

        return img, gt

    def __len__(self):
        return self.data_len


    @staticmethod
    def num_train_examples(): return 2975

    @staticmethod
    def num_test_examples(): return 500

    @staticmethod
    def num_classes(): return 1000

    @staticmethod
    def get_train_set(use_augmentation):

        train_dataset = Dataset(tasks=['semantic', 'depth'], is_training=True)

        return train_dataset

    @staticmethod
    def get_test_set():
        test_dataset = Dataset(tasks=['semantic', 'depth'], is_training=False)
        return test_dataset


DataLoader = base.DataLoader
