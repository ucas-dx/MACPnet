#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 12:39
# @Author  : Denxun
# @FileName: datasets.py
# @Software: PyCharm

import os
import torch
import numpy as np
from skimage.util import img_as_ubyte
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms
import random
# import skimage
seed =123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
class CellSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'segmentations')
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = io.imread(img_name)

        mask_name = os.path.join(self.mask_dir, self.mask_filenames[idx])
        mask = io.imread(mask_name)
        image = img_as_ubyte(image)
        mask = img_as_ubyte(mask)

        sample = {'image': image, 'mask': mask}

        # 对image应用转换操作
        if self.image_transform:
            sample['image'] = self.image_transform(sample['image'])

        # 对mask应用转换操作
        if self.mask_transform:
            sample['mask'] = self.mask_transform(sample['mask'])

        return sample


# 定义transforms操作，包括将图像转换为灰度图像
image_transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([512, 512]),
    transforms.Grayscale(),
    transforms.ToTensor(),
    #transforms.Normalize(0.40081004856666735,0.0877802242064718),
    lambda x: torch.as_tensor(x, dtype=torch.float32)
])
image_transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([512, 512]),
    transforms.Grayscale(),
    transforms.ToTensor(),
    #transforms.Normalize(0.3579441845027396,0.10539796972687705),
    lambda x: torch.as_tensor(x, dtype=torch.float32)
])
mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([512, 512]),
    transforms.Grayscale(),
    transforms.ToTensor(),
    lambda x: torch.as_tensor(x, dtype=torch.float32)
])

root_dir_train = r"PATH FOR YOUR DATA"
root_dir_Val = r'PATH FOR YOUR DATA'
root_dir_test = r'PATH FOR YOUR DATA'


cell_dataset_train = CellSegmentationDataset(root_dir_train, image_transform=image_transform_train, mask_transform=mask_transform)
cell_dataset_test = CellSegmentationDataset(root_dir_test, image_transform=image_transform_train, mask_transform=mask_transform)
cell_dataset_test1 = CellSegmentationDataset(root_dir_Val, image_transform=image_transform_train, mask_transform=mask_transform)

train_loader = torch.utils.data.DataLoader(cell_dataset_train, batch_size=8, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(cell_dataset_test, batch_size=1, shuffle=False, num_workers=0)
Val_loader = torch.utils.data.DataLoader(cell_dataset_test1, batch_size=1, shuffle=False, num_workers=0)
