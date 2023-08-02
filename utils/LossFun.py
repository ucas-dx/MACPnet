#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 12:45
# @Author  : Denxun
# @FileName: LossFun.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        iflat = inputs.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def dice_coefficient(predicted, target):
    smooth = 0.1
    product = predicted * target
    intersection = 2*(product.sum(2).sum(2).sum(1) + smooth)
    union = predicted.sum(2).sum(2).sum(1) + target.sum(2).sum(2).sum(1) + smooth
    return (intersection / union).mean()

class FocalLoss_Binary(nn.Module):
    def __init__(self, alpha=0.4,gamma=2,weight=None,ignore_index=None):
        super(FocalLoss_Binary, self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.weight=weight
        self.ignore_index =ignore_index
        self.bce_fn=nn.BCEWithLogitsLoss(weight=self.weight)
    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore
            labels = labels[mask]
            preds = preds[mask]
        logpt = -self.bce_fn(preds,labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

def iou(predicted, target):
    smooth = 0.1
    intersection = (predicted * target).sum(2).sum(2).sum(1)
    union = predicted.sum(2).sum(2).sum(1) + target.sum(2).sum(2).sum(1) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 计算正样本的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  # 计算Focal Loss
        return focal_loss.mean()
