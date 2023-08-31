#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 12:50
# @Author  : Denxun
# @FileName: train.py
# @Software: PyCharm
from model import MACPnet
from utils import *
import torch.nn as nn
import os
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
import tqdm
import random
from datasets import *
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
model = MACPnet.MACPnet()


def train_and_test_model(model,device, num_epochs=120, learning_rate=0.01,stepsize=50,stepgamma=0.5):
    model=model.to(device)
    # model_train.load_state_dict(torch.load('Epoch_90nnunet_attention.pth'))
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = DiceLoss()
    criterion3 = FocalLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    scheduler = StepLR(optimizer, step_size=stepsize, gamma=stepgamma)
    print(learning_rate)
    train_loss_history = []
    train_dice_history = []
    train_iou_history = []
    val_loss_history = []
    val_dice_history = []
    val_iou_history = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0
        for batch in tqdm.tqdm(train_loader):
            images = batch['image'].to(device)
            # print(images.shape)
            masks = batch['mask'].to(device)
            outputs = model(images)
            optimizer.zero_grad()
            loss1 = criterion1(outputs, masks)
            loss2 = criterion2(outputs, masks)
            loss = loss1+loss2
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_dice += dice_coefficient(torch.sigmoid(outputs), masks).item()
            train_iou += iou(torch.sigmoid(outputs), masks).item()
        scheduler.step()
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
        train_loss_history.append(train_loss)
        train_dice_history.append(train_dice)
        train_iou_history.append(train_iou)
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        with torch.no_grad():
            for batch in Val_loader:
                images = batch['image'].to(device)
                # print(images.shape)
                masks = batch['mask'].to(device)
                outputs = model(images)
                optimizer.zero_grad()
                loss1 = criterion1(outputs, masks)
                loss2 = criterion2(outputs, masks)
                loss3 = criterion3(outputs, masks)
                loss = loss1 + loss2 + loss3
                val_loss += loss.item()
                val_dice+= dice_coefficient(torch.sigmoid(outputs), masks).item()
                val_iou += iou(torch.sigmoid(outputs), masks).item()

        val_loss /= len(Val_loader)
        val_dice /= len(Val_loader)
        val_iou /= len(Val_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}')
        print(f'Test  - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model_train.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': loss,
    #
    # },  f'Epoch {epoch + 1}'+'model_dict.pt' )

    return model, train_loss_history, train_dice_history, train_iou_history, val_loss_history, val_dice_history, val_iou_history


if __name__ == '__main__':
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # trained_model, _, _, _, _, _, _ = train_and_test_model(model,
    #             device, num_epochs=150, learning_rate=0.01,stepsize=50)
    data=torch.ones(1,1,512,512)
    print(model(data).shape)
