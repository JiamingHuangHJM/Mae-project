import os
import argparse
import math
import sqlite3
from tkinter.messagebox import YESNO
import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from models import *
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import accuracy, set_seed, lr_func

SEED = 10001
BASE_LR = 1e-3
WEIGHT_DECAY = 0.05
TOTAL_EPOCH = 100
WARMUP = 5
PRETRAINED_PATH = None
OUTPUT_PATH = "vit_from_scratch.pt"
BATCH_SIZE = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(SEED)
    
train_dataset = torchvision.datasets.CIFAR10(
    './data', train=True, download=True, 
    transform=Compose([ToTensor(), Normalize(0.5, 0.5)])
    )

val_dataset = torchvision.datasets.CIFAR10(
    './data', train=False, download=True, 
    transform=Compose([ToTensor(), Normalize(0.5, 0.5)])
    )

trainloader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=4)

if PRETRAINED_PATH:
    model = MAE()
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location='cpu'))
    sw = SummaryWriter(os.path.join('logs', 'pretrain'))
else:
    model = MAE()
    sw = SummaryWriter(os.path.join('logs', 'scratch'))

model = ViT_cls(model.encoder, num_classes=10).to(device)

cel = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR * 128/256, betas=(0.9, 0.999), WEIGHT_DECAY=WEIGHT_DECAY)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

best_val_acc = 0
optimizer.zero_grad()
for epoch in range(TOTAL_EPOCH):
    
    loss_steps = []
    acc_steps = []
    model.train()
    for x, y in tqdm(iter(trainloader)):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = cel(pred, y)
        acc = accuracy(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_steps.append(loss.item())
        acc_steps.append(acc.item())

    lr_scheduler.step()

    train_loss_avg = sum(loss_steps) / len(acc_steps)
    train_acc_avg = sum(acc_steps) / len(acc_steps)
    print(f'Epoch {epoch}, avg train loss: {train_loss_avg}, avg train acc: {train_acc_avg}.')

    model.eval()
    with torch.no_grad():
        losses = []
        acces = []
        for x, y in tqdm(iter(valloader)):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = cel(pred, y)

            acc = accuracy(pred, y)
            losses.append(loss.item())
            acces.append(acc.item())

        val_loss_avg = sum(losses) / len(losses)
        val_acc_avg = sum(acces) / len(acces)

    print(f'Epoch {epoch}, avg val loss: {train_loss_avg}, avg val acc: {train_acc_avg}.')

    if val_acc_avg > best_val_acc:
        best_val_acc = val_acc_avg
        torch.save(model, OUTPUT_PATH)

    sw.add_scalars('cls/loss', {'train' : train_loss_avg, 'val' : val_loss_avg}, global_step=epoch)
    sw.add_scalars('cls/acc', {'train' : train_acc_avg, 'val' : val_acc_avg}, global_step=epoch)