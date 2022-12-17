import os
import torchvision.transforms as transforms
import torch
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, eval_images
from tqdm import tqdm
from utils import load_dataset

from models import *

IN_SHAPE = 32
EPOCHS = 800
BATCH_SIZE = 128
LR = 1.5e-4

sw = SummaryWriter(os.path.join('logs', 'pre-trained-model'))

device  = "cuda" if torch.cuda.is_available() else "cpu"
mae = MAE(mask_ratio=0.75)
mae.to(device)
data_loader_train, data_loader_val = load_dataset(BATCH_SIZE, 4)
test_image, _ = next(iter(data_loader_val))
optimizer = torch.optim.AdamW(mae.parameters(), LR=LR, betas=(0.9, 0.95), weight_decay=0.05)

for epoch in range(EPOCHS):
    loss_total = 0.0
    mae.train()
    with tqdm(data_loader_train, unit="batch") as tepoch:
        for idx, (x, _) in enumerate(tepoch):
            optimizer.zero_grad()
            x = x.to(device, non_blocking=True)
            predicted_x, mask = mae(x)
            temp_loss = mask * (predicted_x - x) ** 2 
            sizes =  3 * IN_SHAPE**2
            temp_loss = 3 * temp_loss.view(x.shape[0], sizes) / 4  
            loss = torch.sum(torch.mean(temp_loss, dim=-1))
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

    epoch_loss = loss_total / len(data_loader_train)
    mae.eval()
    with torch.no_grad():
        image = eval_images(mae, epoch, test_image, device)
        sw.add_image('image', 0.5 * (image + 1), global_step=epoch)


    torch.save(mae.state_dict(), './MAE' + str(epoch) + '.pt')