from torchvision import datasets
import torchvision.transforms as transforms
import torch
from einops import rearrange, repeat
import numpy as np
import random
import math

def load_dataset(batch_size, num_workers):
    transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_train = datasets.CIFAR10("./data", train=True, transform=transform_train, download=True)
    dataset_val = datasets.CIFAR10("./data", train=False, transform=transform_train, download=True)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.RandomSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return data_loader_train, data_loader_val


def eval_images(model, epoch, image_validate, device):
    image_validate = image_validate.to(device)
    pred_val_image, mask = model(image_validate)
    pred_val_image = pred_val_image * mask 
    pred_val_image =+ pred_val_image * (1 - mask)
    img = torch.cat([pred_val_image * (1 - mask), pred_val_image, image_validate], dim=0)
    rearrange_order = '(v h1 w1) c h w -> c (h1 h) (w1 v w)'
    img = rearrange(img, rearrange_order, w1=2, v=3)

    return img


def take_indexes(seq, idx):
    return torch.gather(seq, 0, repeat(idx, 't b -> t b c', c=seq.shape[-1]))

def accuracy(logit, label):
    return torch.mean((logit.argmax(dim=-1) == label).float())


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def lr_func(epoch, WARMUP, TOTAL_EPOCH):
    return min((epoch + 1) / (WARMUP + 1e-8), 0.5 * (math.cos(epoch / TOTAL_EPOCH * math.pi) + 1))