# MNISTの前半
# https://zenn.dev/kouch/articles/a28316bffbdba8

# MNISTの後半


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_datasets = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_datasets  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_datasets, batch_size=256, shuffle=False, num_workers=0)

