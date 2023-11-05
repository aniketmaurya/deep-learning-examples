import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from lightning.fabric import Fabric
from timm import create_model
from lightning.fabric.plugins import TransformerEnginePrecision
from tqdm import tqdm


def load_cifar10():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 32

    train_set = torchvision.datasets.CIFAR10(
        root="~/data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )

    return train_loader

