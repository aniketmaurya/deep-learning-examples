from time import perf_counter

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from timm import create_model
from tqdm import tqdm


def load_data():
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


device = torch.device("cuda")
train_loader = load_data()
model = create_model("resnet50", num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

t0 = perf_counter()
for i in range(2):
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
t1 = perf_counter()

print(t1 - t0)
