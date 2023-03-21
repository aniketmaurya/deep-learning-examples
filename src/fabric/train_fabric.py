from time import perf_counter

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from lightning.fabric import Fabric
from timm import create_model
from tqdm import tqdm

fabric = Fabric(accelerator="auto", devices=1, strategy="auto")
fabric.launch()


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


def main():
    train_loader = load_data()

    model = model = create_model("resnet50", num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ⚡️⚡️⚡️⚡️⚡️ Setup model and optimizer with Fabric ⚡️⚡️⚡️⚡️⚡️
    model, optimizer = fabric.setup(model, optimizer)
    # setup dataloader with Fabric
    train_loader = fabric.setup_dataloaders(train_loader)

    # ⚡️⚡️⚡️⚡️⚡️ Access the Device and strategy ⚡️⚡️⚡️⚡️⚡️
    print(f"training on {fabric.device} with {fabric.strategy} strategy")

    t0 = perf_counter()
    for i in range(2):
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            # ⚡️⚡️⚡️⚡️⚡️ fabric.backward(...) instead of loss.backward() ⚡️⚡️⚡️⚡️⚡️
            fabric.backward(loss)
            optimizer.step()
    t1 = perf_counter()
    print(t1 - t0)


if __name__ == "__main__":
    main()
