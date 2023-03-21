from time import perf_counter

import lightning as L
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from benchmark import benchmark_trainer
from timm import create_model


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


class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = create_model("resnet18", num_classes=10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def common_step(self, x, y, stage):
        logits = self.model(x)
        loss = self.criterion(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step(x, y, "train")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)


def main():
    train_loader = load_data()

    model = LitModel()
    trainer = L.Trainer(
        max_epochs=1,
        devices=1,
        logger=False,
    )
    t0 = perf_counter()
    trainer.fit(model, train_dataloders=train_loader)
    t1 = perf_counter()
    optimized_t = t1 - t0
    print(f"time taken to train un-compiled model: {optimized_t}")


if __name__ == "__main__":
    main()
