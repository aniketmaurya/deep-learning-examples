import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchmetrics.classification import Accuracy


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Net()
        self.criterion = nn.CrossEntropyLoss()
        # self.accuracy = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def common_step(self, x, y, stage):
        logits = self.model(x)
        loss = self.criterion(logits, y)
        # acc = self.accuracy(logits, y)
        self.log_dict(
            {
                f"{stage}/loss": loss,
            }
        )
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step(x, y, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step(x, y, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001, momentum=0.9)


def main():

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 8

    train_set = torchvision.datasets.CIFAR10(
        root="~/data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_set = torchvision.datasets.CIFAR10(
        root="~/data", train=False, download=True, transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    model = LitModel()

    trainer = pl.Trainer(max_epochs=2, devices=2)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.validate(model, val_loader)


if __name__ == "main":
    main()
