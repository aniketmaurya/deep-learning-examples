import lightning as L
from benchmark import benchmark_trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


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


class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Net()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def common_step(self, x, y, stage):
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log(f"{stage}/loss", loss, on_epoch=True)
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

    unoptimized_t = benchmark_trainer(
        model,
        train_dataloaders=train_loader,
    )
    print(f"time taken to train unoptimized model: {unoptimized_t}")

    compiled_model = torch.compile(LitModel(), mode="reduce-overhead").to("cuda")
    compiled_model(torch.randn(32, 3, 32, 32).to("cuda"))  # warmup
    optimized_t = benchmark_trainer(
        compiled_model,
        train_dataloaders=train_loader,
    )
    print(f"time taken to train optimized model: {optimized_t}")


if __name__ == "__main__":
    main()
