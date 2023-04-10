import lightning as L
from torch import nn
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, n=100):
        super().__init__()
        self.data = list(range(1, n))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx] * 1.0, self.data[idx] * 2.0
        x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        return x, y


class BoringDataModule(L.LightningDataModule):
    def train_dataloader(self):
        return torch.utils.data.DataLoader(Dataset(100), shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(Dataset(50))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(Dataset(90))


class BoringDataModuleLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(1, 1)

    def forward(self, x):
        return self.l(x)


class BoringModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.save_hyperparameters()
        self.criterion = torch.nn.MSELoss()
        self.model = LinearModel()
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        sched = torch.optim.lr_scheduler.ConstantLR(optimizer)
        return [optimizer], [sched]

    def training_step(self, data):
        x, y = data
        x = x.float()
        y = y.float()
        o = self(x) * 2
        loss = self.criterion(o, y)
        self.log("acc", 0.9)
        return loss

    def validation_step(self, data, *args, **kwargs):
        x, y = data
        x = x.float()
        y = y.float()
        o = self(x) * 2
        loss = self.criterion(o, y)
        self.log("val_acc", 0.9)
        import time

        time.sleep(0.01)
        return loss

    def test_step(self, data, *args, **kwargs):
        x, y = data
        x = x.float()
        y = y.float()
        o = self(x) * 2
        loss = self.criterion(o, y)
        self.log("test_acc", 0.8)
        import time

        time.sleep(0.01)
        return loss


if __name__ == "__main__":
    model = BoringModel()
    dm = BoringDataModule()

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        default_root_dir="test_ckpt",
    )
    trainer.fit(model, datamodule=dm)
