import torch
import torchvision
import torchvision.transforms as transforms


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L


class Net(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 160, 3, padding=1)
        self.conv2 = nn.Conv2d(160, 160, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(160, 320, 3, padding=1)
        self.conv4 = nn.Conv2d(320, 320, 3, padding=1)
        self.conv5 = nn.Conv2d(320, 640, 3, padding=1)
        self.conv6 = nn.Conv2d(640, 640, 3, padding=1)
        self.fc1 = nn.Linear(640 * 4 * 4, 5120)
        self.fc2 = nn.Linear(5120, 2560)
        self.fc3 = nn.Linear(2560, 1280)
        self.fc4 = nn.Linear(1280, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        if batch_idx % 2000 == 1999:
            mem_utilization = [round((mem_total - mem_free) / mem_total *100, 2) for mem_free, mem_total in [torch.cuda.mem_get_info('cuda:0'), torch.cuda.mem_get_info('cuda:1')]]
            print(f"Epoch: {batch_idx+1}, Loss: {loss.item()}, Memory Utilization: {mem_utilization}%")
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.trainer.model.parameters(), lr=0.001, momentum=0.9)
        return optimizer


net = Net()
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='/data/aniket/datasets', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                      shuffle=True, num_workers=8)

trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    strategy="fsdp",
    max_epochs=1,
    enable_progress_bar=False,
)
trainer.fit(net, trainloader)
