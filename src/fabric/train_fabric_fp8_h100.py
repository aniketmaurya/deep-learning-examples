import torch
import torch.nn as nn
import torchvision.transforms as transforms
from lightning.fabric import Fabric
from timm import create_model
from tqdm import tqdm
from data import load_cifar10


# Select 8bit mixed precision via TransformerEngine, with model weights in bfloat16
fabric = Fabric(precision="transformer-engine")
fabric.launch()



train_loader = load_cifar10()
model = model = create_model("resnet50", num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ⚡️⚡️⚡️⚡️⚡️ Setup model and optimizer with Fabric ⚡️⚡️⚡️⚡️⚡️
model, optimizer = fabric.setup(model, optimizer)
# setup dataloader with Fabric
train_loader = fabric.setup_dataloaders(train_loader)

for i in range(2):
    for x, y in tqdm(train_loader):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        # ⚡️⚡️⚡️⚡️⚡️ fabric.backward(...) instead of loss.backward() ⚡️⚡️⚡️⚡️⚡️
        fabric.backward(loss)
        optimizer.step()
