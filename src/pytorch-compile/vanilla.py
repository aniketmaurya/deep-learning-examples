import torch
from torchvision.models import resnet50
from benchmark_inference import benchmark_inference

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


model = resnet50(weights=True).to(device)
compiled_model = torch.compile(model, mode="reduce-overhead")

x = torch.randn(1, 3, 224, 224).to(device)


unoptimized_t = benchmark_inference(model, x)
print(unoptimized_t)

optimized_t = benchmark_inference(compiled_model, x)
print(optimized_t)
