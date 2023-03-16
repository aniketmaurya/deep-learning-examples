import torch
from torchvision.models import resnet50
from time import perf_counter



model = resnet50(weights=True)
compiled_model = torch.compile(model)

x = torch.randn(1, 3, 224, 224)


def benchmark(model, inputs, trials=10):
    # warmup
    model(inputs)

    t0 = perf_counter()
    for i in range(trials):
        model(inputs)
    t1 = perf_counter()
    return t1 - t0


unoptimized_t = benchmark(model, x)
optimized_t = benchmark(compiled_model, x)
