import sys
from pathlib import Path

import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import FSDPStrategy
from lit_gpt import GPT, Config
from lit_gpt.model import Block

checkpoint_dir = Path("/data/aniket/Llama-2-7b-hf/")
config = Config.from_json(checkpoint_dir / "lit_config.json")

model_file = "lit_model.pth"
checkpoint_path = checkpoint_dir / model_file

strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=True)
fabric = Fabric(devices=1, accelerator="gpu", precision="bf16-true", strategy=strategy)
fabric.launch()

fabric.print(
    f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr
)
with fabric.init_module(empty_init=False):
    model = GPT(config)
    print("inside the context manager")
    print(next(model.lm_head.parameters()))
print(next(model.lm_head.parameters())[0][0].device)
model.eval()
model = fabric.setup_module(model)
fabric.print(
    f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr
)
