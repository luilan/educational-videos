"""The full U-Net's per-step noise error, measured exactly like train_ablation.py does (for the comparison chart).

    python3 eval_full.py     → results_full.json (needs ../g06_learning_to_denoise/out/unet_ema.pt)
"""
import json

import torch
import torch.nn.functional as F

import diffusion as D
from judge import to_tensor
from shapes import make_dataset

model = D.UNet(ch=(32, 64, 128))
model.load_state_dict(torch.load("../g06_learning_to_denoise/out/unet_ema.pt"))
model.eval()
test_x0 = to_tensor(make_dataset(256, seed=99)[0]) * 2 - 1
LEVELS = [10, 50, 100, 200, 400, 600, 800, 950]
per_level = {}
g = torch.Generator().manual_seed(4)
with torch.no_grad():
    for lv in LEVELS:
        eps = torch.randn(test_x0.shape, generator=g)
        tt = torch.full((len(test_x0),), lv)
        per_level[lv] = F.mse_loss(model(D.noise(test_x0, tt, eps), tt), eps).item()
print(per_level)
json.dump({"variant": "full", "parameters": sum(p.numel() for p in model.parameters()), "levels": LEVELS,
           "eps_mse_by_step": per_level}, open("results_full.json", "w"), indent=1)
