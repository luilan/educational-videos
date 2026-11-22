"""Episode 8, fair comparison: sample all three networks (full, no skips, no time) with both samplers from episode 7.

    python3 eval_samplers.py   # ~10 minutes on an 8-core CPU → results_samplers.json, out/<variant>_eta1_samples.png

DDIM (η = 0) and fresh noise (η = 1), 50 steps, 500 samples from the same starting noise, scored by episode 3's judge.
"""
import json
import os

import numpy as np
import torch

import diffusion as D
from judge import load_judge, to_tensor
from shapes import make_dataset, save_grid

torch.set_num_threads(int(os.environ.get("THREADS", min(8, os.cpu_count()))))
N, STEPS = 500, 50
judge = load_judge()
train01 = to_tensor(make_dataset(20_000, seed=0)[0])
WEIGHTS = {"full": ("../g06_learning_to_denoise/out/unet_ema.pt", {}),
           "noskip": ("out/noskip_ema.pt", {"use_skips": False}),
           "notime": ("out/notime_ema.pt", {"use_time": False})}


@torch.no_grad()
def sample(model, x, steps, eta, generator):
    ts = torch.linspace(D.T, 1, steps).round().long().tolist()
    for i, t in enumerate(ts):
        eps = D.predict_eps(model, x, t)
        ab = D.ALPHA_BAR[t - 1]
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        ab_s = D.ALPHA_BAR[ts[i + 1] - 1] if i + 1 < len(ts) else torch.tensor(1.0)
        sigma = eta * ((1 - ab_s) / (1 - ab)).sqrt() * (1 - ab / ab_s).sqrt()
        x = ab_s.sqrt() * x0 + (1 - ab_s - sigma ** 2).clamp(min=0).sqrt() * eps
        if eta > 0 and i + 1 < len(ts):
            x = x + sigma * torch.randn(x.shape, generator=generator)
    return x.clamp(-1, 1)


@torch.no_grad()
def score(x):
    x01 = (x + 1) / 2
    probs = [l.softmax(-1) for l in judge(x01)]
    ok = torch.stack([p.max(-1).values >= 0.9 for p in probs[:3]] + [probs[3][:, 1] >= 0.5], 1).all(1)
    labels = torch.stack([p.argmax(-1) for p in probs[:3]], 1)
    combos = {tuple(l) for l, k in zip(labels.tolist(), ok.tolist()) if k}
    d = torch.cdist(x01[:200].flatten(1), train01.flatten(1)) ** 2 / x01[0].numel()
    return {"recognizable": ok.float().mean().item(), "combinations": len(combos),
            "nearest_train_mse": d.min(1).values.mean().item()}


res = {}
start = torch.randn(N, 3, 32, 32, generator=torch.Generator().manual_seed(1))
for name, (path, kw) in WEIGHTS.items():
    model = D.UNet(ch=(32, 64, 128), **kw).eval()
    model.load_state_dict(torch.load(path))
    res[name] = {}
    for eta in (0, 1):
        x = sample(model, start.clone(), STEPS, eta, torch.Generator().manual_seed(2))
        res[name][f"eta{eta}"] = score(x)
        print(name, f"eta={eta}", res[name][f"eta{eta}"], flush=True)
        if eta == 1:
            img = ((x[:16] + 1) / 2 * 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            save_grid(img, f"out/{name}_eta1_samples.png")
with open("results_samplers.json", "w") as f:
    json.dump({"n": N, "steps": STEPS, "results": res}, f, indent=1)
