"""Episode 9, the fair comparison: episode 6's diffusion model sampled WITH fresh noise (η = 1, episode 7's better
sampler) at 5 and 10 steps, from the same starting noise as the flow model.

    python3 diffusion_fresh_noise.py    # ~1 minute → results_fresh.json, out/diffusion_eta1_05.png
(Episode 7 measured 20 and 50 steps from the same noise: 95.4% and 95.2%.)
"""
import json
import os

import numpy as np
import torch

import diffusion as D
from judge import load_judge, to_tensor
from shapes import make_dataset, save_grid

torch.set_num_threads(int(os.environ.get("THREADS", min(8, os.cpu_count()))))
judge = load_judge()
train01 = to_tensor(make_dataset(20_000, seed=0)[0])
model = D.UNet(ch=(32, 64, 128))
model.load_state_dict(torch.load("../g06_learning_to_denoise/out/unet_ema.pt"))
model.eval()


@torch.no_grad()
def sample(x, steps, eta, generator):
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


start = torch.randn(500, 3, 32, 32, generator=torch.Generator().manual_seed(1))
res = {}
for k in (5, 10):
    x = sample(start.clone(), k, 1.0, torch.Generator().manual_seed(2))
    res[f"eta1_steps{k}"] = score(x)
    print(k, res[f"eta1_steps{k}"], flush=True)
    img = ((x[:16] + 1) / 2 * 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8)
    save_grid(img, f"out/diffusion_eta1_{k:02d}.png")
json.dump(res, open("results_fresh.json", "w"), indent=1)
