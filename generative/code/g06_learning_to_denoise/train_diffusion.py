"""Episode 6 experiments: learn to undo noise, first on 2D points, then on Shapes pictures.

    python3 train_diffusion.py toy       # 2D spiral, a tiny MLP denoiser, ~2 minutes
    python3 train_diffusion.py shapes    # the U-Net from diffusion.py, 20,000 steps, ~1.5 hours on an 8-core CPU
    → results_<run>.json and out/*.png; the Shapes model's averaged weights go to out/unet_ema.pt (used by episodes 7–8)

Training is the same in both cases: take a clean example, pick a random step t, noise it with episode 5's jump, and
train the network to predict the noise ε that was added (mean squared error). Samples here use DDIM with 50 steps
(episode 7 compares samplers); quality is scored by the judge from episode 3.
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import diffusion as D
from shapes import make_dataset, save_grid

run = sys.argv[1] if len(sys.argv) > 1 else "shapes"
torch.manual_seed(0)
torch.set_num_threads(min(8, os.cpu_count()))
OUT = "out"
os.makedirs(OUT, exist_ok=True)


def to_images(x):
    return ((x.clamp(-1, 1) + 1) / 2 * 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8)


# ================================================================ 2D: a spiral and a tiny MLP
if run == "toy":
    n = 10_000
    rng = np.random.default_rng(0)
    t_ = np.sqrt(rng.uniform(0, 1, n)) * 3 * math.pi                  # episode 1's spiral
    data = torch.tensor(np.stack([t_ * np.cos(t_), t_ * np.sin(t_)], 1) / (3 * math.pi) + rng.normal(0, 0.03, (n, 2)),
                        dtype=torch.float32) * 2                       # spread it to about ±2

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2 + 32, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                     nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2))

        def forward(self, x, t):
            return self.net(torch.cat([x, D.timestep_embedding(t, 32)], 1))

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(1, 6001):
        x0 = data[torch.randint(0, n, (512,))]
        t = torch.randint(1, D.T + 1, (512,))
        eps = torch.randn_like(x0)
        a = D.ALPHA_BAR[t - 1][:, None]
        loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():                                               # DDPM sampling, all 1,000 steps
        x = torch.randn(2000, 2, generator=torch.Generator().manual_seed(1))
        snaps = {}
        for t in range(D.T, 0, -1):
            eps = model(x, torch.full((len(x),), t))
            a, ab, b = 1 - D.BETA[t - 1], D.ALPHA_BAR[t - 1], D.BETA[t - 1]
            x = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            if t > 1:
                x = x + (b * (1 - D.ALPHA_BAR[t - 2]) / (1 - ab)).sqrt() * torch.randn(x.shape)
            if t - 1 in (1000 - 1, 750, 500, 250, 100, 0):
                snaps[t - 1] = x[:600].numpy().round(3).tolist()
        # how close are samples to the spiral? distance to the nearest training point
        d = torch.cdist(x, data).min(1).values
        noise_d = torch.cdist(torch.randn(2000, 2), data).min(1).values
    res = {"steps": 6000, "loss_first_100": float(np.mean(losses[:100])), "loss_last_500": float(np.mean(losses[-500:])),
           "loss_curve": [float(np.mean(losses[i:i + 100])) for i in range(0, 6000, 100)],
           "sample_nearest_median": d.median().item(), "noise_nearest_median": noise_d.median().item(),
           "data": (data[:600].numpy().round(3)).tolist(), "snapshots": snaps}
    with open("results_toy.json", "w") as f:
        json.dump(res, f)
    print(f"toy: loss {res['loss_first_100']:.3f} → {res['loss_last_500']:.3f}; median distance to the spiral: "
          f"samples {res['sample_nearest_median']:.3f}, pure noise {res['noise_nearest_median']:.3f}")
    sys.exit()

# ================================================================ Shapes: the U-Net
from judge import load_judge, recognizable, to_tensor     # noqa: E402

STEPS, BATCH, LR = int(os.environ.get("STEPS", 20_000)), 128, 2e-4
CHECK = [500, 2000, 5000, 10_000, STEPS]
images, _, _ = make_dataset(20_000, seed=0)
x_all = to_tensor(images) * 2 - 1
judge = load_judge()
model = D.UNet(ch=(32, 64, 128))
ema = copy.deepcopy(model).eval()
opt = torch.optim.Adam(model.parameters(), lr=LR)
params = sum(p.numel() for p in model.parameters())
print(f"U-Net: {params:,} parameters")


def nearest(x01):
    d = torch.cdist(x01.flatten(1), ((x_all + 1) / 2).flatten(1)) ** 2 / x01[0].numel()
    return d.min(1).values.mean().item()


def evaluate(net, n=500):
    g = torch.Generator().manual_seed(1)
    s = (D.sample_ddim(net, n, steps=50, generator=g) + 1) / 2
    probs = [l.softmax(-1) for l in judge(s)]
    ok = torch.stack([p.max(-1).values >= 0.9 for p in probs[:3]] + [probs[3][:, 1] >= 0.5], 1).all(1)
    labels = torch.stack([p.argmax(-1) for p in probs[:3]], 1)
    combos = {tuple(l) for l, k in zip(labels.tolist(), ok.tolist()) if k}
    return s, ok.float().mean().item(), len(combos), nearest(s[:200])


# what the network is asked: a noisy picture at several steps, and the noise inside it
demo_x0 = x_all[:1].repeat(5, 1, 1, 1)
demo_t = torch.tensor([50, 200, 400, 600, 900])
demo_eps = torch.randn(demo_x0.shape, generator=torch.Generator().manual_seed(3))
demo_xt = D.noise(demo_x0, demo_t, demo_eps)
save_grid(to_images(demo_x0[:1]), f"{OUT}/demo_clean.png")

log, history = [], []
t0 = time.time()
for step in range(1, STEPS + 1):
    x0 = x_all[torch.randint(0, len(x_all), (BATCH,))]
    loss = D.training_loss(model, x0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), model.parameters()):
            pe.lerp_(p, 1 - 0.999)
    log.append(loss.item())
    if step in CHECK:
        s, share, combos, near = evaluate(ema)
        with torch.no_grad():                               # the model's guess of the clean picture from x_t
            eps_hat = ema(demo_xt, demo_t)
            ab = D.ALPHA_BAR[demo_t - 1].view(-1, 1, 1, 1)
            x0_hat = ((demo_xt - (1 - ab).sqrt() * eps_hat) / ab.sqrt()).clamp(-1, 1)
            eps_err = F.mse_loss(eps_hat, demo_eps, reduction="none").mean([1, 2, 3]).tolist()
        history.append({"step": step, "loss_last_200": float(np.mean(log[-200:])), "recognizable": share,
                        "combinations": combos, "nearest_train_mse": near, "demo_eps_mse": eps_err,
                        "minutes": (time.time() - t0) / 60})
        save_grid(to_images(s[:16] * 2 - 1), f"{OUT}/samples_step{step:05d}.png")
        save_grid(to_images(torch.cat([demo_xt, (demo_eps / 3).clamp(-1, 1), eps_hat.clamp(-3, 3) / 3, x0_hat])),
                  f"{OUT}/denoise_step{step:05d}.png", cols=5)
        torch.save(ema.state_dict(), f"{OUT}/unet_ema.pt")
        print(f"step {step:5d}: loss {history[-1]['loss_last_200']:.4f}  recognizable {share:.1%}  combinations "
              f"{combos}/135  nearest {near:.4f}  {history[-1]['minutes']:.0f} min", flush=True)

with open("results_shapes.json", "w") as f:
    json.dump({"parameters": params, "steps": STEPS, "batch": BATCH, "lr": LR, "schedule": "cosine, T = 1000",
               "loss_curve": [float(np.mean(log[i:i + 200])) for i in range(0, STEPS, 200)],
               "real_recognizable": recognizable(judge, (x_all[:2000] + 1) / 2)[0],
               "demo_steps": demo_t.tolist(), "history": history}, f, indent=1)
