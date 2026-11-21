"""Episode 5 experiments: diffusion's forward process, which slowly turns a picture into noise.

    python3 forward_noise.py          # about a minute on a CPU; writes results.json and out/*.png

Pixels are scaled to [−1, 1]. One step adds a little Gaussian noise and shrinks the picture a little:
    x_t = sqrt(1 − β_t) · x_{t−1} + sqrt(β_t) · ε
With ᾱ_t = (1 − β_1)(1 − β_2)…(1 − β_t), you can jump to any step in one go:
    x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 − ᾱ_t) · ε
We check that both give the same distribution, and compare two schedules for β over T = 1,000 steps:
  - linear (DDPM, 2020): β from 0.0001 to 0.02;
  - cosine (Nichol & Dhariwal, 2021): ᾱ_t = cos²((t/T + 0.008) / 1.008 · π/2), normalized to ᾱ_0 = 1.
For each we record the signal-to-noise ratio ᾱ/(1 − ᾱ) and the share of noised pictures whose shape, color and
position the judge (episode 3) still reads correctly.
"""
import json
import math
import os

import numpy as np
import torch

from judge import load_judge, recognizable, to_tensor
from shapes import make_dataset, save_grid

torch.manual_seed(0)
OUT = "out"
os.makedirs(OUT, exist_ok=True)
T = 1000


def linear_alpha_bar():
    beta = torch.linspace(1e-4, 0.02, T)
    return torch.cumprod(1 - beta, 0), beta


def cosine_alpha_bar(s=0.008):
    t = torch.arange(T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0])[1:]
    beta = (1 - ab / torch.cat([torch.ones(1), ab[:-1]])).clamp(max=0.999)
    return torch.cumprod(1 - beta, 0), beta


def to_images(x):                                    # [−1, 1] → uint8
    return ((x.clamp(-1, 1) + 1) / 2 * 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8)


images, captions, labels = make_dataset(2000, seed=1)
labels = torch.tensor(labels)
x0 = to_tensor(images) * 2 - 1
judge = load_judge()
res = {"T": T, "caption": captions[0]}

# ---------------------------------------------------------------- 1. step by step vs one jump (linear schedule)
ab_lin, beta_lin = linear_alpha_bar()
x = x0[:1].repeat(2000, 1, 1, 1)                     # the same picture, 2,000 independent noisings
for t in range(500):                                 # 500 small steps
    x = torch.sqrt(1 - beta_lin[t]) * x + torch.sqrt(beta_lin[t]) * torch.randn_like(x)
jump = torch.sqrt(ab_lin[499]) * x0[:1] + torch.sqrt(1 - ab_lin[499]) * torch.randn(2000, 3, 32, 32)
res["step_vs_jump_t500"] = {
    "alpha_bar": ab_lin[499].item(),
    "steps_mean_of_pixel_means": x.mean(0).mean().item(), "jump_mean_of_pixel_means": jump.mean(0).mean().item(),
    "steps_mean_std": x.std(0).mean().item(), "jump_mean_std": jump.std(0).mean().item(),
    "max_abs_diff_of_pixel_means": (x.mean(0) - jump.mean(0)).abs().max().item(),
    "expected_std": math.sqrt(1 - ab_lin[499].item())}

# ---------------------------------------------------------------- 2. the two schedules
ab_cos, _ = cosine_alpha_bar()
show = [0, 50, 100, 200, 300, 500, 700, 1000]
for name, ab in [("linear", ab_lin), ("cosine", ab_cos)]:
    g = torch.Generator().manual_seed(2)
    eps = torch.randn(1, 3, 32, 32, generator=g)
    row = [x0[0]] + [(torch.sqrt(ab[t - 1]) * x0[0] + torch.sqrt(1 - ab[t - 1]) * eps[0]) for t in show[1:]]
    save_grid(to_images(torch.stack(row)), f"{OUT}/noising_{name}.png", cols=len(show))
    curve = []
    for t in range(0, T + 1, 25):
        if t == 0:
            xt, a = x0, 1.0
        else:
            a = ab[t - 1].item()
            xt = math.sqrt(a) * x0 + math.sqrt(1 - a) * torch.randn(x0.shape, generator=g)
        _, read = recognizable(judge, ((xt + 1) / 2).clamp(0, 1))
        correct = (read == labels).all(1).float().mean().item()          # shape, color and position all right
        snr = a / (1 - a) if a < 1 else float("inf")
        curve.append({"t": t, "alpha_bar": a, "snr_db": 10 * math.log10(snr) if t else None, "read_correctly": correct})
    last = max(c["t"] for c in curve if c["read_correctly"] >= 0.5)
    res[name] = {"curve": curve, "last_t_half_read": last,
                 "alpha_bar_at": {t: ab[t - 1].item() for t in show[1:]},
                 "final_alpha_bar": ab[-1].item(),
                 "steps_with_snr_below_minus20db": int((ab / (1 - ab) < 0.01).sum())}
    print(name, "judge reads half the pictures correctly until t =", last, "| ᾱ_T =", f"{ab[-1].item():.2e}",
          "| steps with SNR < −20 dB:", res[name]["steps_with_snr_below_minus20db"])

with open("results.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res["step_vs_jump_t500"], indent=1))
