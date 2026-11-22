"""Episode 7 experiments: how many steps does sampling need, and does the fresh noise matter?

    python3 sample_compare.py        # ~30 minutes on an 8-core CPU → results.json and out/*.png

Uses episode 6's trained U-Net (../g06_learning_to_denoise/out/unet_ema.pt). One sampler covers both classic ones:
from step t to an earlier step s, guess the clean picture x0 from the predicted noise, then

    x_s = √ᾱ_s · x0  +  √(1 − ᾱ_s − σ²) · ε̂  +  σ · z,      σ = η · √((1 − ᾱ_s)/(1 − ᾱ_t)) · √(1 − ᾱ_t/ᾱ_s)

η = 1 adds fresh noise z at every step (DDPM, Ho et al. 2020; with all 1,000 steps it is exactly DDPM's ancestral
sampler); η = 0 adds none and is deterministic (DDIM, Song et al. 2020). Both can skip steps.
"""
import json
import os
import time

import numpy as np
import torch

import diffusion as D
from judge import load_judge, to_tensor
from shapes import make_dataset, save_grid

torch.set_num_threads(min(8, os.cpu_count()))
OUT = "out"
os.makedirs(OUT, exist_ok=True)
N = 500
STEP_COUNTS = [10, 20, 50, 100, 250, 1000]

model = D.UNet(ch=(32, 64, 128))
model.load_state_dict(torch.load("../g06_learning_to_denoise/out/unet_ema.pt"))
model.eval()
judge = load_judge()
images, _, _ = make_dataset(20_000, seed=0)
train01 = to_tensor(images)


def to_images(x):
    return ((x.clamp(-1, 1) + 1) / 2 * 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8)


@torch.no_grad()
def sample(x, steps, eta, generator=None, record=()):
    """Go from noise x at t = 1,000 to a picture in `steps` steps. record: steps t at which to keep (x_t, x0 guess)."""
    ts = torch.linspace(D.T, 1, steps).round().long().tolist()
    kept = {}
    for i, t in enumerate(ts):
        eps = D.predict_eps(model, x, t)
        ab = D.ALPHA_BAR[t - 1]
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        if t in record:
            kept[t] = (x.clone(), x0.clone())
        ab_s = D.ALPHA_BAR[ts[i + 1] - 1] if i + 1 < len(ts) else torch.tensor(1.0)
        sigma = eta * ((1 - ab_s) / (1 - ab)).sqrt() * (1 - ab / ab_s).sqrt()
        x = ab_s.sqrt() * x0 + (1 - ab_s - sigma ** 2).clamp(min=0).sqrt() * eps
        if eta > 0 and i + 1 < len(ts):
            x = x + sigma * torch.randn(x.shape, generator=generator)
    return x.clamp(-1, 1), kept


@torch.no_grad()
def readings(x01):
    probs = [l.softmax(-1) for l in judge(x01)]
    ok = torch.stack([p.max(-1).values >= 0.9 for p in probs[:3]] + [probs[3][:, 1] >= 0.5], 1).all(1)
    labels = torch.stack([p.argmax(-1) for p in probs[:3]], 1)
    return ok, labels


def score(x):
    x01 = (x + 1) / 2
    ok, labels = readings(x01)
    combos = {tuple(l) for l, k in zip(labels.tolist(), ok.tolist()) if k}
    d = torch.cdist(x01[:200].flatten(1), train01.flatten(1)) ** 2 / x01[0].numel()
    return {"recognizable": ok.float().mean().item(), "combinations": len(combos),
            "nearest_train_mse": d.min(1).values.mean().item()}


res = {"n": N, "runs": []}
start = torch.randn(N, 3, 32, 32, generator=torch.Generator().manual_seed(1))

# ---------------------------------------------------------------- 1. quality and time vs number of steps
for eta in (0.0, 1.0):
    for steps in STEP_COUNTS:
        t0 = time.time()
        x, _ = sample(start.clone(), steps, eta, generator=torch.Generator().manual_seed(2))
        sec = time.time() - t0
        r = {"eta": eta, "steps": steps, "seconds": sec, "ms_per_sample": 1000 * sec / N, **score(x)}
        res["runs"].append(r)
        save_grid(to_images(x[:16]), f"{OUT}/samples_eta{int(eta)}_steps{steps:04d}.png")
        print(f"eta {eta:.0f} steps {steps:4d}: recognizable {r['recognizable']:.1%}  combinations {r['combinations']:3d}"
              f"  nearest {r['nearest_train_mse']:.4f}  {sec:6.1f} s ({r['ms_per_sample']:.1f} ms/sample)", flush=True)

# ---------------------------------------------------------------- 2. the same starting noise
few = start[:8]
rows = []
for steps in (10, 20, 50, 1000):
    rows.append(sample(few.clone(), steps, 0.0)[0])                                  # DDIM: same picture?
for seed in (2, 3):
    rows.append(sample(few.clone(), 1000, 1.0, generator=torch.Generator().manual_seed(seed))[0])   # DDPM, twice
save_grid(to_images(torch.cat(rows)), f"{OUT}/same_noise.png", cols=8)
res["same_noise_rows"] = ["DDIM 10", "DDIM 20", "DDIM 50", "DDIM 1000", "DDPM 1000 (noise A)", "DDPM 1000 (noise B)"]
d = lambda a, b: ((a - b) ** 2).mean().item() / 4                                   # in [0, 1] pixel units
res["same_noise_mse"] = {"ddim50_vs_ddim1000": d(rows[2], rows[3]), "ddim10_vs_ddim1000": d(rows[0], rows[3]),
                         "ddpmA_vs_ddpmB": d(rows[4], rows[5])}
_, la = readings((rows[4] + 1) / 2)
_, lb = readings((rows[5] + 1) / 2)
_, l50 = readings((rows[2] + 1) / 2)
_, l1000 = readings((rows[3] + 1) / 2)
res["same_noise_same_reading"] = {"ddim50_vs_ddim1000": (l50 == l1000).all(1).sum().item(),
                                  "ddpmA_vs_ddpmB": (la == lb).all(1).sum().item(), "of": 8}

# ---------------------------------------------------------------- 3. when are the decisions made? (DDPM, 1,000 steps)
REC = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 1]
x, kept = sample(start.clone(), 1000, 1.0, generator=torch.Generator().manual_seed(2), record=REC)
ok, final = readings((x + 1) / 2)
decided = []
for t in REC:
    _, lab = readings((kept[t][1] + 1) / 2)
    same = (lab == final)[ok]                       # only samples the judge can read at the end
    decided.append({"t": t, "position": same[:, 2].float().mean().item(), "color": same[:, 1].float().mean().item(),
                    "shape": same[:, 0].float().mean().item()})
    print(f"t {t:4d}: final reading already in the clean guess — position {decided[-1]['position']:.0%}  "
          f"color {decided[-1]['color']:.0%}  shape {decided[-1]['shape']:.0%}", flush=True)
res["decided"] = decided
res["decided_n"] = int(ok.sum())
show = [1000, 800, 600, 500, 400, 300, 200, 1]
traj = torch.cat([torch.stack([kept[t][0][0] for t in show]), torch.stack([kept[t][1][0] for t in show])])
save_grid(to_images(traj), f"{OUT}/trajectory.png", cols=8)
res["trajectory_steps"] = show

with open("results.json", "w") as f:
    json.dump(res, f, indent=1)
