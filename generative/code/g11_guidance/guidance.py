"""Episode 11: classifier-free guidance. Episode 10's conditional model, sampled at different guidance scales.

    python3 guidance.py                                           # episode 10's model → results.json, out/w*.png
    STEPS=8000 NAME=mid python3 train_cond.py                     # stopped early; keeps out/mid_{2000..8000}_ema.pt
    CKPT=out/mid_6000_ema.pt NAME=mid6k python3 guidance.py       # → results_mid6k.json, out/mid6k_w*.png

Classifier-free guidance (Ho & Salimans 2022): at every step, ask the network twice, once with the description and
once without, and push past the conditional prediction, away from the unconditional one:
    ε̂ = ε̂(no label) + w · (ε̂(label) − ε̂(no label))
w = 0 ignores the description, w = 1 is plain conditioning (episode 10), w > 1 exaggerates whatever the description
changes. All samples: 50 steps with fresh noise (η = 1), the same starting noise for every w.
"late" runs apply guidance only while t < 500 (the less noisy half of sampling) and plain conditioning before
(compare Kynkäänniemi et al. 2024, "Applying guidance in a limited interval").
"""
import json
import os
import sys

import numpy as np
import torch

import diffusion as D
from judge import load_judge
from shapes import save_grid
from train_cond import HELD_OUT, NULL, CondUNet, describe

torch.set_num_threads(int(os.environ.get("THREADS", min(8, os.cpu_count()))))
STEPS = 50
SCALES = [0, 1, 1.5, 2, 3, 5, 8]
judge = load_judge()
model = CondUNet()
model.load_state_dict(torch.load(os.environ.get("CKPT", "../g10_conditioning/out/cond_ema.pt")))
NAME = os.environ.get("NAME", "")
PRE = f"{NAME}_" if NAME else ""
model.eval()


@torch.no_grad()
def sample(y, seed, w, late_only=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(y), 3, 32, 32, generator=g)
    ts = torch.linspace(D.T, 1, STEPS).round().long().tolist()
    clipped = []
    for i, t in enumerate(ts):
        eps = D.predict_eps(model, x, t, y, guidance=float(w) if (t < 500 or not late_only) else 1.0, null=NULL)
        ab = D.ALPHA_BAR[t - 1]
        raw = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        clipped.append((raw.abs() > 1).float().mean().item())
        x0 = raw.clamp(-1, 1)
        ab_s = D.ALPHA_BAR[ts[i + 1] - 1] if i + 1 < len(ts) else torch.tensor(1.0)
        sigma = ((1 - ab_s) / (1 - ab)).sqrt() * (1 - ab / ab_s).sqrt()
        x = ab_s.sqrt() * x0 + (1 - ab_s - sigma ** 2).clamp(min=0).sqrt() * eps
        if i + 1 < len(ts):
            x = x + sigma * torch.randn(x.shape, generator=g)
    return (x.clamp(-1, 1) + 1) / 2, float(np.mean(clipped[-10:]))


@torch.no_grad()
def follows(x01, y):
    probs = [l.softmax(-1) for l in judge(x01)]
    ok = torch.stack([p.max(-1).values >= 0.9 for p in probs[:3]] + [probs[3][:, 1] >= 0.5], 1).all(1)
    lab = torch.stack([p.argmax(-1) for p in probs[:3]], 1)
    return ok, ((lab == y) | (y == NULL)).all(1) & ok, lab


def diversity(x01, groups):
    """Mean squared pixel difference between two samples of the same request, averaged over requests."""
    d = []
    for g in x01.split(groups):
        f = g.flatten(1)
        dist = torch.cdist(f, f) ** 2 / f.shape[1]
        n = len(f)
        d.append((dist.sum() / (n * (n - 1))).item())
    return float(np.mean(d))


def to_img(x01):
    return (x01 * 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8)


if __name__ == "__main__":
    held = set(HELD_OUT)
    trained = [(s, c, p) for s in range(3) for c in range(5) for p in range(9) if (s, c, p) not in held]
    y_tr = torch.tensor([c for c in trained for _ in range(2)])
    y_ho = torch.tensor([c for c in HELD_OUT for _ in range(32)])
    # diversity: four requests, 32 samples each; and "something red" (only the color said), 64 samples
    div_req = [(0, 0, 4), (1, 2, 0), (2, 3, 8), (1, 4, 5)]
    y_div = torch.tensor([c for c in div_req for _ in range(32)])
    y_red = torch.tensor([(NULL, 0, NULL)] * 64)

    res = {"steps": STEPS, "scales": SCALES, "runs": []}
    for w in SCALES:
        x, clip_tr = sample(y_tr, 1, w)
        ok, f, lab_tr = follows(x, y_tr)
        xh, _ = sample(y_ho, 2, w)
        _, fh, _ = follows(xh, y_ho)
        xd, _ = sample(y_div, 3, w)
        okd, fd, _ = follows(xd, y_div)
        xr, _ = sample(y_red, 4, w)
        okr, fr, labr = follows(xr, y_red)
        r = {"w": w, "recognizable": ok.float().mean().item(), "follows": f.float().mean().item(),
             "held_out_follows": fh.float().mean().item(), "diversity": diversity(xd, 32),
             "diversity_follows": fd.float().mean().item(), "clipped_late": clip_tr,
             "red_follows": fr.float().mean().item(),
             "red_shapes": [int(((labr[:, 0] == s) & okr).sum()) for s in range(3)],
             "red_positions": [int(((labr[:, 2] == p) & okr).sum()) for p in range(9)]}
        if w > 1.5:
            xl, _ = sample(y_tr, 1, w, late_only=True)
            okl, fl, labl = follows(xl, y_tr)
            r["late"] = {"recognizable": okl.float().mean().item(), "follows": fl.float().mean().item()}
        r["per_attribute"] = (lab_tr == y_tr).float().mean(0).tolist()
        res["runs"].append(r)
        print(r, flush=True)
        tag = f"{w:g}".replace(".", "p")
        save_grid(to_img(torch.cat([xd[k * 32:k * 32 + 8] for k in range(3)])), f"out/{PRE}w{tag}_requests.png")
        save_grid(to_img(xr[:16]), f"out/{PRE}w{tag}_red.png")
    res["show"] = [describe(c) for c in div_req[:3]]
    with open(f"results_{NAME}.json" if NAME else "results.json", "w") as f:
        json.dump(res, f, indent=1)
