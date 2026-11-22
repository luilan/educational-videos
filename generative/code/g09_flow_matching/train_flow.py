"""Episode 9 experiments: flow matching against diffusion, with the same networks and the same training budget.

    python3 train_flow.py toy       # 2D spiral: diffusion, flow matching and reflow, tiny MLPs, ~10 minutes
    python3 train_flow.py shapes    # the episode 6 U-Net trained with flow matching, 20,000 steps, ~2.5 hours
    → results_<run>.json and out/*.png   (THREADS=4 to share a CPU)

Flow matching (Lipman et al. 2022; "rectified flow", Liu et al. 2022): mix picture and noise on a straight line,
x_t = (1 − t)·x0 + t·ε with t in [0, 1], and train the network to predict the direction of that line, the velocity
v = ε − x0 (mean squared error). To generate, start from noise at t = 1 and take Euler steps x ← x − Δt·v̂ to t = 0.
Diffusion samples are drawn with DDIM (η = 0), which is also a deterministic ODE solver, so both get the same budget
of network calls.
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
torch.set_num_threads(int(os.environ.get("THREADS", min(8, os.cpu_count()))))
OUT = "out"
os.makedirs(OUT, exist_ok=True)


def tscale(t):
    """Continuous t in [0, 1] → the network's step input, on the same 0–1,000 scale as diffusion's steps."""
    return t * D.T


@torch.no_grad()
def sample_flow(model, x, steps, keep_path=False):
    ts = torch.linspace(1, 0, steps + 1)
    path = [x.clone()] if keep_path else None
    for i in range(steps):
        t, dt = ts[i], ts[i] - ts[i + 1]
        x = x - dt * model(x, torch.full((len(x),), tscale(t).item()))
        if keep_path:
            path.append(x.clone())
    return x, path


@torch.no_grad()
def sample_ddim_path(model, x, steps, lim=1.0):
    """DDIM from t = 1,000 down, keeping every intermediate x. The clean guess is clipped to the data range (±lim),
    as in diffusion.py: at t = 1,000 almost nothing of the picture is left and dividing by √ᾱ would blow up."""
    ts = torch.linspace(D.T, 1, steps).round().long().tolist()
    path = [x.clone()]
    for i, t in enumerate(ts):
        eps = model(x, torch.full((len(x),), t))
        ab = D.ALPHA_BAR[t - 1]
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-lim, lim)
        ab_s = D.ALPHA_BAR[ts[i + 1] - 1] if i + 1 < len(ts) else torch.tensor(1.0)
        x = ab_s.sqrt() * x0 + (1 - ab_s).sqrt() * eps
        path.append(x.clone())
    return x, path


# ================================================================ 2D: the spiral
if run == "toy":
    n = 10_000
    rng = np.random.default_rng(0)
    t_ = np.sqrt(rng.uniform(0, 1, n)) * 3 * math.pi
    data = torch.tensor(np.stack([t_ * np.cos(t_), t_ * np.sin(t_)], 1) / (3 * math.pi) + rng.normal(0, 0.03, (n, 2)),
                        dtype=torch.float32) * 2

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2 + 32, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                     nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2))

        def forward(self, x, t):
            return self.net(torch.cat([x, D.timestep_embedding(t, 32)], 1))

    models = {}
    for method in ("diffusion", "flow", "reflow"):
        torch.manual_seed(0)
        if method == "reflow":
            # rectified flow's "reflow": pair each noise with the point the trained flow carries it to, and train a
            # new flow on straight lines between those pairs (no other pairing of noise and data is mixed in)
            pair_eps = torch.randn(n, 2, generator=torch.Generator().manual_seed(7))
            pair_x0 = sample_flow(models["flow"], pair_eps.clone(), 100)[0]
        model = MLP()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for step in range(1, 10_001):
            idx = torch.randint(0, n, (512,))
            x0 = data[idx]
            eps = torch.randn_like(x0)
            if method == "reflow":
                x0, eps = pair_x0[idx], pair_eps[idx]
            if method == "diffusion":
                t = torch.randint(1, D.T + 1, (512,))
                a = D.ALPHA_BAR[t - 1][:, None]
                loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t), eps)
            else:
                t = torch.rand(512)
                loss = F.mse_loss(model((1 - t[:, None]) * x0 + t[:, None] * eps, tscale(t)), eps - x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        models[method] = model.eval()

    start = torch.randn(2000, 2, generator=torch.Generator().manual_seed(1))
    res = {"data": data[:600].numpy().round(3).tolist(), "by_steps": [], "paths": {}}
    LIM = 2.5                                                  # the spiral lies within about ±2.1
    ref = data[torch.randperm(n, generator=torch.Generator().manual_seed(5))[:2000]]
    for steps in (1, 2, 4, 8, 16, 50, 200):
        row = {"steps": steps}
        for method in ("diffusion", "flow", "reflow"):
            x = (sample_ddim_path(models[method], start.clone(), steps, LIM) if method == "diffusion"
                 else sample_flow(models[method], start.clone(), steps))[0]
            row[method] = {"precision": torch.cdist(x, data).min(1).values.median().item(),   # samples → spiral
                           "coverage": torch.cdist(ref, x).min(1).values.median().item()}     # spiral → samples
            if steps in (1, 4, 16):
                row[method]["points"] = x[:600].numpy().round(3).tolist()
        res["by_steps"].append(row)
        print(f"{steps:4d} steps (sample→spiral / spiral→sample): " + "   ".join(
            f"{m} {row[m]['precision']:.3f} / {row[m]['coverage']:.3f}" for m in ("diffusion", "flow", "reflow")), flush=True)
    # paths of 40 points with 50 steps; straightness = straight-line distance / path length
    for method in ("diffusion", "flow", "reflow"):
        f = sample_ddim_path if method == "diffusion" else sample_flow
        _, path = f(models[method], start.clone(), 50, LIM) if method == "diffusion" else f(models[method], start.clone(), 50, True)
        p = torch.stack(path)                                  # (51, 2000, 2)
        length = (p[1:] - p[:-1]).norm(dim=-1).sum(0)
        chord = (p[-1] - p[0]).norm(dim=-1)
        res["paths"][method] = {"straightness_median": (chord / length).median().item(),
                                "points": p[:, :40].numpy().round(3).tolist()}
        print(f"{method}: median straightness {res['paths'][method]['straightness_median']:.3f}")
    res["noise"] = {"precision": torch.cdist(start, data).min(1).values.median().item(),
                    "coverage": torch.cdist(ref, start).min(1).values.median().item()}
    with open("results_toy.json", "w") as f:
        json.dump(res, f)
    sys.exit()

# ================================================================ Shapes: the same U-Net, trained with flow matching
from judge import load_judge, to_tensor     # noqa: E402

STEPS, BATCH, LR = int(os.environ.get("STEPS", 20_000)), 128, 2e-4
CHECK = [2000, 5000, 10_000, STEPS]
EVAL_STEPS = [5, 10, 20, 50]
images, _, _ = make_dataset(20_000, seed=0)
x_all = to_tensor(images) * 2 - 1
judge = load_judge()
model = D.UNet(ch=(32, 64, 128))
ema = copy.deepcopy(model).eval()
opt = torch.optim.Adam(model.parameters(), lr=LR)
params = sum(p.numel() for p in model.parameters())
print(f"U-Net: {params:,} parameters", flush=True)


def to_images(x):
    return ((x.clamp(-1, 1) + 1) / 2 * 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8)


def score(x, n_near=200):
    s = (x.clamp(-1, 1) + 1) / 2
    probs = [l.softmax(-1) for l in judge(s)]
    ok = torch.stack([p.max(-1).values >= 0.9 for p in probs[:3]] + [probs[3][:, 1] >= 0.5], 1).all(1)
    labels = torch.stack([p.argmax(-1) for p in probs[:3]], 1)
    combos = {tuple(l) for l, k in zip(labels.tolist(), ok.tolist()) if k}
    d = torch.cdist(s[:n_near].flatten(1), ((x_all + 1) / 2).flatten(1)) ** 2 / s[0].numel()
    return {"recognizable": ok.float().mean().item(), "combinations": len(combos),
            "nearest_train_mse": d.min(1).values.mean().item()}


start = torch.randn(500, 3, 32, 32, generator=torch.Generator().manual_seed(1))    # the same noise as episode 7
log, history = [], []
t0 = time.time()
for step in range(1, STEPS + 1):
    x0 = x_all[torch.randint(0, len(x_all), (BATCH,))]
    eps = torch.randn_like(x0)
    t = torch.rand(BATCH)
    tt = t.view(-1, 1, 1, 1)
    loss = F.mse_loss(model((1 - tt) * x0 + tt * eps, tscale(t)), eps - x0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), model.parameters()):
            pe.lerp_(p, 1 - 0.999)
    log.append(loss.item())
    if step in CHECK:
        h = {"step": step, "loss_last_200": float(np.mean(log[-200:])), "minutes": (time.time() - t0) / 60}
        for k in (EVAL_STEPS if step == STEPS else [50]):
            x, _ = sample_flow(ema, start.clone(), k)
            h[f"euler{k}"] = score(x)
            save_grid(to_images(x[:16]), f"{OUT}/flow_euler{k:02d}_step{step:05d}.png")
        history.append(h)
        torch.save(ema.state_dict(), f"{OUT}/flow_ema.pt")
        print(f"step {step:5d}: loss {h['loss_last_200']:.4f}  " + "  ".join(
            f"euler{k}: {h[f'euler{k}']['recognizable']:.1%} ({h[f'euler{k}']['combinations']})"
            for k in (EVAL_STEPS if step == STEPS else [50])) + f"  {h['minutes']:.0f} min", flush=True)

# diffusion (episode 6's model) at the same few-step budgets, from the same starting noise
diff = D.UNet(ch=(32, 64, 128))
diff.load_state_dict(torch.load("../g06_learning_to_denoise/out/unet_ema.pt"))
diff.eval()
compare = {}
for k in EVAL_STEPS:
    x, _ = sample_ddim_path(diff, start.clone(), k)
    compare[f"ddim{k}"] = score(x)
    save_grid(to_images(x[:16]), f"{OUT}/diffusion_ddim{k:02d}.png")
    print(f"diffusion DDIM {k}: {compare[f'ddim{k}']['recognizable']:.1%} ({compare[f'ddim{k}']['combinations']})")
with open("results_shapes.json", "w") as f:
    json.dump({"parameters": params, "steps": STEPS, "batch": BATCH, "lr": LR,
               "loss_curve": [float(np.mean(log[i:i + 200])) for i in range(0, STEPS, 200)],
               "history": history, "diffusion_same_noise": compare}, f, indent=1)
