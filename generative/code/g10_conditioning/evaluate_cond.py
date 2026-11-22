"""Episode 10: does the conditional model draw what it is told?

    python3 evaluate_cond.py     # ~15 minutes on an 8-core CPU → results_cond.json, out/*.png  (needs out/cond_ema.pt)

All samples: 50 steps with fresh noise (η = 1, episode 7's better sampler), plain conditioning (no guidance; that is
episode 11). "Follows the request" = the judge recognizes the picture AND reads the requested shape, color and position.
"""
import json
import os

import numpy as np
import torch

import diffusion as D
from judge import load_judge, to_tensor
from shapes import COLOR_NAMES, POSITIONS, SHAPES, make_dataset, save_grid
from train_cond import HELD_OUT, NULL, CondUNet, describe

torch.set_num_threads(int(os.environ.get("THREADS", min(8, os.cpu_count()))))
STEPS = 50
judge = load_judge()
model = CondUNet()
model.load_state_dict(torch.load("out/cond_ema.pt"))
model.eval()
train01 = to_tensor(make_dataset(20_000, seed=0)[0])


@torch.no_grad()
def sample(y, seed, guidance=None):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(y), 3, 32, 32, generator=g)
    ts = torch.linspace(D.T, 1, STEPS).round().long().tolist()
    for i, t in enumerate(ts):
        eps = D.predict_eps(model, x, t, y, guidance, NULL)
        ab = D.ALPHA_BAR[t - 1]
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        ab_s = D.ALPHA_BAR[ts[i + 1] - 1] if i + 1 < len(ts) else torch.tensor(1.0)
        sigma = ((1 - ab_s) / (1 - ab)).sqrt() * (1 - ab / ab_s).sqrt()
        x = ab_s.sqrt() * x0 + (1 - ab_s - sigma ** 2).clamp(min=0).sqrt() * eps
        if i + 1 < len(ts):
            x = x + sigma * torch.randn(x.shape, generator=g)
    return (x.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def read(x01):
    probs = [l.softmax(-1) for l in judge(x01)]
    ok = torch.stack([p.max(-1).values >= 0.9 for p in probs[:3]] + [probs[3][:, 1] >= 0.5], 1).all(1)
    return ok, torch.stack([p.argmax(-1) for p in probs[:3]], 1)


def follows(x01, y):
    ok, lab = read(x01)
    said = y != NULL
    match = ((lab == y) | ~said).all(1) & ok
    per = {name: ((lab[:, k] == y[:, k]) & ok)[said[:, k]].float().mean().item()
           for k, name in enumerate(("shape", "color", "position")) if said[:, k].any()}
    return ok, lab, match, per


def to_img(x01):
    return (x01 * 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8)


res = {"steps": STEPS, "sampler": "fresh noise (eta = 1)"}
held = set(HELD_OUT)
trained = [(s, c, p) for s in range(3) for c in range(5) for p in range(9) if (s, c, p) not in held]

# 1. every trained combination, 4 samples each
y = torch.tensor([c for c in trained for _ in range(4)])
x = sample(y, 1)
ok, lab, match, per = follows(x, y)
res["trained"] = {"combinations": len(trained), "samples": len(y), "recognizable": ok.float().mean().item(),
                  "follows": match.float().mean().item(), "per_attribute": per}
print("trained", res["trained"], flush=True)

# 2. the eight combinations it never saw, 50 samples each
y = torch.tensor([c for c in HELD_OUT for _ in range(50)])
x = sample(y, 2)
ok, lab, match, per = follows(x, y)
res["held_out"] = {"samples": len(y), "recognizable": ok.float().mean().item(), "follows": match.float().mean().item(),
                   "per_attribute": per,
                   "by_combination": [{"request": describe(c), "follows": match[i * 50:(i + 1) * 50].float().mean().item()}
                                      for i, c in enumerate(HELD_OUT)]}
print("held out", {k: v for k, v in res["held_out"].items() if k != "by_combination"}, flush=True)
for b in res["held_out"]["by_combination"]:
    print("   ", b, flush=True)
save_grid(to_img(torch.cat([x[i * 50:i * 50 + 8] for i in range(4)])), "out/held_out.png")

# 3. four trained requests, one grid row each (for the video)
show = [(2, 0, 4), (0, 3, 0), (1, 2, 8), (0, 4, 5)]          # red triangle center, yellow circle top left, ...
y = torch.tensor([c for c in show for _ in range(8)])
save_grid(to_img(sample(y, 3)), "out/requests.png")
res["requests_png"] = [describe(c) for c in show]

# 4. part of a description: only "red", only "triangle", only "top right"; 200 samples each
partial = {"red": (NULL, 0, NULL), "triangle": (2, NULL, NULL), "top right": (NULL, NULL, 2)}
res["partial"] = {}
for k, (name, c) in enumerate(partial.items()):
    y = torch.tensor([c] * 200)
    x = sample(y, 10 + k)
    ok, lab, match, per = follows(x, y)
    free = [i for i in range(3) if c[i] == NULL]
    names = (SHAPES, COLOR_NAMES, POSITIONS)
    spread = {("shape", "color", "position")[i]: {names[i][v]: int(((lab[:, i] == v) & ok).sum()) for v in range(len(names[i]))}
              for i in free}
    res["partial"][name] = {"recognizable": ok.float().mean().item(), "follows": match.float().mean().item(),
                            "spread_of_unsaid": spread}
    print("partial", name, res["partial"][name], flush=True)
    save_grid(to_img(x[:8]), f"out/partial_{name.replace(' ', '_')}.png")

# 5. no description at all (every label dropped): the same network as an unconditional model
y = torch.full((500, 3), NULL)
x = sample(y, 20)
ok, lab, _, _ = follows(x, y)
combos = {tuple(l) for l, k in zip(lab.tolist(), ok.tolist()) if k}
res["unconditional"] = {"samples": 500, "recognizable": ok.float().mean().item(), "combinations": len(combos)}
print("unconditional", res["unconditional"], flush=True)
save_grid(to_img(x[:16]), "out/unconditional.png")

with open("results_cond.json", "w") as f:
    json.dump(res, f, indent=1)
