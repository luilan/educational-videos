"""Episode 11: the pictures for the video (run after guidance.py and the mid6k sweep).

    python3 figures.py     # ~10 minutes → out/fixes.png, out/ladder.png, out/predictions.png, figures.json

fixes.png       the half-trained model (6,000 steps): requests it gets wrong at w = 1 and right at w = 1.5, from the
                same starting noise (row 1: w = 1, row 2: w = 1.5). Same pictures as results_mid6k.json's w = 1 and 1.5.
ladder.png      the finished model (episode 10's): eight requests, same noise; rows w = 1, 2, 3, and 3 guided only
                while t < 500.
predictions.png one sampling run of "red circle, center" with the finished model at w = 3: at five steps, the
                clean-picture guess without the request, with it, and the guided one (rows), clipped to the pixel range.
"""
import json
import os

import torch

os.environ.setdefault("CKPT", "out/mid_6000_ema.pt")
import diffusion as D                                    # noqa: E402
import guidance as G                                     # noqa: E402
from shapes import save_grid                             # noqa: E402
from train_cond import HELD_OUT, NULL, describe          # noqa: E402

out = {}
held = set(HELD_OUT)
trained = [(s, c, p) for s in range(3) for c in range(5) for p in range(9) if (s, c, p) not in held]

# 1. what guidance fixes in the half-trained model (same requests, seed and noise as guidance.py)
y = torch.tensor([c for c in trained for _ in range(2)])
x1, _ = G.sample(y, 1, 1)
x15, _ = G.sample(y, 1, 1.5)
_, f1, lab1 = G.follows(x1, y)
ok15, f15, _ = G.follows(x15, y)
fixed = (~f1 & f15).nonzero().flatten()
broke = (f1 & ~f15).nonzero().flatten()
pick = fixed[torch.randperm(len(fixed), generator=torch.Generator().manual_seed(0))[:8]].sort().values
save_grid(G.to_img(torch.cat([x1[pick], x15[pick]])), "out/fixes.png")
names = ["circle", "square", "triangle"]
out["fixes"] = {"pictures": len(y), "fixed": len(fixed), "broken": len(broke),
                "shown": [{"request": describe(tuple(y[i].tolist())),
                           "w1_judge": None if not G.follows(x1[i:i + 1], y[i:i + 1])[0].item() else describe(tuple(lab1[i].tolist()))}
                          for i in pick.tolist()]}
print(out["fixes"], flush=True)

# 2. the finished model at higher w
G.model.load_state_dict(torch.load("../g10_conditioning/out/cond_ema.pt"))
req = [(0, 0, 4), (1, 2, 0), (2, 3, 8), (1, 4, 5), (2, 1, 2), (0, 3, 6), (1, 0, 7), (2, 2, 1)]
y = torch.tensor(req)
rows = [G.sample(y, 7, w, late_only=late)[0] for w, late in ((1, False), (2, False), (3, False), (3, True))]
save_grid(G.to_img(torch.cat(rows)), "out/ladder.png")
out["ladder"] = {"requests": [describe(c) for c in req],
                 "rows": ["w = 1", "w = 2", "w = 3", "w = 3, only t < 500"],
                 "follows": [G.follows(r, y)[1].tolist() for r in rows]}
print(out["ladder"], flush=True)

# 3. two predictions inside one run
with torch.no_grad():
    y = torch.tensor([(0, 0, 4)])
    g = torch.Generator().manual_seed(11)
    x = torch.randn(1, 3, 32, 32, generator=g)
    ts = torch.linspace(D.T, 1, G.STEPS).round().long().tolist()
    show = [ts[k] for k in (5, 15, 25, 35, 45)]
    guesses = {"no request": [], "request": [], "guided": []}
    for i, t in enumerate(ts):
        tt = torch.full((1,), t, dtype=torch.long)
        cond = G.model(x, tt, y)
        unc = G.model(x, tt, torch.full_like(y, NULL))
        eps = unc + 3 * (cond - unc)
        ab = D.ALPHA_BAR[t - 1]
        if t in show:
            for k, e in (("no request", unc), ("request", cond), ("guided", eps)):
                guesses[k].append(((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1))
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        ab_s = D.ALPHA_BAR[ts[i + 1] - 1] if i + 1 < len(ts) else torch.tensor(1.0)
        sigma = ((1 - ab_s) / (1 - ab)).sqrt() * (1 - ab / ab_s).sqrt()
        x = ab_s.sqrt() * x0 + (1 - ab_s - sigma ** 2).clamp(min=0).sqrt() * eps
        if i + 1 < len(ts):
            x = x + sigma * torch.randn(x.shape, generator=g)
    grid = torch.cat([torch.cat(guesses[k]) for k in guesses])
    save_grid(G.to_img((grid + 1) / 2), "out/predictions.png", cols=5)
    out["predictions"] = {"request": describe((0, 0, 4)), "w": 3, "steps_t": show, "rows": list(guesses)}

with open("figures.json", "w") as f:
    json.dump(out, f, indent=1)
