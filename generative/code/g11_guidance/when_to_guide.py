"""Episode 11: in which steps does guidance break the finished model?

    python3 when_to_guide.py     # ~10 minutes → results_when.json

Episode 10's finished model, the 127 trained combinations once each (seed 1), w = 3 applied in different ranges of
steps (w = 1 elsewhere). Also: the share of pixels in the clean-picture guess that fall outside [−1, 1] and must be
clipped, at a few steps, for w = 1 and w = 3 guided at every step.
"""
import json

import torch

import diffusion as D
import guidance as G                     # loads episode 10's finished model and the judge
from train_cond import HELD_OUT, NULL

held = set(HELD_OUT)
y = torch.tensor([(s, c, p) for s in range(3) for c in range(5) for p in range(9) if (s, c, p) not in held])


@torch.no_grad()
def run(w, lo, hi, seed=1):
    """Guide with w while lo <= t < hi; return follows and, per step, the clipped share of the clean-picture guess."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(y), 3, 32, 32, generator=g)
    ts = torch.linspace(D.T, 1, G.STEPS).round().long().tolist()
    clipped = {}
    for i, t in enumerate(ts):
        eps = D.predict_eps(G.model, x, t, y, guidance=float(w) if lo <= t < hi else 1.0, null=NULL)
        ab = D.ALPHA_BAR[t - 1]
        raw = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        clipped[t] = (raw.abs() > 1).float().mean().item()
        x0 = raw.clamp(-1, 1)
        ab_s = D.ALPHA_BAR[ts[i + 1] - 1] if i + 1 < len(ts) else torch.tensor(1.0)
        sigma = ((1 - ab_s) / (1 - ab)).sqrt() * (1 - ab / ab_s).sqrt()
        x = ab_s.sqrt() * x0 + (1 - ab_s - sigma ** 2).clamp(min=0).sqrt() * eps
        if i + 1 < len(ts):
            x = x + sigma * torch.randn(x.shape, generator=g)
    ok, f, lab = G.follows((x.clamp(-1, 1) + 1) / 2, y)
    return {"recognizable": ok.float().mean().item(), "follows": f.float().mean().item(),
            "per_attribute": (lab == y).float().mean(0).tolist()}, clipped


res = {"pictures": len(y), "w": 3, "ranges": {}}
for name, lo, hi in (("never (w = 1)", 0, 0), ("every step", 0, 10_000), ("only t >= 500", 500, 10_000),
                     ("only t < 500", 0, 500), ("only 100 <= t < 700", 100, 700)):
    r, clipped = run(3, lo, hi)
    res["ranges"][name] = r
    if name in ("never (w = 1)", "every step"):
        res.setdefault("clipped_share", {})[name] = {t: clipped[t] for t in sorted(clipped, reverse=True)[::7]}
    print(name, r, flush=True)
with open("results_when.json", "w") as f:
    json.dump(res, f, indent=1)
