"""Episode 11: how much does guidance help a model that has not finished training?

    STEPS=8000 NAME=mid python3 train_cond.py     # saves out/mid_{2000,4000,6000,8000}_ema.pt
    python3 scan_checkpoints.py                   # → results_scan.json

For each checkpoint (and episode 10's finished model): one sample of each of the 127 trained combinations at
w = 1, 2, 3, with guidance at every step and with guidance only while t < 500. Same sampler and judge as guidance.py.
"""
import glob
import json
import os
import re

import torch

os.environ.setdefault("CKPT", "../g10_conditioning/out/cond_ema.pt")
import guidance as G                                     # noqa: E402  (loads the judge and a model)

ckpts = sorted(glob.glob("out/mid_*_ema.pt"), key=lambda p: int(re.findall(r"_(\d+)_", p)[0]))
ckpts.append("../g10_conditioning/out/cond_ema.pt")
held = set(G.HELD_OUT)
y = torch.tensor([(s, c, p) for s in range(3) for c in range(5) for p in range(9) if (s, c, p) not in held])
res = []
for path in ckpts:
    G.model.load_state_dict(torch.load(path))
    steps = int(re.findall(r"_(\d+)_", path)[0]) if "mid_" in path else 20000
    row = {"steps": steps, "checkpoint": path}
    for w in (1, 2, 3):
        for late in ((False,) if w == 1 else (False, True)):
            x, _ = G.sample(y, 1, w, late_only=late)
            ok, f, lab = G.follows(x, y)
            row[f"w{w}{'_late' if late else ''}"] = {"recognizable": ok.float().mean().item(), "follows": f.float().mean().item(),
                                                     "per_attribute": (lab == y).float().mean(0).tolist()}
    res.append(row)
    print(steps, {k: (round(v["recognizable"], 3), round(v["follows"], 3)) for k, v in row.items() if k.startswith("w")}, flush=True)
with open("results_scan.json", "w") as f:
    json.dump(res, f, indent=1)
