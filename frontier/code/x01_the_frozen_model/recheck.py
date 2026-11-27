"""Do the single-number edits that raised the 250-question probe score also help on all 1,000 test questions?
Reads model.pt and drops.npy from frozen.py; adds "improvements" to results.json."""
import json

import numpy as np
import torch

from arith import TinyGPT, accuracy, split

torch.set_num_threads(8)
train, test = split()
model = TinyGPT()
model.load_state_dict(torch.load("model.pt"))
model.eval()
params = dict(model.named_parameters())
names = list(params)
offsets = np.cumsum([0] + [params[n].numel() for n in names])
drops = np.load("drops.npy")
res = json.load(open("results.json"))
gains = -drops
base_probe, base_test = res["acc_probe"], res["acc_test"]
hist = {f"+{k} question{'s' if k > 1 else ''}": int((np.round(gains * 250) == k).sum()) for k in (1, 2, 3)}
hist["+4 or more"] = int((np.round(gains * 250) >= 4).sum())
best = np.argsort(-gains)[:20]
rows = []
with torch.no_grad():
    for k in best:
        t = int(np.searchsorted(offsets, k, side="right") - 1)
        flat = params[names[t]].data.view(-1)
        i = int(k - offsets[t])
        old = flat[i].item()
        flat[i] = 0.0
        rest = accuracy(model, test[250:])
        flat[i] = old
        rows.append({"tensor": names[t], "index": i, "probe_gain": float(gains[k]), "acc_rest": rest})
base_rest = accuracy(model, test[250:])
res["improvements"] = {"count": int((gains > 0.0005).sum()), "by_size": hist, "base_rest": base_rest,
                       "top20": rows, "top20_better_on_rest": sum(r["acc_rest"] > base_rest for r in rows),
                       "top20_worse_on_rest": sum(r["acc_rest"] < base_rest for r in rows)}
json.dump(res, open("results.json", "w"), indent=1)
print(json.dumps(res["improvements"], indent=1)[:2500])
