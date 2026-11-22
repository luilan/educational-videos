"""Episodes 10–11: a diffusion model told what to draw (shape, color, position), with label dropout for guidance.

    python3 train_cond.py          # ~3.5 hours on a CPU (20,000 steps) → out/cond_ema.pt, results_train.json
    python3 evaluate_cond.py       # episode 10: does it draw what it is told? → results_cond.json
    (episode 11's guidance sweep is ../g11_guidance/guidance.py, using the same weights)

The U-Net is episode 6's, plus three label embeddings (shape, color, position) added to the time embedding.
Label dropout: 10% of training pictures get no label at all (so the same network is also an unconditional model,
which guidance needs); otherwise each of the three labels is dropped on its own 10% of the time, so the model can also
be told only part of a description ("something red").
Eight shape-color-position combinations are left out of training on purpose, to test whether the model can draw a
combination it has never seen.
"""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import diffusion as D
from shapes import COLOR_NAMES, POSITIONS, SHAPES, make_dataset

NULL = 9                                    # index meaning "not said" for each of the three labels
HELD_OUT = [(2, 4, 0), (0, 3, 8), (1, 0, 4), (2, 1, 6), (0, 2, 2), (1, 4, 7), (2, 3, 5), (0, 0, 3)]


class CondUNet(D.UNet):
    """Episode 6's U-Net with three label embeddings added to the time embedding."""

    def __init__(self, ch=(32, 64, 128), t_dim=128):
        super().__init__(ch=ch, t_dim=t_dim)
        self.labels = nn.ModuleList([nn.Embedding(NULL + 1, t_dim) for _ in range(3)])

    def forward(self, x, t, y=None):
        if y is None:
            y = torch.full((len(x), 3), NULL, dtype=torch.long)
        temb = self.t_mlp(D.timestep_embedding(t, self.t_dim)) + sum(e(y[:, k]) for k, e in enumerate(self.labels))
        h1 = self.down1(self.inp(x), temb)
        h2 = self.down2(self.pool1(h1), temb)
        h = self.mid2(self.mid1(self.pool2(h2), temb), temb)
        h = self.up2(torch.cat([self.upc2(h), h2], 1), temb)
        h = self.up1(torch.cat([self.upc1(h), h1], 1), temb)
        return self.out(h)


def drop_labels(y, gen=None):
    """10%: no label at all; otherwise each label dropped on its own with probability 10%."""
    y = y.clone()
    all_ = torch.rand(len(y), generator=gen) < 0.1
    each = torch.rand(y.shape, generator=gen) < 0.1
    y[each | all_[:, None]] = NULL
    return y


def held_out_mask(labels):
    return np.array([tuple(l) in set(HELD_OUT) for l in labels.tolist()])


def describe(c):
    s, col, p = c
    return f"{COLOR_NAMES[col]} {SHAPES[s]}, {POSITIONS[p]}"


if __name__ == "__main__":
    from judge import to_tensor
    from shapes import save_grid

    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("THREADS", min(8, os.cpu_count()))))
    os.makedirs("out", exist_ok=True)
    STEPS, BATCH, LR = int(os.environ.get("STEPS", 20_000)), 128, 2e-4
    images, _, labels = make_dataset(20_000, seed=0)
    keep = ~held_out_mask(labels)
    x_all, y_all = to_tensor(images[keep]) * 2 - 1, torch.tensor(labels[keep])
    print(f"{len(x_all):,} training pictures ({(~keep).sum()} with held-out combinations removed)", flush=True)
    model = CondUNet()
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    params = sum(p.numel() for p in model.parameters())
    print(f"{params:,} parameters", flush=True)
    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(x_all), (BATCH,))
        x0, y = x_all[idx], drop_labels(y_all[idx])
        t = torch.randint(1, D.T + 1, (BATCH,))
        eps = torch.randn_like(x0)
        loss = F.mse_loss(model(D.noise(x0, t, eps), t, y), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.lerp_(p, 1 - 0.999)
        log.append(loss.item())
        if step % 2000 == 0 or step == STEPS:
            print(f"step {step:5d}: loss {np.mean(log[-200:]):.4f}  {(time.time() - t0) / 60:.0f} min", flush=True)
            torch.save(ema.state_dict(), "out/cond_ema.pt")
    with open("results_train.json", "w") as f:
        json.dump({"parameters": params, "steps": STEPS, "batch": BATCH, "lr": LR,
                   "training_pictures": int(len(x_all)), "held_out": HELD_OUT,
                   "loss_curve": [float(np.mean(log[i:i + 200])) for i in range(0, STEPS, 200)],
                   "minutes": (time.time() - t0) / 60}, f, indent=1)
