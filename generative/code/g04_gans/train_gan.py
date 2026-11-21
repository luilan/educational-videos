"""Episode 4 experiments: a GAN (a generator and a discriminator trained against each other) on Shapes.

    python3 train_gan.py dcgan         # the classic recipe (DCGAN: batch norm in both networks)
    python3 train_gan.py stable        # no batch norm in D, plus an R1 gradient penalty on real pictures
    python3 train_gan.py collapse      # "stable" with a much too fast generator (10× learning rate, 5 G steps per D step)
    → results_<run>.json and out/<run>_*.png

The generator turns 64 random numbers into a 32×32 picture. The discriminator outputs one number: how real a picture
looks. Samples come from an exponential moving average (EMA) of the generator's weights (decay 0.999), a standard
trick that smooths out the oscillations of GAN training ("raw" = the generator itself). After every epoch we draw 500
samples from a fixed set of random inputs and record:
  - the share the judge (judge.py, episode 3) recognizes;
  - how many of the 135 (shape, color, position) combinations appear among the recognized samples (coverage);
  - the distance to the nearest training picture.
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from judge import load_judge, recognizable, to_tensor
from shapes import make_dataset, save_grid

RUNS = {"dcgan": {"d_bn": True, "r1": 0.0, "g_lr": 2e-4, "d_lr": 2e-4, "betas": (0.5, 0.999), "g_steps": 1},
        "stable": {"d_bn": False, "r1": 1.0, "g_lr": 2e-4, "d_lr": 2e-4, "betas": (0.0, 0.99), "g_steps": 1},
        "collapse": {"d_bn": False, "r1": 1.0, "g_lr": 2e-3, "d_lr": 2e-4, "betas": (0.0, 0.99), "g_steps": 5}}
run = sys.argv[1] if len(sys.argv) > 1 else "stable"
cfg = RUNS[run]
torch.manual_seed(0)
torch.set_num_threads(min(8, os.cpu_count()))
OUT = "out"
os.makedirs(OUT, exist_ok=True)
Z, EPOCHS, BATCH = 64, int(os.environ.get("EPOCHS", 30)), 64


def to_images(x):
    return (x.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)


def up(c_in, c_out):                                         # double the size, then a 3×3 convolution
    return [nn.Upsample(scale_factor=2), nn.Conv2d(c_in, c_out, 3, 1, 1), nn.BatchNorm2d(c_out), nn.ReLU()]


generator = nn.Sequential(                                   # 64 random numbers → 3×32×32
    nn.Linear(Z, 256 * 4 * 4), nn.BatchNorm1d(256 * 4 * 4), nn.ReLU(), nn.Unflatten(1, (256, 4, 4)),
    *up(256, 128), *up(128, 64), *up(64, 32),                                  # 8×8, 16×16, 32×32
    nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh())                                      # pixels in [−1, 1]
ema = copy.deepcopy(generator)                               # a slowly moving average of the generator's weights
bn = (lambda c: nn.BatchNorm2d(c)) if cfg["d_bn"] else (lambda c: nn.Identity())
discriminator = nn.Sequential(                               # 3×32×32 → one number (a logit: real vs fake)
    nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
    nn.Conv2d(64, 128, 4, 2, 1), bn(128), nn.LeakyReLU(0.2),
    nn.Conv2d(128, 256, 4, 2, 1), bn(256), nn.LeakyReLU(0.2),
    nn.Flatten(), nn.Linear(256 * 4 * 4, 1))
opt_g = torch.optim.Adam(generator.parameters(), lr=cfg["g_lr"], betas=cfg["betas"])
opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg["d_lr"], betas=cfg["betas"])
bce = nn.BCEWithLogitsLoss()


def d_loss_fn(real, fake):
    real = real.requires_grad_(cfg["r1"] > 0)
    real_score, fake_score = discriminator(real), discriminator(fake)
    loss = bce(real_score, torch.ones_like(real_score)) + bce(fake_score, torch.zeros_like(fake_score))
    if cfg["r1"] > 0:                                        # R1: keep D's slope small around real pictures
        grad, = torch.autograd.grad(real_score.sum(), real, create_graph=True)
        loss = loss + cfg["r1"] / 2 * grad.pow(2).sum([1, 2, 3]).mean()
    return loss


def g_loss_fn(fake_score):                                   # "non-saturating": make D say real
    return bce(fake_score, torch.ones_like(fake_score))


@torch.no_grad()
def update_ema(decay=0.999):
    for pe, p in zip(ema.parameters(), generator.parameters()):
        pe.lerp_(p, 1 - decay)
    for be, b in zip(ema.buffers(), generator.buffers()):     # batch-norm statistics: average them too
        if be.dtype.is_floating_point:
            be.lerp_(b, 1 - decay)
        else:
            be.copy_(b)


images, _, _ = make_dataset(20_000, seed=0)
x_all = to_tensor(images)                                    # [0, 1] for the judge; the GAN works in [−1, 1]
judge = load_judge()
z_fixed = torch.randn(500, Z, generator=torch.Generator().manual_seed(1))


def nearest(x):
    d = torch.cdist(x.flatten(1), x_all.flatten(1)) ** 2 / x[0].numel()
    return d.min(1).values.mean().item()


@torch.no_grad()
def evaluate(net):
    net.eval()
    s = (net(z_fixed) + 1) / 2
    net.train()
    share, labels = recognizable(judge, s)
    probs = [l.softmax(-1) for l in judge(s)]
    ok = torch.stack([p.max(-1).values >= 0.9 for p in probs[:3]] + [probs[3][:, 1] >= 0.5], 1).all(1)
    combos = {tuple(l) for l, k in zip(labels.tolist(), ok.tolist()) if k}
    return s, share, len(combos), nearest(s[:200])


history = []
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(len(x_all))
    d_sum = g_sum = n = 0
    for i in range(0, len(x_all) - BATCH + 1, BATCH):
        real = x_all[perm[i:i + BATCH]] * 2 - 1
        fake = generator(torch.randn(BATCH, Z))
        # discriminator: real → 1, fake → 0
        d_loss = d_loss_fn(real, fake.detach())
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        # generator: make the discriminator say 1 for fakes
        for _ in range(cfg["g_steps"]):
            fake = generator(torch.randn(BATCH, Z))
            g_loss = g_loss_fn(discriminator(fake))
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            update_ema()
        d_sum, g_sum, n = d_sum + d_loss.item(), g_sum + g_loss.item(), n + 1
    raw, raw_share, raw_combos, _ = evaluate(generator)
    samples, share, combos, near = evaluate(ema)
    history.append({"epoch": epoch, "d_loss": d_sum / n, "g_loss": g_sum / n, "recognizable": share,
                    "combinations": combos, "nearest_train_mse": near, "raw_recognizable": raw_share,
                    "raw_combinations": raw_combos, "seconds": time.time() - t0})
    print(f"{run} epoch {epoch:2d}: D {d_sum / n:.3f}  G {g_sum / n:.3f}  recognizable {share:.1%} (raw {raw_share:.1%})  "
          f"combinations {combos}/135 (raw {raw_combos})  nearest {near:.4f}  {time.time() - t0:.0f}s", flush=True)
    if epoch in (1, 3, 10, 20, EPOCHS):
        save_grid(to_images(samples[:16]), f"{OUT}/{run}_epoch{epoch:02d}.png")
        save_grid(to_images(raw[:16]), f"{OUT}/{run}_raw_epoch{epoch:02d}.png")

save_grid(to_images(samples[:32]), f"{OUT}/{run}_final.png")
params = {"generator": sum(p.numel() for p in generator.parameters()),
          "discriminator": sum(p.numel() for p in discriminator.parameters())}
with open(f"results_{run}.json", "w") as f:
    json.dump({"run": run, "config": cfg, "epochs": EPOCHS, "z": Z, "parameters": params,
               "real_recognizable": recognizable(judge, x_all[:2000])[0], "history": history}, f, indent=1)
