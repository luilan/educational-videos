"""Episode 8 experiments: what do the U-Net's skip connections and its time input buy?

    python3 train_ablation.py noskip     # the same U-Net without skip connections
    python3 train_ablation.py notime     # the same U-Net that is never told the step t
    → results_<variant>.json and out/<variant>_*.png  (~2.5 hours each on an 8-core CPU; THREADS=4 to share a CPU)

Everything else matches episode 6's run exactly (same data, seed, 20,000 steps, batch 128, Adam 0.0002, EMA 0.999,
DDIM-50 samples scored by episode 3's judge), so the full model's numbers come from
../g06_learning_to_denoise/results_shapes.json.
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import diffusion as D
from judge import load_judge, to_tensor
from shapes import make_dataset, save_grid

variant = sys.argv[1]
kw = {"noskip": {"use_skips": False}, "notime": {"use_time": False}}[variant]
torch.manual_seed(0)
torch.set_num_threads(int(os.environ.get("THREADS", min(8, os.cpu_count()))))
OUT = "out"
os.makedirs(OUT, exist_ok=True)
STEPS, BATCH, LR = int(os.environ.get("STEPS", 20_000)), 128, 2e-4
CHECK = [2000, 5000, 10_000, STEPS]
images, _, _ = make_dataset(20_000, seed=0)
x_all = to_tensor(images) * 2 - 1
judge = load_judge()
model = D.UNet(ch=(32, 64, 128), **kw)
ema = copy.deepcopy(model).eval()
opt = torch.optim.Adam(model.parameters(), lr=LR)
params = sum(p.numel() for p in model.parameters())
print(f"{variant}: {params:,} parameters", flush=True)


def to_images(x):
    return ((x.clamp(-1, 1) + 1) / 2 * 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8)


def evaluate(net, n=500):
    s = (D.sample_ddim(net, n, steps=50, generator=torch.Generator().manual_seed(1)) + 1) / 2
    probs = [l.softmax(-1) for l in judge(s)]
    ok = torch.stack([p.max(-1).values >= 0.9 for p in probs[:3]] + [probs[3][:, 1] >= 0.5], 1).all(1)
    labels = torch.stack([p.argmax(-1) for p in probs[:3]], 1)
    combos = {tuple(l) for l, k in zip(labels.tolist(), ok.tolist()) if k}
    d = torch.cdist(s[:200].flatten(1), ((x_all + 1) / 2).flatten(1)) ** 2 / s[0].numel()
    return s, ok.float().mean().item(), len(combos), d.min(1).values.mean().item()


# the same demo as episode 6: one picture noised to five steps
demo_x0 = x_all[:1].repeat(5, 1, 1, 1)
demo_t = torch.tensor([50, 200, 400, 600, 900])
demo_eps = torch.randn(demo_x0.shape, generator=torch.Generator().manual_seed(3))
demo_xt = D.noise(demo_x0, demo_t, demo_eps)
# per-step error: how well is the noise predicted at each noise level? (fixed batch of 256 test pictures)
test_x0 = to_tensor(make_dataset(256, seed=99)[0]) * 2 - 1
LEVELS = [10, 50, 100, 200, 400, 600, 800, 950]

log, history = [], []
t0 = time.time()
for step in range(1, STEPS + 1):
    loss = D.training_loss(model, x_all[torch.randint(0, len(x_all), (BATCH,))])
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), model.parameters()):
            pe.lerp_(p, 1 - 0.999)
    log.append(loss.item())
    if step in CHECK:
        s, share, combos, near = evaluate(ema)
        with torch.no_grad():
            eps_hat = ema(demo_xt, demo_t)
            ab = D.ALPHA_BAR[demo_t - 1].view(-1, 1, 1, 1)
            x0_hat = ((demo_xt - (1 - ab).sqrt() * eps_hat) / ab.sqrt()).clamp(-1, 1)
            per_level = {}
            g = torch.Generator().manual_seed(4)
            for lv in LEVELS:
                eps = torch.randn(test_x0.shape, generator=g)
                tt = torch.full((len(test_x0),), lv)
                per_level[lv] = F.mse_loss(ema(D.noise(test_x0, tt, eps), tt), eps).item()
        history.append({"step": step, "loss_last_200": float(np.mean(log[-200:])), "recognizable": share,
                        "combinations": combos, "nearest_train_mse": near, "eps_mse_by_step": per_level,
                        "minutes": (time.time() - t0) / 60})
        save_grid(to_images(s[:16] * 2 - 1), f"{OUT}/{variant}_samples_step{step:05d}.png")
        save_grid(to_images(torch.cat([demo_xt, x0_hat])), f"{OUT}/{variant}_denoise_step{step:05d}.png", cols=5)
        torch.save(ema.state_dict(), f"{OUT}/{variant}_ema.pt")
        print(f"step {step:5d}: loss {history[-1]['loss_last_200']:.4f}  recognizable {share:.1%}  combinations "
              f"{combos}/135  nearest {near:.4f}  {history[-1]['minutes']:.0f} min", flush=True)

with open(f"results_{variant}.json", "w") as f:
    json.dump({"variant": variant, "parameters": params, "steps": STEPS,
               "loss_curve": [float(np.mean(log[i:i + 200])) for i in range(0, STEPS, 200)],
               "levels": LEVELS, "history": history}, f, indent=1)
