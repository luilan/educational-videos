"""Episode 3 experiments: a variational autoencoder (VAE) next to a plain autoencoder.

    python3 train_vae.py          # ~10 minutes on an 8-core CPU; writes results.json and out/*.png

Both use episode 2's network with an 8-number code and train for 40 epochs on 20,000 Shapes pictures. The VAE's
encoder outputs a mean and a spread for every code number; training samples a code from them (the
reparameterization trick) and adds a KL penalty that pulls every picture's code distribution toward a standard
normal. Then, on 2,000 new pictures and on generated samples:
  - reconstruction error;
  - samples from random codes: distance to the nearest training picture, and the share the judge (judge.py) can read;
  - halfway codes;
  - a VAE with a 2-number code: where test pictures land, and a 10×10 map of decoded codes.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from judge import breakdown, load_judge, recognizable, to_tensor
from shapes import make_dataset, save_grid

torch.manual_seed(0)
torch.set_num_threads(min(8, os.cpu_count()))
OUT = "out"
os.makedirs(OUT, exist_ok=True)
EPOCHS, BATCH = 40, 128


def to_images(x):
    return (x.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)


def encoder_body():
    return nn.Sequential(
        nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
        nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
        nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.Flatten())


def decoder(code_size):
    return nn.Sequential(
        nn.Linear(code_size, 128 * 4 * 4), nn.LeakyReLU(0.1), nn.Unflatten(1, (128, 4, 4)),
        nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
        nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
        nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())


class Autoencoder(nn.Module):                      # episode 2
    def __init__(self, code_size):
        super().__init__()
        self.body, self.to_code, self.decoder = encoder_body(), nn.Linear(2048, code_size), decoder(code_size)

    def encode(self, x):
        return self.to_code(self.body(x))

    def loss(self, x):
        x_hat = self.decoder(self.encode(x))
        return ((x_hat - x) ** 2).mean(), {}


class VAE(nn.Module):
    def __init__(self, code_size):
        super().__init__()
        self.body, self.decoder = encoder_body(), decoder(code_size)
        self.to_mean, self.to_logvar = nn.Linear(2048, code_size), nn.Linear(2048, code_size)

    def encode(self, x):                           # the code's mean: used for reconstructions and plots
        return self.to_mean(self.body(x))

    def loss(self, x):
        h = self.body(x)
        mean, logvar = self.to_mean(h), self.to_logvar(h)
        code = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)            # reparameterization trick
        x_hat = self.decoder(code)
        recon = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum") / len(x)
        kl = 0.5 * (mean ** 2 + logvar.exp() - 1 - logvar).sum() / len(x)        # KL(q(code|x) ‖ N(0, I))
        return recon + kl, {"recon": recon.item(), "kl": kl.item()}


def train(model):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0, log = time.time(), []
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xtr))
        sums = {}
        for i in range(0, len(xtr), BATCH):
            loss, parts = model.loss(xtr[perm[i:i + BATCH]])
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0) + v * BATCH
        log.append({k: v / len(xtr) for k, v in sums.items()})
    return model.eval(), time.time() - t0, log


def nearest(x):
    d = torch.cdist(x.flatten(1), xtr.flatten(1)) ** 2 / x[0].numel()
    return d.min(1).values.mean().item()


train_imgs, _, _ = make_dataset(20_000, seed=0)
test_imgs, test_caps, test_labels = make_dataset(2_000, seed=1)
xtr, xte = to_tensor(train_imgs), to_tensor(test_imgs)
judge = load_judge()
real_share, _ = recognizable(judge, xte)
res = {"real_recognizable": real_share, "real_breakdown": breakdown(judge, xte), "test_captions": test_caps[:16], "models": {}}
print(f"judge: real test pictures {real_share:.1%} recognizable")
save_grid(to_images(xte[:8]), f"{OUT}/originals.png")

for name, model in [("autoencoder", Autoencoder(8)), ("vae", VAE(8))]:
    model, secs, log = train(model)
    with torch.no_grad():
        codes = model.encode(xte)
        recon = model.decoder(codes)
        g = torch.Generator().manual_seed(1)
        if name == "vae":                           # a VAE's codes are meant to follow N(0, I)
            z = torch.randn(500, 8, generator=g)
        else:                                       # episode 2's recipe: match each number's mean and spread
            z = codes.mean(0) + codes.std(0) * torch.randn(500, 8, generator=g)
        samples = model.decoder(z)
        mid = model.decoder((codes[:8] + codes[8:16]) / 2)
        mid_all = model.decoder((codes[:500] + codes[500:1000]) / 2)
    share, _ = recognizable(judge, samples)
    parts = breakdown(judge, samples)
    mid_share, _ = recognizable(judge, mid_all)
    res["models"][name] = {
        "seconds": secs, "log": log, "test_mse": ((recon - xte) ** 2).mean().item(),
        "wrong_pixel_share": ((recon - xte).abs() > 0.25).any(1).float().mean().item(),
        "random_nearest_train_mse": nearest(samples[:200]), "random_recognizable": share, "random_breakdown": parts,
        "halfway_recognizable": mid_share, "code_mean": codes.mean(0).tolist(), "code_std": codes.std(0).tolist()}
    save_grid(to_images(recon[:8]), f"{OUT}/recon_{name}.png")
    save_grid(to_images(samples[:16]), f"{OUT}/random_{name}.png")
    save_grid(to_images(torch.cat([xte[:8], mid, xte[8:16]])), f"{OUT}/halfway_{name}.png")
    if name == "vae":                               # a walk from picture 0 to picture 8 in 7 steps
        t = torch.linspace(0, 1, 7)[:, None]
        with torch.no_grad():
            walk = model.decoder((1 - t) * codes[0] + t * codes[8])
        save_grid(to_images(walk), f"{OUT}/walk_vae.png", cols=7)
    m = res["models"][name]
    print(f"{name}: {secs:.0f}s, test MSE {m['test_mse']:.5f}, wrong {m['wrong_pixel_share']:.2%}, random→nearest "
          f"{m['random_nearest_train_mse']:.4f}, random recognizable {share:.1%} {parts}, halfway recognizable {mid_share:.1%}")

# ---------------------------------------------------------------- a VAE with a 2-number code: the whole map
vae2, secs, log = train(VAE(2))
with torch.no_grad():
    codes2 = vae2.encode(xte)
    grid = torch.linspace(-2, 2, 10)
    z = torch.stack([torch.stack([gx, -gy]) for gy in grid for gx in grid])   # rows top→bottom, columns left→right
    tiles = vae2.decoder(z)
    rand2 = vae2.decoder(torch.randn(500, 2, generator=torch.Generator().manual_seed(1)))
    mse2 = ((vae2.decoder(codes2) - xte) ** 2).mean().item()
share2, _ = recognizable(judge, rand2)
parts2 = breakdown(judge, rand2)
save_grid(to_images(rand2[:16]), f"{OUT}/random_vae2.png")
save_grid(to_images(tiles), f"{OUT}/map_vae2.png", cols=10, scale=2)
res["vae2"] = {"seconds": secs, "log": log, "codes": codes2.round(decimals=3).tolist(), "labels": test_labels.tolist(),
               "test_mse": mse2, "random_recognizable": share2, "random_breakdown": parts2}
print(f"vae2: {secs:.0f}s, test MSE {res['vae2']['test_mse']:.5f}, random recognizable {share2:.1%} {parts2}")

with open("results.json", "w") as f:
    json.dump(res, f, indent=1)
