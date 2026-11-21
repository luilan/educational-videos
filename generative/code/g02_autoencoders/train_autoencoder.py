"""Episode 2 experiments: an autoencoder squeezes a picture into a few numbers and rebuilds it.

    python3 train_autoencoder.py          # ~10 minutes on an 8-core CPU; writes results.json and out/*.png

Trains the same small convolutional autoencoder with codes of 2, 4, 8, 16 and 32 numbers on 20,000 Shapes pictures,
then measures on 2,000 new pictures:
  - reconstruction error (mean squared error) and the share of pixels that come back clearly wrong;
  - what the decoder makes from random codes (drawn with each code number's own mean and spread);
  - what it makes halfway between the codes of two real pictures;
  - where the 2,000 test pictures land in the 2-number code space (results.json: codes_2d, labels = shape, color,
    position indices from shapes.py).
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from shapes import make_dataset, save_grid

torch.manual_seed(0)
torch.set_num_threads(min(8, os.cpu_count()))
OUT = "out"
os.makedirs(OUT, exist_ok=True)
SIZES = [2, 4, 8, 16, 32]
EPOCHS, BATCH = 15, 128


def to_tensor(images):
    return torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0


def to_images(x):
    return (x.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)


class Autoencoder(nn.Module):
    def __init__(self, code_size):
        super().__init__()
        self.encoder = nn.Sequential(                                   # 3×32×32 → code
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),      # 32×16×16
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),     # 64×8×8
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),   # 128×4×4
            nn.Flatten(), nn.Linear(128 * 4 * 4, code_size))
        self.decoder = nn.Sequential(                                   # code → 3×32×32
            nn.Linear(code_size, 128 * 4 * 4), nn.LeakyReLU(0.1), nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


def nearest(x):
    """Mean over x of the squared error to the closest training picture: how far from any real picture it is."""
    d = torch.cdist(x.flatten(1), xtr.flatten(1)) ** 2 / x[0].numel()
    return d.min(1).values.mean().item()


train_imgs, _, _ = make_dataset(20_000, seed=0)
test_imgs, test_caps, test_labels = make_dataset(2_000, seed=1)
xtr, xte = to_tensor(train_imgs), to_tensor(test_imgs)
res = {"nearest_note": "squared error per number to the closest of the 20,000 training pictures",
       "train_pictures": len(xtr), "test_pictures": len(xte), "numbers_per_picture": 3072, "epochs": EPOCHS, "runs": {}}
show = xte[:8]
save_grid(to_images(show), f"{OUT}/originals.png")
res["test_captions"] = test_caps[:8]

for size in SIZES:
    model = Autoencoder(size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0, curve = time.time(), []
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(xtr))
        total = 0.0
        for i in range(0, len(xtr), BATCH):
            x = xtr[perm[i:i + BATCH]]
            loss = ((model(x) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        curve.append(total / len(xtr))
    model.eval()                                   # batch norm: use the running averages from training
    with torch.no_grad():
        codes = model.encoder(xte)
        recon = model.decoder(codes)
        mse = ((recon - xte) ** 2).mean().item()
        wrong = ((recon - xte).abs() > 0.25).any(1).float().mean().item()      # pixel clearly off in any channel
        mu, sd = codes.mean(0), codes.std(0)
        g = torch.Generator().manual_seed(1)
        rand = model.decoder(mu + sd * torch.randn(8, size, generator=g))
        mid = model.decoder((codes[:8] + codes[8:16]) / 2)
        near_rand, near_mid = nearest(rand), nearest(mid)
    save_grid(to_images(recon[:8]), f"{OUT}/recon_{size}.png")
    save_grid(to_images(rand), f"{OUT}/random_{size}.png")
    save_grid(to_images(torch.cat([xte[:8], mid, xte[8:16]])), f"{OUT}/halfway_{size}.png")
    params = sum(p.numel() for p in model.parameters())
    res["runs"][size] = {"test_mse": mse, "wrong_pixel_share": wrong, "train_curve": curve,
                         "seconds": time.time() - t0, "parameters": params,
                         "compression": 3072 / size,
                         "random_nearest_train_mse": near_rand, "halfway_nearest_train_mse": near_mid, "first_codes": codes[:3].tolist()}
    print(f"code {size:2d}: test MSE {mse:.5f}, wrong pixels {wrong:.2%}, random→nearest {near_rand:.4f}, "
          f"halfway→nearest {near_mid:.4f}, {time.time() - t0:.0f}s, {params:,} params")
    if size == 2:                                  # a 2-number code can be drawn as a point on a plane
        res["codes_2d"] = {"codes": codes.round(decimals=3).tolist(), "labels": test_labels.tolist()}
    torch.save(model.state_dict(), f"{OUT}/autoencoder_{size}.pt")

with open("results.json", "w") as f:
    json.dump(res, f, indent=1)
