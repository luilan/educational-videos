"""Episode 13, part 1: the autoencoder in front of latent diffusion.

    python3 train_autoencoder.py     # about an hour on a CPU → out/autoencoder.pt, results_autoencoder.json, out/recon.png

32×32×3 pictures (3,072 numbers) ↔ 8×8×4 latents (256 numbers, 12× fewer), 4× smaller per side; Stable Diffusion's autoencoder
shrinks 8× per side (512×512×3 ↔ 64×64×4, 48× fewer numbers). Like Stable Diffusion's, it is a lightly
regularized VAE: mean squared error plus a very small KL term (weight 1e-6), so the latents stay smooth and well
scaled without blurring the pictures. With weight 1e-4 none of the reconstructions were recognizable (blurry, colorless
blobs; results_autoencoder_kl1e-4.json); L1 loss without KL was still blurry after 2,000 steps. Episode 10's eight held-out combinations are left out of training.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from judge import load_judge, to_tensor
from shapes import make_dataset, save_grid

HELD_OUT = [(2, 4, 0), (0, 3, 8), (1, 0, 4), (2, 1, 6), (0, 2, 2), (1, 4, 7), (2, 3, 5), (0, 0, 3)]
LATENT = 4


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(),                 # 16×16
            nn.Conv2d(64, 128, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(128, 128, 4, 2, 1), nn.SiLU(),               # 8×8
            nn.Conv2d(128, 128, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(128, 2 * LATENT, 1))                         # mean and log-variance
        self.dec = nn.Sequential(
            nn.Conv2d(LATENT, 128, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.SiLU(),      # 16×16
            nn.Conv2d(128, 64, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.SiLU(),        # 32×32
            nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(64, 3, 3, 1, 1))
        self.register_buffer("scale", torch.tensor(1.0))           # set after training: makes latents ~ unit variance

    def encode(self, x, sample=False):
        mean, logvar = self.enc(x * 2 - 1).chunk(2, 1)
        z = mean + (0.5 * logvar).exp() * torch.randn_like(mean) if sample else mean
        return z * self.scale, mean, logvar

    def decode(self, z):
        return ((self.dec(z / self.scale) + 1) / 2).clamp(0, 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("THREADS", min(8, os.cpu_count()))))
    STEPS, BATCH, LR, KL = int(os.environ.get("STEPS", 6000)), 128, 1e-3, float(os.environ.get("KL", 1e-6))
    L1 = os.environ.get("LOSS", "mse") == "l1"
    OUT = os.environ.get("OUT", "out")
    images, _, labels = make_dataset(20_000, seed=0)
    keep = np.array([tuple(l) not in set(HELD_OUT) for l in labels.tolist()])
    x_all = to_tensor(images[keep])
    ae = Autoencoder()
    opt = torch.optim.Adam(ae.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        x = x_all[torch.randint(0, len(x_all), (BATCH,))]
        _, mean, logvar = ae.encode(x, sample=True)
        z = mean + (0.5 * logvar).exp() * torch.randn_like(mean)
        rec = (ae.dec(z) + 1) / 2
        mse = F.mse_loss(rec, x)
        rec_loss = F.l1_loss(rec, x) if L1 else mse
        kl = 0.5 * (mean ** 2 + logvar.exp() - 1 - logvar).mean()
        loss = rec_loss + KL * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        log.append((mse.item(), kl.item()))
        if step % 1000 == 0:
            m = np.mean(log[-200:], 0)
            print(f"step {step}: mse {m[0]:.5f}  kl {m[1]:.3f}  {(time.time() - t0) / 60:.0f} min", flush=True)
    ae.eval()
    with torch.no_grad():
        z = torch.cat([ae.encode(x_all[i:i + 1000])[0] for i in range(0, 5000, 1000)])
        ae.scale.fill_(1 / z.std().item())
        test_img, _, test_lab = make_dataset(1000, seed=99)
        x_test = to_tensor(test_img)
        rec = ae.decode(ae.encode(x_test)[0])
        judge = load_judge()
        probs = [l.softmax(-1) for l in judge(rec)]
        ok = torch.stack([p.max(-1).values >= 0.9 for p in probs[:3]] + [probs[3][:, 1] >= 0.5], 1).all(1)
        same = torch.stack([p.argmax(-1) for p in probs[:3]], 1).eq(torch.tensor(test_lab)).all(1)
        probs_real = [l.softmax(-1) for l in judge(x_test)]
        ok_real = torch.stack([p.max(-1).values >= 0.9 for p in probs_real[:3]] + [probs_real[3][:, 1] >= 0.5], 1).all(1)
        save_grid(np.concatenate([test_img[:8], (rec[:8].permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)]),
                  f"{OUT}/recon.png")
        zs = ae.encode(x_test[:8])[0]
    torch.save(ae.state_dict(), f"{OUT}/autoencoder.pt")
    res = {"steps": STEPS, "kl_weight": KL, "parameters": {"encoder": sum(p.numel() for p in ae.enc.parameters()),
                                                           "decoder": sum(p.numel() for p in ae.dec.parameters())},
           "picture_numbers": 3 * 32 * 32, "latent_numbers": LATENT * 8 * 8, "scale": ae.scale.item(),
           "test_mse": F.mse_loss(rec, x_test).item(), "recognizable_reconstructions": ok.float().mean().item(),
           "same_reading_as_original": (ok & same).float().mean().item(), "recognizable_originals": ok_real.float().mean().item(),
           "latents_of_first_8": zs.numpy().round(3).tolist(),
           "curve": [[float(a), float(b)] for a, b in np.array(log).reshape(-1, 100, 2).mean(1)], "minutes": (time.time() - t0) / 60}
    with open(f"{OUT}/results_autoencoder.json" if OUT != "out" else "results_autoencoder.json", "w") as f:
        json.dump(res, f, indent=1)
    print({k: v for k, v in res.items() if k not in ("curve", "latents_of_first_8")})
