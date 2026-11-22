"""Episode 13: pictures for the video (after train_autoencoder.py and TEXT=words latent_diffusion.py).

    python3 figures.py      # ~2 minutes → out/latents.png, figures.json

clip_real_latents (figures.json): 1,000 test pictures encoded, the latents clipped to ±c, decoded and read by the
               judge: share still exactly right.
latents.png    four test pictures (top row) and the four channels of each one's 8×8 latent (rows 2-5, scaled up,
               blue negative, red positive).
"""
import json

import numpy as np
import torch
from PIL import Image

import latent_diffusion as L
from judge import to_tensor
from shapes import make_dataset, save_grid

torch.manual_seed(0)
torch.set_num_threads(4)
ae, _, _ = L.load_models("words")
out = {}


def heat(m, lim):
    """8×8 array → 32×32 RGB tile: blue for negative, red for positive, dark for zero (like the pictures' background)."""
    v = np.clip(m / lim, -1, 1)
    bg = np.array([22, 24, 32], float)
    rgb = np.where(v[..., None] >= 0, bg + v[..., None] * (np.array([230, 60, 50]) - bg),
                   bg - v[..., None] * (np.array([60, 110, 240]) - bg))
    return np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((32, 32), Image.NEAREST))


# 1. latents of four pictures
img, _, lab = make_dataset(4, seed=3)
with torch.no_grad():
    z = ae.encode(to_tensor(img))[0].numpy()
lim = np.abs(z).max()
tiles = list(img) + [heat(z[i, ch], lim) for ch in range(4) for i in range(4)]
save_grid(np.stack(tiles), "out/latents.png", cols=4)
out["latents"] = {"pictures": lab.tolist(), "max_abs": float(lim)}

# 2. what clipping does to real latents: encode test pictures, clip, decode, read
from judge import load_judge
judge = load_judge()
img, _, lab = make_dataset(1000, seed=7)
lab = torch.tensor(lab)
out["clip_real_latents"] = {}
with torch.no_grad():
    z = ae.encode(to_tensor(img))[0]
    out["clip_real_latents"]["max_abs"] = z.abs().max().item()
    out["clip_real_latents"]["share_beyond_4"] = (z.abs() > 4).float().mean().item()
    for c in (None, 8, 6, 4):
        p = [l.softmax(-1) for l in judge(ae.decode(z if c is None else z.clamp(-c, c)))]
        ok = torch.stack([q.max(-1).values >= 0.9 for q in p[:3]] + [p[3][:, 1] >= 0.5], 1).all(1)
        right = (torch.stack([q.argmax(-1) for q in p[:3]], 1) == lab).all(1) & ok
        out["clip_real_latents"][f"clip {c}" if c else "no clip"] = right.float().mean().item()

print(out, flush=True)
with open("figures.json", "w") as f:
    json.dump(out, f, indent=1)
