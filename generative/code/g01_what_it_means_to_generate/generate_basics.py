"""Episode 1 experiments: what it means to generate, and why counting stops working.

    python3 generate_basics.py          # writes results.json and the PNGs in out/

1. A 2D toy dataset (a spiral). A "counting" generator: split the plane into 30×30 cells, count the points in each
   cell, then sample a cell in proportion to its count and a point inside it. In 2D this works.
2. The same idea for 32×32 color images: how many cells would there be?
3. A generator that learns each pixel separately (the color distribution at every pixel), then samples pixels
   independently. Its samples have the right colors in the right places on average, but the pixels don't fit together.
4. The average image, and the copy machine (return a random training image), for comparison.
"""
import json
import math
import os

import numpy as np

from shapes import COLORS, BACKGROUND, make_dataset, save_grid

OUT = "out"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
res = {}

# ---------------------------------------------------------------- 1. a 2D toy and a counting generator
n = 10_000
t = np.sqrt(rng.uniform(0, 1, n)) * 3 * math.pi                    # spiral parameter
spiral = np.stack([t * np.cos(t), t * np.sin(t)], 1) / (3 * math.pi) + rng.normal(0, 0.03, (n, 2))
BINS, LO, HI = 30, -1.15, 1.15
edges = np.linspace(LO, HI, BINS + 1)
counts, _, _ = np.histogram2d(spiral[:, 0], spiral[:, 1], bins=[edges, edges])
p = counts.ravel() / counts.sum()
cells = rng.choice(BINS * BINS, size=2000, p=p)
w = (HI - LO) / BINS
samples = np.stack([LO + (cells // BINS + rng.uniform(0, 1, 2000)) * w,
                    LO + (cells % BINS + rng.uniform(0, 1, 2000)) * w], 1)
res["toy"] = {"points": n, "cells": BINS * BINS, "occupied_cells": int((counts > 0).sum()),
              "points_per_cell": n / (BINS * BINS),
              "counts": counts.astype(int).tolist(),
              "data": np.round(spiral[:600], 3).tolist(), "samples": np.round(samples[:600], 3).tolist()}

# ---------------------------------------------------------------- 2. the same counting idea for images
dims = 32 * 32 * 3
res["count"] = {"numbers_per_image": dims,
                "cells_2_levels_log10": dims * math.log10(2),       # every number just "dark" or "bright"
                "cells_256_levels_log10": dims * math.log10(256),
                "cells_by_dims": {d: d * math.log10(BINS) for d in (1, 2, 3, 10, 100)}}

# ---------------------------------------------------------------- 3. pixels learned one by one
N = 20_000
images, captions, labels = make_dataset(N, seed=0)
palette = np.array([BACKGROUND] + list(COLORS.values()), np.uint8)              # 6 possible colors
idx = (images[:, :, :, None, :] == palette[None, None, None]).all(-1).argmax(-1)  # (N, 32, 32) color index
freq = np.stack([(idx == k).mean(0) for k in range(len(palette))], -1)          # (32, 32, 6) per-pixel distribution
cum = freq.cumsum(-1)
u = rng.uniform(0, 1, (16, 32, 32, 1))
sample_idx = (u > cum[None]).sum(-1).clip(0, len(palette) - 1)
pixel_samples = palette[sample_idx]
real = images[:16]


def colors_used(im):
    return len({tuple(v) for v in im.reshape(-1, 3)})


def shape_pixels(im):
    return int((im != np.array(BACKGROUND, np.uint8)).any(-1).sum())


def largest_blob(im):
    """Pixels in the biggest 4-connected group of non-background pixels."""
    fg = (im != np.array(BACKGROUND, np.uint8)).any(-1)
    seen, best = np.zeros_like(fg), 0
    for y0, x0 in zip(*np.nonzero(fg)):
        if seen[y0, x0]:
            continue
        stack, size = [(y0, x0)], 0
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            size += 1
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < 32 and 0 <= xx < 32 and fg[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        best = max(best, size)
    return best


res["pixels"] = {
    "dataset_images": N,
    "real_colors_mean": float(np.mean([colors_used(im) for im in real])),
    "sample_colors_mean": float(np.mean([colors_used(im) for im in pixel_samples])),
    "real_shape_pixels_mean": float(np.mean([shape_pixels(im) for im in images[:2000]])),
    "sample_shape_pixels_mean": float(np.mean([shape_pixels(im) for im in pixel_samples])),
    "real_largest_blob_mean": float(np.mean([largest_blob(im) for im in real])),
    "sample_largest_blob_mean": float(np.mean([largest_blob(im) for im in pixel_samples])),
    "busiest_pixel_shape_prob": float(1 - freq[:, :, 0].min()),
}
save_grid(real[:8], f"{OUT}/real.png", cols=8)
res["pixels"]["real_captions"] = captions[:8]
shape_prob = 1 - freq[:, :, 0]                                     # chance that each pixel is part of a shape
heat = (shape_prob / shape_prob.max() * 255).round().astype(np.uint8)
save_grid(np.repeat(heat[None, :, :, None], 3, -1), f"{OUT}/shape_probability.png", cols=1, scale=8)
save_grid(pixel_samples[:8], f"{OUT}/pixel_samples.png", cols=8)

# ---------------------------------------------------------------- 4. the average image and the copy machine
mean_img = images.mean(0).round().astype(np.uint8)
save_grid(mean_img[None], f"{OUT}/average.png", cols=1, scale=8)
bright = np.clip((images.mean(0) - BACKGROUND) * 4 + BACKGROUND, 0, 255).round().astype(np.uint8)   # contrast × 4
save_grid(bright[None], f"{OUT}/average_x4.png", cols=1, scale=8)
res["copy"] = {"distinct_captions": 135, "first_caption_examples": captions[:3]}
save_grid(images[rng.choice(N, 8)], f"{OUT}/copies.png", cols=8)

with open("results.json", "w") as f:
    json.dump(res, f, indent=1)
for k in ("toy", "count", "pixels"):
    print(k, {a: b for a, b in res[k].items() if a not in ("data", "samples")})
