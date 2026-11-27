"""Draw one weight table of model.pt as a picture: out/weights.png (blue = negative, orange = positive)."""
import os

import numpy as np
import torch
from PIL import Image

os.makedirs("out", exist_ok=True)
w = torch.load("model.pt")["blocks.0.mlp.0.weight"].numpy().T          # 64 × 256
x = np.clip(w / np.abs(w).max() * 2.5, -1, 1)[..., None]
neg, mid, pos = np.array([40, 110, 200]), np.array([20, 22, 28]), np.array([245, 140, 50])
rgb = np.where(x < 0, mid + (neg - mid) * -x, mid + (pos - mid) * x).astype(np.uint8)
Image.fromarray(rgb).resize((256 * 4, 64 * 4), Image.NEAREST).save("out/weights.png")
print(w.shape)
