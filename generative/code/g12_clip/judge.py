"""A judge for generated Shapes pictures: a small classifier that reads shape, color and position, and says whether
the picture is one clean shape at all.

Besides the three readings it has a fourth head, "clean", trained on real pictures (half of them lightly blurred
and noised, so slightly soft pictures still count) against junk: two pictures averaged, two shapes in one picture, shapes in colors
outside the palette, heavy blur, and noise. A generated picture counts as **recognizable** when the judge is at least
90% sure of its shape, color and position, and thinks "clean" more likely than "junk".

Checked with `python3 judge.py` (2,000 new pictures): real 100% recognizable; slightly blurred (Gaussian σ = 0.5
pixel) 95%, σ = 0.7: 60%, σ = 1: 8%; every kind of junk 0–1%. So the share of recognizable samples is a simple quality
score for a generator.

    from judge import load_judge, recognizable
    judge = load_judge()                       # episode 3's weights; trains one (~3 minutes) if they are missing
    share, labels = recognizable(judge, x)     # x: float tensor (N, 3, 32, 32) in [0, 1]
"""
import os

import numpy as np
import torch
import torch.nn as nn

from shapes import COLOR_NAMES, COLORS, POSITIONS, SHAPES, draw, make_dataset

PATH = "out/judge.pt"
SURE = 0.9


class Judge(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),       # 16×16
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),      # 8×8
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(128, len(SHAPES)), nn.Linear(128, len(COLORS)),
                                    nn.Linear(128, len(POSITIONS)), nn.Linear(128, 2)])     # last: junk / clean

    def forward(self, x):
        h = self.body(x)
        return [head(h) for head in self.heads]


def to_tensor(images):
    return torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0


def blur(x, sigma):
    k = torch.arange(-4, 5, dtype=torch.float32)
    k = torch.exp(-k ** 2 / (2 * sigma ** 2))
    k = (k / k.sum()).view(1, 1, 1, 9).repeat(3, 1, 1, 1)
    x = nn.functional.conv2d(nn.functional.pad(x, (4, 4, 0, 0), mode="replicate"), k, groups=3)
    return nn.functional.conv2d(nn.functional.pad(x, (0, 0, 4, 4), mode="replicate"), k.transpose(2, 3), groups=3)


PALETTE = np.array(list(COLORS.values())) / 255.0


def off_palette_colors(n, rng):
    """Random colors at least 0.3 away (RGB, 0–1) from every palette color."""
    out = []
    while len(out) < n:
        c = rng.uniform(0.15, 1.0, 3)
        if np.sqrt(((PALETTE - c) ** 2).sum(1)).min() > 0.3:
            out.append(c)
    return torch.tensor(np.array(out), dtype=torch.float32).view(n, 3, 1, 1)


def junk(n, rng, x, pos):
    """n junk pictures of each of five kinds. pos: the position label of each picture in x."""
    kinds = []
    i = torch.tensor(rng.integers(0, len(x), n))
    j = torch.tensor(rng.integers(0, len(x), n))
    same = pos[i] == pos[j]                                                    # pairs must be at different places
    j[same] = torch.tensor([int(rng.choice(np.nonzero(pos != pos[k])[0])) for k in i[same]], dtype=torch.long)
    kinds.append((x[i] + x[j]) / 2)                                            # two pictures averaged
    two = []
    for _ in range(n):                                                         # two shapes in one picture
        p1, p2 = rng.choice(9, 2, replace=False)
        a = draw(SHAPES[rng.integers(3)], COLOR_NAMES[rng.integers(5)], p1, rng)
        b = draw(SHAPES[rng.integers(3)], COLOR_NAMES[rng.integers(5)], p2, rng)
        mask = (b != a[0, 0]).any(-1)
        a[mask] = b[mask]
        two.append(a)
    kinds.append(to_tensor(np.stack(two)))
    off = x[i].clone()                                                         # a color outside the palette
    shape_px = (off - off[:, :, :1, :1]).abs().sum(1, keepdim=True) > 0.1
    kinds.append(torch.where(shape_px, off_palette_colors(n, rng), off))
    kinds.append(blur(x[j], float(rng.uniform(1.6, 2.5))))                     # heavy blur
    kinds.append((x[i] + torch.tensor(rng.normal(0, 0.25, (n, 3, 32, 32)), dtype=torch.float32)).clamp(0, 1))
    return torch.cat(kinds)


def train_judge(epochs=8, seed=123):
    with torch.random.fork_rng():                  # leave the caller's random state untouched
        return _train_judge(epochs, seed)


def _train_judge(epochs, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    imgs, _, labels = make_dataset(20_000, seed=seed)
    x, y = to_tensor(imgs), torch.tensor(labels)
    pos = y[:, 2].numpy()
    judge = Judge()
    opt = torch.optim.Adam(judge.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 128):
            idx = perm[i:i + 128]
            real = x[idx]
            sigma = float(rng.uniform(0.3, 0.8))                               # half lightly softened, still clean
            noise = torch.rand(len(real), 1, 1, 1) * 0.04 * torch.randn_like(real)
            soft = torch.rand(len(real), 1, 1, 1) < 0.5
            real = torch.where(soft, (blur(real, sigma) + noise).clamp(0, 1), real)
            bad = junk(26, rng, x, pos)
            logits = judge(torch.cat([real, bad]))
            loss = sum(nn.functional.cross_entropy(l[:len(real)], y[idx, k]) for k, l in enumerate(logits[:3]))
            clean = torch.cat([torch.ones(len(real)), torch.zeros(len(bad))]).long()
            loss = loss + nn.functional.cross_entropy(logits[3], clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    torch.save(judge.state_dict(), PATH)
    return judge


SHARED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g03_variational_autoencoders", "out", "judge.pt")


def load_judge():
    """The judge from episode 3 (its weights are in the repository), else a local copy, else train one (seed 123)."""
    judge = Judge()
    for path in (SHARED, PATH):
        if os.path.exists(path):
            judge.load_state_dict(torch.load(path))
            return judge.eval()
    return train_judge().eval()


@torch.no_grad()
def recognizable(judge, x):
    """Share of pictures the judge reads with ≥ 90% confidence and finds clean rather than junk; and its readings."""
    probs = [l.softmax(-1) for l in judge(x)]
    sure = torch.stack([p.max(-1).values >= SURE for p in probs[:3]] + [probs[3][:, 1] >= 0.5], 1).all(1)
    labels = torch.stack([p.argmax(-1) for p in probs[:3]], 1)
    return sure.float().mean().item(), labels


@torch.no_grad()
def breakdown(judge, x):
    """Shares of pictures that are readable (≥ 90% sure of shape, color, position), clean, and both."""
    probs = [l.softmax(-1) for l in judge(x)]
    read = torch.stack([p.max(-1).values >= SURE for p in probs[:3]], 1).all(1)
    clean = probs[3][:, 1] >= 0.5
    return {"readable": read.float().mean().item(), "clean": clean.float().mean().item(),
            "both": (read & clean).float().mean().item()}


if __name__ == "__main__":
    judge = load_judge()
    imgs, _, labels = make_dataset(2_000, seed=1)
    share, read = recognizable(judge, to_tensor(imgs))
    acc = (read == torch.tensor(labels)).all(1).float().mean().item()
    print(f"real test pictures: {share:.1%} recognizable, {acc:.1%} read correctly")
    x = to_tensor(imgs)
    for sigma in (0.5, 0.7, 1.0):
        print(f"real pictures blurred (sigma {sigma}): {recognizable(judge, blur(x, sigma))[0]:.1%} recognizable")
    bad = junk(400, np.random.default_rng(7), x, np.array(labels)[:, 2])
    for k, name in enumerate(["averaged pairs", "two shapes", "off-palette color", "heavy blur", "noise"]):
        print(f"junk, {name}: {recognizable(judge, bad[k * 400:(k + 1) * 400])[0]:.1%} recognizable")
