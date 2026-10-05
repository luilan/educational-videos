"""Deep Dive, episode 23: Scaling Laws.

1. Model size: train six tiny GPTs, from about 8 thousand to 1.2 million parameters, on the same data; fit a power
   law to loss vs parameters.
2. Data: train one model and record its validation loss as the number of training tokens doubles.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = Path(__file__).parent
torch.set_num_threads(8)
text = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
V, T, BS = len(chars), 64, 32


class Block(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.D, self.H = D, H
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        B, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(self.D, dim=2)
        q, k, v = (z.view(B, t, self.H, self.D // self.H).transpose(1, 2) for z in (q, k, v))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, self.D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, D, L, H=4):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.Sequential(*[Block(D, H) for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)

    def forward(self, idx):
        return self.head(self.ln(self.blocks(self.emb(idx) + self.pos(torch.arange(idx.shape[1])))))


def batch(d, bs=BS, g=None):
    ix = torch.randint(len(d) - T - 1, (bs,), generator=g)
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


@torch.no_grad()
def val_loss(model):
    g = torch.Generator().manual_seed(0)
    return sum(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()
               for x, y in (batch(val_data, g=g) for _ in range(40))) / 40


def non_embedding(model):
    return sum(p.numel() for name, p in model.named_parameters() if not name.startswith(("emb", "pos")))


def fit_power_law(xs, ys):
    """Least-squares line through (log x, log y): y = a * x^slope."""
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)
    return math.exp(my - slope * mx), slope


# ---------------------------------------------------------------- 1. model size
STEPS = 2000
print(f"1. six model sizes, each trained {STEPS:,} steps on {STEPS * BS * T:,} tokens (AdamW, lr 0.002)")
sizes = []
for D, L in ((16, 2), (32, 2), (48, 3), (64, 3), (96, 4), (128, 6)):
    torch.manual_seed(1337)
    model = TinyGPT(D, L)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(STEPS):
        x, y = batch(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    N, vl = non_embedding(model), val_loss(model)
    sizes.append((N, vl))
    print(f"   width {D:>3}, {L} layers: {N:>9,} parameters (excluding embeddings), val loss {vl:.3f}")
a, s = fit_power_law(*zip(*sizes))
print(f"   power-law fit: loss ≈ {a:.2f} × N^{s:.3f}")
for N, vl in sizes:
    print(f"      N = {N:>9,}: fit {a * N ** s:.3f}, measured {vl:.3f}")

# ---------------------------------------------------------------- 2. data
print("\n2. width 96, 4 layers: validation loss as the training tokens double (lr 0.002, constant)")
torch.manual_seed(1337)
model = TinyGPT(96, 4)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
checkpoints = (125, 250, 500, 1000, 2000, 4000)
points = []
for step in range(1, checkpoints[-1] + 1):
    x, y = batch(train_data)
    loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step in checkpoints:
        model.eval()
        points.append((step * BS * T, val_loss(model)))
        model.train()
        print(f"   {step * BS * T:>10,} tokens: val loss {points[-1][1]:.3f}")
a2, s2 = fit_power_law(*zip(*points))
print(f"   power-law fit: loss ≈ {a2:.2f} × tokens^{s2:.3f}")
