"""Deep Dive, episode 21: Batch Size and Gradient Accumulation.

1. Gradient accumulation: 4 micro-batches of 8, each loss divided by 4, give the same gradient as one batch of 32.
2. Gradient noise: how close is a batch's gradient to the "true" gradient (from 4,096 sequences), by batch size?
3. Same number of training sequences, different batch sizes: loss, number of steps, and time per sequence.
"""
import time
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
V, D, H, L, T = len(chars), 128, 4, 4, 64


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        B, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.Sequential(*[Block() for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)

    def forward(self, idx):
        return self.head(self.ln(self.blocks(self.emb(idx) + self.pos(torch.arange(idx.shape[1])))))


def batch(d, bs, g=None):
    ix = torch.randint(len(d) - T - 1, (bs,), generator=g)
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


def lm_loss(model, x, y):
    return F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))


def flat_grad(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()])


@torch.no_grad()
def val_loss(model):
    g = torch.Generator().manual_seed(0)
    return sum(lm_loss(model, *batch(val_data, 32, g)).item() for _ in range(40)) / 40


# ---------------------------------------------------------------- 1. gradient accumulation
torch.manual_seed(0)
model = TinyGPT()
x, y = batch(train_data, 32, torch.Generator().manual_seed(1))
model.zero_grad()
lm_loss(model, x, y).backward()
big = flat_grad(model).clone()
model.zero_grad()
for k in range(4):                                          # 4 micro-batches of 8
    (lm_loss(model, x[8 * k:8 * k + 8], y[8 * k:8 * k + 8]) / 4).backward()   # gradients add up across backward calls
acc = flat_grad(model)
print(f"1. one batch of 32 vs 4 accumulated micro-batches of 8: largest gradient difference "
      f"{(big - acc).abs().max().item():.1e} (gradient size {big.norm().item():.3f})")

# ---------------------------------------------------------------- 2. gradient noise
torch.manual_seed(0)
model = TinyGPT()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for _ in range(300):                                        # train a little, so the gradients are typical
    xb, yb = batch(train_data, 32)
    opt.zero_grad()
    lm_loss(model, xb, yb).backward()
    opt.step()
g = torch.Generator().manual_seed(2)
true = torch.zeros_like(big)
for _ in range(16):                                         # "true" gradient: average over 16 x 256 = 4,096 sequences
    model.zero_grad()
    lm_loss(model, *batch(train_data, 256, g)).backward()
    true += flat_grad(model) / 16
print("\n2. after 300 steps: similarity (cosine) between a batch's gradient and the gradient of 4,096 sequences")
for bs in (1, 4, 16, 64, 256):
    sims = []
    for _ in range(8):
        model.zero_grad()
        lm_loss(model, *batch(train_data, bs, g)).backward()
        sims.append(F.cosine_similarity(flat_grad(model), true, dim=0).item())
    print(f"   batch {bs:>3}: {sum(sims) / len(sims):.2f}")

# ---------------------------------------------------------------- 3. same sequences, different batch sizes
TOTAL = 48_000                                              # training sequences seen in every run
print(f"\n3. {TOTAL:,} training sequences of {T} characters in every run (AdamW)")
for bs, lr in ((8, 1e-3), (32, 1e-3), (128, 1e-3), (128, 2e-3)):
    torch.manual_seed(1337)
    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = TOTAL // bs
    t0 = time.perf_counter()
    for _ in range(steps):
        xb, yb = batch(train_data, bs)
        opt.zero_grad()
        lm_loss(model, xb, yb).backward()
        opt.step()
    sec = time.perf_counter() - t0
    model.eval()
    print(f"   batch {bs:>3}, lr {lr:g}: {steps:>5,} steps, val loss {val_loss(model):.3f}, "
          f"{sec / TOTAL * 1000:.2f} ms per sequence")
