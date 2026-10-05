"""How LLMs Work: Deep Dive, episode 6 — Long context: stretching RoPE.

A controlled experiment at small scale: train a tiny GPT with RoPE on Tiny Shakespeare with a context of only 64
tokens, then make it read 256 tokens, and measure the loss at each position:

  1. no change (plain extrapolation: positions 64-255 were never seen),
  2. position interpolation: squeeze positions by 64/256 so every angle stays inside the trained range,
  3. "NTK-aware" scaling: raise the RoPE base, so slow pairs stretch a lot and fast pairs hardly change,
  4. position interpolation plus a short fine-tune (200 steps) at length 256.

    pip install -r requirements.txt
    python long_context.py
CPU only; training the base model takes about 15 minutes (saved as rope_gpt.pt and reused).
"""
import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = Path(__file__).parent
torch.manual_seed(1337)
torch.set_num_threads(8)
text = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
V, D, H, L = len(chars), 128, 4, 4
HD = D // H
TRAIN_LEN, LONG_LEN = 64, 256


class Rope:
    def __init__(self, base=10_000.0, scale=1.0):
        self.freqs = base ** (-torch.arange(0, HD, 2).float() / HD)
        self.scale = scale                      # position interpolation: multiply positions by this

    def __call__(self, x):                      # x: (B, H, T, HD)
        T = x.shape[2]
        angle = (torch.arange(T).float() * self.scale)[:, None] * self.freqs[None]
        cos, sin = angle.cos(), angle.sin()
        x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, rope):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, H, HD).transpose(1, 2) for t in (q, k, v))
        q, k = rope(q), rope(k)                                         # RoPE on queries and keys only
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(att.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class RopeGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)                                   # no position embedding table at all
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)
        self.rope = Rope()

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.rope)
        return self.head(self.ln(x))


def batch(d, T, bs=32, g=None):
    ix = torch.randint(len(d) - T - 1, (bs,), generator=g)
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


def train(model, steps, T, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        x, y = batch(train_data, T)
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


@torch.no_grad()
def loss_by_position(model, T=LONG_LEN, batches=40):
    g = torch.Generator().manual_seed(0)
    total = torch.zeros(T)
    for _ in range(batches):
        x, y = batch(val_data, T, g=g)
        total += F.cross_entropy(model(x).transpose(1, 2), y, reduction="none").mean(0)
    return total / batches


def report(name, model):
    lp = loss_by_position(model)
    parts = [(0, 64), (64, 128), (128, 256)]
    print(f"{name:<34}" + "".join(f"{lp[a:b].mean():>10.2f}" for a, b in parts))
    return lp


ckpt = HERE / "rope_gpt.pt"
base = RopeGPT()
if ckpt.exists():
    base.load_state_dict(torch.load(ckpt))
else:
    print(f"training a {sum(p.numel() for p in base.parameters()):,}-parameter RoPE GPT, context {TRAIN_LEN} ...")
    train(base, 3000, TRAIN_LEN, 1e-3)
    torch.save(base.state_dict(), ckpt)
base.eval()

print(f"\nvalidation loss at each position of a {LONG_LEN}-token text (trained on {TRAIN_LEN} tokens):")
print(f"{'':<34}{'pos 0-63':>10}{'64-127':>10}{'128-255':>10}")
curves = {}
curves["plain"] = report("1. no change (extrapolation)", base)
pi = copy.deepcopy(base)
pi.rope = Rope(scale=TRAIN_LEN / LONG_LEN)
curves["pi"] = report("2. position interpolation (x 1/4)", pi)
ntk = copy.deepcopy(base)
ntk.rope = Rope(base=10_000.0 * (LONG_LEN / TRAIN_LEN) ** (HD / (HD - 2)))
curves["ntk"] = report("3. NTK-aware base scaling", ntk)
tuned = copy.deepcopy(pi)
train(tuned, 200, LONG_LEN, 3e-4)
curves["pi_ft"] = report("4. interpolation + 200 steps at 256", tuned)
torch.save({k: v.tolist() for k, v in curves.items()}, HERE / "curves.pt")
