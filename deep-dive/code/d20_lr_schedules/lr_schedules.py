"""Deep Dive, episode 20: Learning-Rate Warmup and Schedules.

Train the same tiny GPT for 3,000 steps with five learning-rate schedules and compare the validation loss along the way:
constant (two values), warmup + cosine decay, warmup + linear decay, and warmup-stable-decay (WSD).
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
V, D, H, L, T = len(chars), 128, 4, 4, 64
STEPS, WARMUP, PEAK = 3000, 200, 3e-3


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


def batch(d, bs=32, g=None):
    ix = torch.randint(len(d) - T - 1, (bs,), generator=g)
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


@torch.no_grad()
def val_loss(model):
    g = torch.Generator().manual_seed(0)
    return sum(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()
               for x, y in (batch(val_data, g=g) for _ in range(40))) / 40


def warm(s):
    return min(1.0, (s + 1) / WARMUP)


SCHEDULES = {                                     # learning rate at step s (0-based), as a multiple of PEAK
    "constant 0.001": lambda s: 1 / 3,
    "constant 0.003": lambda s: 1.0,
    "warmup + cosine": lambda s: warm(s) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, max(0, s - WARMUP)
                                                                                         / (STEPS - WARMUP))))),
    "warmup + linear": lambda s: warm(s) * (1 - 0.9 * max(0, s - WARMUP) / (STEPS - WARMUP)),
    "warmup-stable-decay": lambda s: warm(s) * (1.0 if s < 0.8 * STEPS else 1 - 0.9 * (s - 0.8 * STEPS) / (0.2 * STEPS)),
}

print(f"tiny GPT, {L} layers, {STEPS:,} steps, peak learning rate {PEAK:g}, {WARMUP} warmup steps, decays end at 10%")
print("validation loss at steps " + ", ".join(str(s) for s in range(500, STEPS + 1, 500)))
for name, f in SCHEDULES.items():
    torch.manual_seed(1337)
    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, f)
    curve = []
    for step in range(1, STEPS + 1):
        x, y = batch(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 2400:
            model.eval()
            curve.append((step, val_loss(model)))
            model.train()
    print(f"   {name:<20} " + "  ".join(f"{v:.3f}" for s, v in curve if s % 500 == 0)
          + f"   (step 2,400: {dict(curve)[2400]:.3f})")
