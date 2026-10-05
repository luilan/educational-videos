"""Deep Dive, episode 16: Pre-Norm, Post-Norm, and Stability.

Post-norm (the original Transformer, BERT):  x = norm(x + block(x))
Pre-norm  (GPT-2 onward, Llama, Qwen):       x = x + block(norm(x))
1. At initialization: how big is the gradient reaching each layer?
2. Train a 12-layer tiny GPT both ways, at two learning rates, with and without warmup.
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
V, D, H, L, T = len(chars), 128, 4, 12, 64


class Block(nn.Module):
    def __init__(self, pre):
        super().__init__()
        self.pre = pre
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def attn(self, h):
        B, t, _ = h.shape
        q, k, v = self.qkv(h).split(D, dim=2)
        q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        return self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, D))

    def forward(self, x):
        if self.pre:                                               # pre-norm: normalize what the block reads
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))
        x = self.ln1(x + self.attn(x))                             # post-norm: normalize the stream itself
        return self.ln2(x + self.mlp(x))


class TinyGPT(nn.Module):
    def __init__(self, pre):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.Sequential(*[Block(pre) for _ in range(L)])
        self.ln = nn.LayerNorm(D) if pre else nn.Identity()       # pre-norm needs one final norm
        self.head = nn.Linear(D, V)

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


# ---------------------------------------------------------------- 1. gradients at initialization
print(f"1. {L}-layer tiny GPT at initialization: gradient size reaching each layer's weights")
for pre in (True, False):
    torch.manual_seed(1337)
    model = TinyGPT(pre)
    x, y = batch(train_data, g=torch.Generator().manual_seed(0))
    F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).backward()
    norms = [torch.cat([p.grad.flatten() for p in b.parameters()]).norm().item() for b in model.blocks]
    name = "pre-norm " if pre else "post-norm"
    print(f"   {name}: layer 0 {norms[0]:.3f}   layer 6 {norms[6]:.3f}   layer 11 {norms[11]:.3f}   "
          f"(last / first = {norms[11] / norms[0]:.1f})")

# ---------------------------------------------------------------- 2. training
STEPS = 1500


def train(pre, lr, warmup):
    torch.manual_seed(1337)
    model = TinyGPT(pre)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) if warmup else 1.0)
    curve = []
    for step in range(1, STEPS + 1):
        x, y = batch(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step in (100, 500, 1500):
            model.eval()
            curve.append(val_loss(model))
            model.train()
    return curve


print(f"\n2. {L}-layer tiny GPT, {STEPS:,} steps: validation loss at 100 / 500 / 1,500 steps "
      f"(uniform guessing: {math.log(V):.2f})")
for lr in (1e-3, 3e-3):
    for warmup in (0, 300):
        for pre in (True, False):
            c = train(pre, lr, warmup)
            name = "pre-norm " if pre else "post-norm"
            print(f"   lr {lr:g}, warmup {warmup:>3}: {name} " + " / ".join(f"{v:.2f}" for v in c))

# ---------------------------------------------------------------- 3. the first 100 steps at the high learning rate
print("\n3. training loss in the first 100 steps, learning rate 0.003, no warmup")
for pre in (True, False):
    torch.manual_seed(1337)
    model = TinyGPT(pre)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    seen = []
    for step in range(1, 101):
        x, y = batch(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (1, 5, 10, 20, 50, 100):
            seen.append(f"{step}: {loss.item():.2f}")
    print(f"   {'pre-norm ' if pre else 'post-norm'} " + "   ".join(seen))
