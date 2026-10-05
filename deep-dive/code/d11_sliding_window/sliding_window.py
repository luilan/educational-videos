"""Deep Dive, episode 11: Sliding Windows and Sparse Attention.

1. A sliding-window mask: each token sees only the last W tokens, so the work grows like T x W, not T x T.
2. Stacked layers see further: measure, with gradients, how far back each output can reach after 1-4 layers.
3. Train the same tiny GPT on 256-character texts with full attention and with windows of 64 and 16.
4. A task that needs long range: repeat a random 128-token sequence. Can a window of 16 copy from 128 tokens back?
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = Path(__file__).parent
torch.set_num_threads(8)


def window_mask(T, W):
    """True where attention is allowed: causal, and at most W - 1 tokens back."""
    i, j = torch.arange(T)[:, None], torch.arange(T)[None]
    return (j <= i) & (i - j < W)


# ---------------------------------------------------------------- 1. the mask and the work
print("1. window of 4 on 8 tokens (1 = allowed):")
print(window_mask(8, 4).int())
for T in (4_096, 131_072):
    full = T * (T + 1) // 2
    win = int(window_mask(min(T, 8192), 4096).sum()) if T <= 8192 else 4096 * T - 4096 * 4095 // 2
    print(f"   {T:,} tokens: full causal {full:,} pairs, window 4,096 {win:,} pairs ({win / full:.1%})")

# ---------------------------------------------------------------- the model
text = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
V, D, H, L, T = len(chars), 128, 4, 4, 256


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, mask):
        B, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:t, :t])
        x = x + self.proj(att.transpose(1, 2).reshape(B, t, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, W, layers=L):
        super().__init__()
        self.register_buffer("mask", window_mask(T, W))
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.ModuleList([Block() for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)

    def embed(self, idx):
        return self.emb(idx) + self.pos(torch.arange(idx.shape[1]))

    def forward_from(self, x):
        for b in self.blocks:
            x = b(x, self.mask)
        return self.head(self.ln(x))

    def forward(self, idx):
        return self.forward_from(self.embed(idx))


# ---------------------------------------------------------------- 2. how far back can a token reach?
print("\n2. window 16: how far back the last token's output depends on the input (via gradients)")
torch.manual_seed(0)
for layers in (1, 2, 3, 4):
    m = TinyGPT(16, layers)
    x = m.embed(torch.randint(V, (1, T))).detach().requires_grad_(True)
    m.forward_from(x)[0, -1].sum().backward()
    touched = (x.grad[0].abs().sum(-1) > 0).nonzero().flatten()
    print(f"   {layers} layer{'s' if layers > 1 else ' '}: reaches back {T - 1 - touched.min().item()} tokens "
          f"(= {layers} x 15)")


# ---------------------------------------------------------------- 3. train full vs windows
def batch(d, bs=16, g=None):
    ix = torch.randint(len(d) - T - 1, (bs,), generator=g)
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


@torch.no_grad()
def val_loss(model):
    g = torch.Generator().manual_seed(0)
    total = torch.zeros(T)
    for _ in range(20):
        x, y = batch(val_data, g=g)
        total += F.cross_entropy(model(x).transpose(1, 2), y, reduction="none").mean(0)
    lp = total / 20
    return lp.mean().item(), lp[:16].mean().item(), lp[128:].mean().item()


print(f"\n3. tiny GPT, {L} layers, {T}-character texts, 1,500 steps")
for W in (256, 64, 16):
    torch.manual_seed(1337)
    model = TinyGPT(W)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(1500):
        x, y = batch(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    mean, early, late = val_loss(model)
    pairs = int(window_mask(T, W).sum())
    name = "full attention" if W == T else f"window {W}"
    print(f"   {name:<15} val loss {mean:.3f} (positions 0-15: {early:.2f}, 128-255: {late:.3f})   "
          f"{pairs:,} pairs ({pairs / (T * (T + 1) // 2):.0%})")


# ---------------------------------------------------------------- 4. a long-range task: copy from 128 tokens back
def copy_batch(bs=16, g=None):
    half = torch.randint(16, (bs, T // 2), generator=g)               # 16 random symbols, then the same again
    seq = torch.cat([half, half], 1)
    return seq[:, :-1], seq[:, 1:]


print(f"\n4. copy task: 128 random symbols (out of 16), then the same 128 again; 1,500 steps")
print(f"   loss on the repeated half (guessing: ln 16 = {torch.log(torch.tensor(16.0)):.2f})")
for W in (256, 64, 16):
    torch.manual_seed(1337)
    model = TinyGPT(W)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(1500):
        x, y = copy_batch()
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        x, y = copy_batch(64, torch.Generator().manual_seed(0))
        lp = F.cross_entropy(model(x).transpose(1, 2), y, reduction="none").mean(0)
    name = "full attention" if W == T else f"window {W}"
    print(f"   {name:<15} {lp[T // 2:].mean():.3f}   (reach with {L} layers: {min(T - 1, L * (W - 1))} tokens)")
