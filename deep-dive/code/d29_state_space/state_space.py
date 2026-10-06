"""Deep Dive, episode 29: State-Space Models.

Replace attention with a recurrence: a fixed-size state h that is updated once per token,
    h_t = a ⊙ h_{t-1} + (1 − a) ⊙ u_t,
1. fixed decay a (learned per channel, the same for every token), and
2. selective decay a_t computed from the token itself (the idea behind Mamba-style models).
Then compare with attention: loss after training, memory while generating, and time per generated token.
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


class Mixer(nn.Module):
    """Mixes information across positions: attention, or a recurrence with fixed or selective decay."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "attention":
            self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        else:
            self.inp, self.gate, self.out = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
            if kind == "fixed":
                self.decay = nn.Parameter(torch.linspace(-2, 4, D))       # a = sigmoid(decay): one rate per channel
            else:
                self.decay = nn.Linear(D, D)                              # a_t = sigmoid(W x_t): chosen per token

    def forward(self, x):
        B, t, _ = x.shape
        if self.kind == "attention":
            q, k, v = self.qkv(x).split(D, dim=2)
            q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
            return self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, D))
        u = self.inp(x)
        a = torch.sigmoid(self.decay).expand(B, t, D) if self.kind == "fixed" else torch.sigmoid(self.decay(x))
        h, outs = torch.zeros(B, D), []
        for i in range(t):                                                # one fixed-size update per token
            h = a[:, i] * h + (1 - a[:, i]) * u[:, i]
            outs.append(h)
        y = torch.stack(outs, 1)
        return self.out(y * F.silu(self.gate(x)))


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.mix = Mixer(kind)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        x = x + self.mix(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D) if kind == "attention" else None   # a recurrence knows the order by itself
        self.blocks = nn.Sequential(*[Block(kind) for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)

    def forward(self, idx):
        x = self.emb(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1]))
        return self.head(self.ln(self.blocks(x)))


def batch(d, bs=32, g=None):
    ix = torch.randint(len(d) - T - 1, (bs,), generator=g)
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


@torch.no_grad()
def val_loss(model):
    g = torch.Generator().manual_seed(0)
    return sum(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()
               for x, y in (batch(val_data, g=g) for _ in range(40))) / 40


# ---------------------------------------------------------------- 1-2. training
print(f"1-2. tiny language models, {L} layers, 2,000 steps: validation loss")
for kind in ("attention", "fixed", "selective"):
    torch.manual_seed(1337)
    model = TinyLM(kind)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2000):
        x, y = batch(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    name = {"attention": "attention", "fixed": "recurrence, fixed decay", "selective": "recurrence, selective decay"}[kind]
    print(f"   {name:<30} val loss {val_loss(model):.3f}   {sum(p.numel() for p in model.parameters()):,} parameters")

# ---------------------------------------------------------------- 3. memory while generating
print(f"\n3. memory per layer while generating (float32, {D} numbers per token)")
for tokens in (1_000, 100_000):
    print(f"   after {tokens:>7,} tokens: attention KV cache {2 * tokens * D * 4 / 2**20:8.2f} MiB, "
          f"recurrent state {D * 4 / 1024:.2f} KiB")

# ---------------------------------------------------------------- 4. time per generated token
print("\n4. time to process one new token in one layer (this CPU)")
torch.set_num_threads(1)
q, h, a, u = torch.randn(1, D), torch.zeros(1, D), torch.rand(1, D), torch.randn(1, D)
for tokens in (1_000, 10_000, 100_000):
    K, Vv = torch.randn(tokens, D), torch.randn(tokens, D)
    reps = 200
    t0 = time.perf_counter()
    for _ in range(reps):
        w = (q @ K.T / D ** 0.5).softmax(-1)
        o = w @ Vv
    att = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        h2 = a * h + (1 - a) * u
    rec = (time.perf_counter() - t0) / reps
    print(f"   {tokens:>7,} tokens of context: attention {att * 1e6:8.1f} µs, recurrence {rec * 1e6:5.1f} µs")
