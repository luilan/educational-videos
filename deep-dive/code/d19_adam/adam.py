"""Deep Dive, episode 19: Adam and AdamW.

1. Adam by hand: running averages of the gradient (m) and of its square (v); step = lr * m / (sqrt(v) + eps).
   Check it against torch.optim.Adam.
2. Scale the loss by 1,000: SGD's steps grow 1,000 times, Adam's do not change.
3. Train the same tiny GPT with SGD, SGD + momentum, Adam and AdamW.
4. Weight decay: L2 added to the gradient (Adam) vs decoupled decay (AdamW).
5. Memory: Adam keeps two extra numbers per parameter.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = Path(__file__).parent
torch.set_num_threads(8)

# ---------------------------------------------------------------- 1. Adam by hand
class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
        self.params, self.lr, (self.b1, self.b2), self.eps = list(params), lr, betas, eps
        self.wd, self.decoupled, self.t = weight_decay, decoupled, 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            if self.wd and self.decoupled:
                p.mul_(1 - self.lr * self.wd)                    # AdamW: shrink the weight directly
            elif self.wd:
                g = g + self.wd * p                              # Adam + L2: add the decay to the gradient
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)            # running average of the gradient
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)     # running average of its square
            m_hat = m / (1 - self.b1 ** self.t)                   # bias correction (both start at zero)
            v_hat = v / (1 - self.b2 ** self.t)
            p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))


torch.manual_seed(0)
w1 = torch.randn(5, 3, requires_grad=True)
w2 = w1.detach().clone().requires_grad_(True)
x = torch.randn(8, 5)
mine, ref = MyAdam([w1], lr=0.01), torch.optim.Adam([w2], lr=0.01)
for _ in range(10):
    for w, opt in ((w1, mine), (w2, ref)):
        w.grad = None
        ((x @ w) ** 2).mean().backward()
        opt.step()
print(f"1. my Adam vs torch.optim.Adam after 10 steps: largest difference {(w1 - w2).abs().max().item():.1e}")

# ---------------------------------------------------------------- 2. scale invariance
print("\n2. first step size (largest change in any weight), loss scaled by 1 and by 1,000")
for scale in (1, 1000):
    for name in ("SGD", "Adam"):
        torch.manual_seed(0)
        w = torch.randn(5, 3, requires_grad=True)
        before = w.detach().clone()
        opt = torch.optim.SGD([w], lr=0.01) if name == "SGD" else torch.optim.Adam([w], lr=0.01)
        (scale * ((x @ w) ** 2).mean()).backward()
        opt.step()
        print(f"   {name:<4} loss × {scale:<5} step {(w.detach() - before).abs().max().item():.4f}")

# ---------------------------------------------------------------- 3. train with each optimizer
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


def batch(d, bs=32, g=None):
    ix = torch.randint(len(d) - T - 1, (bs,), generator=g)
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


@torch.no_grad()
def val_loss(model):
    g = torch.Generator().manual_seed(0)
    return sum(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()
               for x, y in (batch(val_data, g=g) for _ in range(40))) / 40


def train(make_opt, steps=1500):
    torch.manual_seed(1337)
    model = TinyGPT()
    opt = make_opt(model.parameters())
    for _ in range(steps):
        x, y = batch(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model, val_loss(model)


print("\n3. tiny GPT, 4 layers, 1,500 steps: validation loss")
runs = {
    "SGD, lr 0.1": lambda p: torch.optim.SGD(p, lr=0.1),
    "SGD, lr 1.0": lambda p: torch.optim.SGD(p, lr=1.0),
    "SGD + momentum 0.9, lr 0.1": lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9),
    "Adam, lr 0.001": lambda p: torch.optim.Adam(p, lr=1e-3),
    "my Adam, lr 0.001": lambda p: MyAdam(p, lr=1e-3),
}
for name, make in runs.items():
    _, loss = train(make)
    print(f"   {name:<28} {loss:.3f}")

# ---------------------------------------------------------------- 4. weight decay
print("\n4. weight decay 0.1, lr 0.003: Adam + L2 in the gradient vs AdamW (decoupled)")
for name, make in (("no decay", lambda p: MyAdam(p, lr=3e-3)),
                   ("Adam + L2", lambda p: MyAdam(p, lr=3e-3, weight_decay=0.1)),
                   ("AdamW", lambda p: MyAdam(p, lr=3e-3, weight_decay=0.1, decoupled=True))):
    model, loss = train(make)
    size = torch.cat([p.detach().flatten() for p in model.parameters()]).norm().item()
    print(f"   {name:<10} val loss {loss:.3f}   size of all weights {size:.1f}")

# ---------------------------------------------------------------- 5. memory
n_gpt2 = 124_439_808
print(f"\n5. GPT-2 small: {n_gpt2:,} parameters = {n_gpt2 * 4 / 2**20:,.0f} MiB in float32; "
      f"Adam's m and v add {2 * n_gpt2 * 4 / 2**20:,.0f} MiB")
