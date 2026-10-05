"""Deep Dive, episode 22: Mixed Precision.

1. Three number formats: float32, float16, bfloat16. Range and precision.
2. Rounding: small updates to a weight of 1.0 vanish in 16-bit formats.
3. Underflow: tiny gradients become zero in float16; loss scaling rescues them.
4. Train the same tiny GPT in float32, with bfloat16 compute and float32 master weights, and with pure bfloat16 weights.
5. GPT-2 in bfloat16: half the memory, almost the same loss.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = Path(__file__).parent
torch.set_num_threads(8)

# ---------------------------------------------------------------- 1. the formats
print("1. format     bits   largest value     step just above 1.0   smallest positive (normal)")
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(f"   {str(dt)[6:]:<10} {fi.bits:>4}   {fi.max:>14.4g}   {fi.eps:>18.4g}   {fi.tiny:>18.4g}")

# ---------------------------------------------------------------- 2. rounding
print("\n2. a weight of 1.0 plus a small update")
for upd in (1e-2, 1e-3, 1e-4):
    row = []
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        w = torch.tensor(1.0, dtype=dt)
        row.append(f"{str(dt)[6:]}: {(w + torch.tensor(upd, dtype=dt)).item():.7f}")
    print(f"   1.0 + {upd:g}  →  " + "   ".join(row))

# ---------------------------------------------------------------- 3. underflow and loss scaling
g = torch.tensor([1e-3, 1e-5, 1e-7, 1e-9])
print(f"\n3. gradients {g.tolist()}")
print(f"   stored in float16:              {g.half().float().tolist()}")
print(f"   × 1024 in float16, then ÷ 1024: {((g * 1024).half().float() / 1024).tolist()}")
print(f"   stored in bfloat16:             {g.bfloat16().float().tolist()}")

# ---------------------------------------------------------------- 4. training
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
def val_loss(model, dtype=None):
    g = torch.Generator().manual_seed(0)
    total = 0.0
    for _ in range(40):
        x, y = batch(val_data, g=g)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype is not None):
            logits = model(x)
        total += F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1)).item()
    return total / 40


print("\n4. tiny GPT, 4 layers, 2,000 steps, AdamW lr 0.001: validation loss")
for mode in ("float32", "bf16 compute + float32 weights", "pure bfloat16 weights"):
    torch.manual_seed(1337)
    model = TinyGPT()
    if mode == "pure bfloat16 weights":
        model = model.bfloat16()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2000):
        x, y = batch(train_data)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mode.startswith("bf16")):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    print(f"   {mode:<32} {val_loss(model, None if mode == 'float32' else torch.bfloat16):.3f}")

# ---------------------------------------------------------------- 5. GPT-2 in bfloat16
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tok.model_max_length = 10 ** 6
ids = tok(text[100_000:106_000], return_tensors="pt").input_ids[:, :1024]
print("\n5. GPT-2 on 1,024 tokens of Shakespeare")
for dt in (torch.float32, torch.bfloat16):
    m = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=dt).eval()
    with torch.no_grad():
        logits = m(ids).logits.float()
    loss = F.cross_entropy(logits[0, :-1], ids[0, 1:]).item()
    size = sum(p.numel() * p.element_size() for p in m.parameters()) / 2 ** 20
    print(f"   {str(dt)[6:]:<9} weights {size:,.0f} MiB   loss {loss:.4f}")
