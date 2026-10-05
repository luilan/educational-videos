"""Deep Dive, episode 13: The Residual Stream.

1. GPT-2's residual stream: x = x + attention(x), then x = x + mlp(x), twelve times. How big is it, layer by layer?
2. How big is each block's update, compared with the stream it adds to?
3. Delete one whole layer at a time: how much does the loss suffer?
4. Train the same 8-layer tiny GPT with and without residual connections.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = Path(__file__).parent
torch.set_num_threads(8)
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tok.model_max_length = 10 ** 6
gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
tr = gpt2.transformer
text = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()
ids = tok(text[100_000:106_000], return_tensors="pt").input_ids[:, :512]


@torch.no_grad()
def run(ids, skip=()):
    """GPT-2 forward pass, written as a residual stream. Returns logits and per-layer measurements."""
    T = ids.shape[1]
    x = tr.wte(ids) + tr.wpe(torch.arange(T))
    stats = []
    for l, blk in enumerate(tr.h):
        if l in skip:
            continue                                              # delete the whole layer: the stream flows past it
        a = blk.attn(blk.ln_1(x))[0]
        x_mid = x + a                                             # attention writes into the stream
        m = blk.mlp(blk.ln_2(x_mid))
        stats.append((x.norm(dim=-1), a.norm(dim=-1), m.norm(dim=-1)))
        x = x_mid + m                                             # the MLP writes into the stream
    return gpt2.lm_head(tr.ln_f(x)), stats, x


logits, stats, final = run(ids)
loss0 = F.cross_entropy(logits[0, :-1], ids[0, 1:]).item()
ref = gpt2(ids, labels=ids).loss.item()
print(f"GPT-2 on {ids.shape[1]} tokens of Shakespeare: loss {loss0:.3f} (transformers: {ref:.3f})")

# ---------------------------------------------------------------- 1. size of the stream
print("\n1. size of the residual stream entering each layer (other tokens | first token)")
for l, (xn, an, mn) in enumerate(stats):
    print(f"   layer {l:>2}: {xn[0, 1:].mean():7.1f} | {xn[0, 0]:7.1f}")
print(f"   after the last layer: {final[0, 1:].norm(dim=-1).mean():.1f} | {final[0, 0].norm():.1f}")

# ---------------------------------------------------------------- 2. size of each update
print("\n2. each block's update, as a fraction of the stream it is added to (tokens 1+)")
for l, (xn, an, mn) in enumerate(stats):
    print(f"   layer {l:>2}: attention {(an / xn)[0, 1:].mean():.2f}   MLP {(mn / xn)[0, 1:].mean():.2f}")

# ---------------------------------------------------------------- 3. delete one layer at a time
print(f"\n3. delete one whole layer (baseline loss {loss0:.2f})")
for l in range(12):
    lg, _, _ = run(ids, skip={l})
    print(f"   without layer {l:>2}: loss {F.cross_entropy(lg[0, :-1], ids[0, 1:]).item():.2f}")

# ---------------------------------------------------------------- 4. with and without residual connections
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
V, D, H, L, T = len(chars), 128, 4, 8, 64


class Block(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.residual = residual
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def attn(self, h):
        B, t, _ = h.shape
        q, k, v = self.qkv(h).split(D, dim=2)
        q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        return self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, D))

    def forward(self, x):
        if self.residual:
            x = x + self.attn(self.ln1(x))                        # add to the stream
            return x + self.mlp(self.ln2(x))
        x = self.attn(self.ln1(x))                                # replace the stream
        return self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.Sequential(*[Block(residual) for _ in range(L)])
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


print(f"\n4. tiny GPT, {L} layers, 1,500 steps; validation loss during training")
for residual in (True, False):
    torch.manual_seed(1337)
    model = TinyGPT(residual)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    curve = []
    for step in range(1, 1501):
        x, y = batch(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (100, 250, 500, 1000, 1500):
            model.eval()
            curve.append(f"{step}: {val_loss(model):.2f}")
            model.train()
    name = "with residual connections" if residual else "without (x = block(x))"
    print(f"   {name:<27} " + "   ".join(curve))
freq = torch.bincount(train_data, minlength=V).float() / len(train_data)
print(f"   (uniform guessing: ln {V} = {math.log(V):.2f}; letter frequencies only: "
      f"{-freq[val_data].log().mean():.2f})")
