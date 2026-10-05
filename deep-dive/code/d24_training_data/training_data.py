"""Deep Dive, episode 24: Where Training Data Comes From.

1. How big is our data, in tokens, and how much of it repeats?
2. Quality: mix junk (random characters) into the training data and measure the loss on clean text.
3. Duplication: repeat one passage often during training. The model memorizes it word for word.
"""
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2TokenizerFast

HERE = Path(__file__).parent
torch.set_num_threads(8)
text = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()

# ---------------------------------------------------------------- 1. size and repetition
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tok.model_max_length = 10 ** 7
n_tokens = len(tok(text).input_ids)
lines = [l.strip() for l in text.split("\n") if l.strip()]
counts = Counter(lines)
dup_lines = sum(c for c in counts.values() if c > 1)
print(f"1. Tiny Shakespeare: {len(text):,} characters, {n_tokens:,} GPT-2 tokens")
print(f"   {len(lines):,} non-empty lines; {dup_lines:,} of them ({dup_lines / len(lines):.0%}) appear more than once")
print("   most repeated lines:", ", ".join(f"{l!r} ×{c}" for l, c in counts.most_common(4)))

# ---------------------------------------------------------------- the model
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


def windows(d, bs, g=None):
    ix = torch.randint(len(d) - T - 1, (bs,), generator=g)
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


@torch.no_grad()
def val_loss(model, d=val_data):
    g = torch.Generator().manual_seed(0)
    return sum(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()
               for x, y in (windows(d, 32, g) for _ in range(40))) / 40


def train(make_batch, steps=2000):
    torch.manual_seed(1337)
    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(steps):
        x, y = make_batch(step)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------- 2. junk in the data
print("\n2. a share of each training batch replaced by junk (random characters); loss on clean validation text")
for share in (0.0, 0.25, 0.5):
    def make_batch(step, share=share):
        x, y = windows(train_data, 32)
        k = int(32 * share)
        if k:
            junk = torch.randint(V, (k, T + 1))
            x[:k], y[:k] = junk[:, :-1], junk[:, 1:]
        return x, y
    model = train(make_batch)
    print(f"   {share:.0%} junk: clean validation loss {val_loss(model):.3f}")

# ---------------------------------------------------------------- 3. duplication and memorization
passage = val_data[5000:5000 + T + 1]                     # 65 characters the model would normally never train on
print(f"\n3. a validation passage, repeated in training: {''.join(chars[i] for i in passage[:-1])!r}")
for every in (0, 10, 2, 1):
    def make_batch(step, every=every):
        x, y = windows(train_data, 32)
        if every and step % every == 0:                   # one copy of the passage, every `every` steps
            x[0], y[0] = passage[:-1], passage[1:]
        return x, y
    model = train(make_batch)
    with torch.no_grad():
        logits = model(passage[None, :-1])[0]
        pl = F.cross_entropy(logits, passage[1:]).item()
        acc = (logits.argmax(-1) == passage[1:]).float().mean().item()   # next character right, given the true text
        idx = passage[None, :20].clone()
        for _ in range(45):
            idx = torch.cat([idx, model(idx)[:, -1].argmax(-1, keepdim=True)], 1)
    out = "".join(chars[i] for i in idx[0, 20:])
    want = "".join(chars[i] for i in passage[20:65])
    same = sum(a == b for a, b in zip(out, want))
    label = "never" if not every else f"every {every} steps ({2000 // every} copies)"
    print(f"   {label:<26} passage: loss {pl:.3f}, next character right {acc:.0%}; other text {val_loss(model):.3f}; "
          f"from its first 20 characters it writes {same}/45 right: {out!r}")
