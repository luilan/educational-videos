"""Deep Dive, episode 7: Causal Masking, in Detail.

1. The mask by hand: add -inf above the diagonal, then softmax each row.
2. GPT-2 (real weights): change the last word, and every earlier position's output stays exactly the same.
3. Why it matters: train the same tiny model with and without the mask. Without it, the model "cheats" by reading the
   next character, and its honest loss (predicting from the past only) is terrible.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = Path(__file__).parent
torch.manual_seed(1337)
torch.set_num_threads(8)

# ---------------------------------------------------------------- 1. the mask by hand
print("1. the mask by hand (4 tokens)")
scores = torch.tensor([[2.0, 1.0, 0.5, 3.0],
                       [1.0, 2.0, 1.5, 0.0],
                       [0.5, 1.0, 2.0, 1.0],
                       [1.0, 0.5, 1.0, 2.0]])
mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)      # True above the diagonal = the future
masked = scores.masked_fill(mask, float("-inf"))
weights = masked.softmax(dim=-1)
print("scores + mask:\n", masked)
print("weights (each row sums to 1):\n", weights.round(decimals=3))
print("without the mask, row 0 would be:", scores[0].softmax(-1).round(decimals=3))

# ---------------------------------------------------------------- 2. GPT-2: the future cannot change the past
print("\n2. GPT-2: change the last word, compare earlier positions")
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager").eval()
a = tok("The cat sat on the mat", return_tensors="pt").input_ids
b = tok("The cat sat on the moon", return_tensors="pt").input_ids
with torch.no_grad():
    ha = gpt2(a, output_hidden_states=True).hidden_states[-1][0]
    hb = gpt2(b, output_hidden_states=True).hidden_states[-1][0]
    att = gpt2(a, output_attentions=True).attentions[0][0, 0]           # layer 0, head 0
print("tokens:", [tok.decode(t) for t in a[0]])
print("largest difference at positions 0-4:", (ha[:-1] - hb[:-1]).abs().max().item())
print("largest difference at the last position:", (ha[-1] - hb[-1]).abs().max().item())
print("layer 0, head 0 attention weights:\n", att.round(decimals=2))

# ---------------------------------------------------------------- 3. train with and without the mask
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

    def forward(self, x, causal):
        B, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        att = F.scaled_dot_product_attention(q, k, v, is_causal=causal)   # the only difference
        x = x + self.proj(att.transpose(1, 2).reshape(B, t, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.causal = causal
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        for blk in self.blocks:
            x = blk(x, self.causal)
        return self.head(self.ln(x))


def batch(d, bs=32, g=None):
    ix = torch.randint(len(d) - T - 1, (bs,), generator=g)
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


def train(model, steps=1500):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(steps):
        x, y = batch(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


@torch.no_grad()
def honest_loss(model, n_seq=8):
    """Predict each next character from the past only: feed every prefix separately, keep its last output."""
    g = torch.Generator().manual_seed(0)
    x, y = batch(val_data, bs=n_seq, g=g)
    losses = [F.cross_entropy(model(x[:, :t + 1])[:, -1], y[:, t]) for t in range(T)]
    return torch.stack(losses).mean().item()


@torch.no_grad()
def generate(model, n=120):
    g = torch.Generator().manual_seed(1)
    idx = torch.tensor([[stoi["\n"]]])
    for _ in range(n):
        p = model(idx[:, -T:])[:, -1].softmax(-1)
        idx = torch.cat([idx, torch.multinomial(p, 1, generator=g)], dim=1)
    return "".join(chars[i] for i in idx[0, 1:])


print(f"\n3. the same tiny GPT trained twice ({sum(p.numel() for p in TinyGPT(True).parameters()):,} parameters, "
      f"1,500 steps, {T}-character texts)")
results = {}
for causal in (True, False):
    torch.manual_seed(1337)
    model = TinyGPT(causal)
    tl = train(model)
    model.eval()
    results[causal] = (tl, honest_loss(model), generate(model))
    name = "with the mask" if causal else "without the mask"
    print(f"\n{name}: training loss {tl:.2f}, honest loss {results[causal][1]:.2f}")
    print("sample:", repr(results[causal][2]))
print(f"\none {T}-character text gives {T} training examples in a single pass, thanks to the mask")
