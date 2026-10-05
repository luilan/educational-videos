"""Deep Dive, episode 9: Multi-Query and Grouped-Query Attention.

1. Qwen2.5-0.5B really uses grouped-query attention: 14 query heads share 2 key/value heads. Measure its KV cache.
2. Share GPT-2's keys and values after training (average them within groups), with no retraining: what breaks?
3. Train the same tiny GPT from scratch with 8, 2 and 1 key/value heads: loss, parameters and cache size.
"""
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast

HERE = Path(__file__).parent
torch.set_num_threads(8)
text = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()

# ---------------------------------------------------------------- 1. Qwen2.5-0.5B's KV cache
name = "Qwen/Qwen2.5-0.5B-Instruct"
qtok = AutoTokenizer.from_pretrained(name)
qwen = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).eval()
c = qwen.config
hd = c.hidden_size // c.num_attention_heads
print(f"1. {name}: {c.num_hidden_layers} layers, {c.num_attention_heads} query heads, "
      f"{c.num_key_value_heads} key/value heads, head size {hd}")
ids = qtok("The cat sat on the mat because it was tired.", return_tensors="pt").input_ids
with torch.no_grad():
    cache = qwen(ids, use_cache=True).past_key_values
k0, v0 = cache.layers[0].keys, cache.layers[0].values
print(f"   layer 0 cached keys: shape {tuple(k0.shape)}  (batch, kv heads, tokens, head size)")
per_token = 2 * c.num_hidden_layers * c.num_key_value_heads * hd * 2          # K and V, bf16 = 2 bytes
full = per_token * c.num_attention_heads // c.num_key_value_heads
measured = sum(l.keys.numel() + l.values.numel() for l in cache.layers) * 2 / ids.shape[1]
print(f"   KV cache per token: {per_token:,} bytes (measured {measured:,.0f}); "
      f"with one K/V per query head it would be {full:,}")
for n in (32_768, 131_072):
    print(f"   at {n:,} tokens: {per_token * n / 2**20:,.0f} MiB with GQA vs {full * n / 2**20:,.0f} MiB without")

# ---------------------------------------------------------------- 2. pool GPT-2's keys and values, no retraining
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
ids = tok(text[-6000:], return_tensors="pt").input_ids[:, :1024]


def pooled(model, groups):
    """Average the key and value projections of the heads inside each group (as when converting MHA to GQA)."""
    m = copy.deepcopy(model)
    per = 12 // groups
    for blk in m.transformer.h:
        w, b = blk.attn.c_attn.weight.data, blk.attn.c_attn.bias.data     # (768, 2304): q | k | v
        for start in (768, 1536):                                           # keys, then values
            for g in range(groups):
                cols = slice(start + g * per * 64, start + (g + 1) * per * 64)
                w[:, cols] = w[:, cols].view(768, per, 64).mean(1, keepdim=True).expand(768, per, 64).reshape(768, -1)
                b[cols] = b[cols].view(per, 64).mean(0, keepdim=True).expand(per, 64).reshape(-1)
    return m


print(f"\n2. GPT-2 on {ids.shape[1]} tokens of Shakespeare, keys and values shared after training (no retraining)")
with torch.no_grad():
    for groups in (12, 4, 2, 1):
        m = gpt2 if groups == 12 else pooled(gpt2, groups)
        loss = m(ids, labels=ids).loss.item()
        label = {12: "original (12 K/V heads)", 1: "1 K/V head (multi-query)"}.get(groups, f"{groups} K/V heads")
        print(f"   {label:<26} loss {loss:.2f}")

# ---------------------------------------------------------------- 3. train from scratch with 8, 2, 1 K/V heads
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([stoi[ch] for ch in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
V, D, H, L, T, HD = len(chars), 128, 8, 4, 64, 16


class Block(nn.Module):
    def __init__(self, kv_heads):
        super().__init__()
        self.kv = kv_heads
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.q = nn.Linear(D, H * HD)
        self.k, self.v = nn.Linear(D, kv_heads * HD), nn.Linear(D, kv_heads * HD)   # fewer key/value heads
        self.proj = nn.Linear(H * HD, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        B, t, _ = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, t, H, HD).transpose(1, 2)
        k = self.k(h).view(B, t, self.kv, HD).transpose(1, 2).repeat_interleave(H // self.kv, dim=1)
        v = self.v(h).view(B, t, self.kv, HD).transpose(1, 2).repeat_interleave(H // self.kv, dim=1)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(att.transpose(1, 2).reshape(B, t, H * HD))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, kv_heads):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.Sequential(*[Block(kv_heads) for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)

    def forward(self, idx):
        return self.head(self.ln(self.blocks(self.emb(idx) + self.pos(torch.arange(idx.shape[1])))))


def batch(d, bs=32, g=None):
    ix = torch.randint(len(d) - T - 1, (bs,), generator=g)
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


@torch.no_grad()
def val_loss(model):
    g = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(40):
        x, y = batch(val_data, g=g)
        losses.append(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)))
    return torch.stack(losses).mean().item()


print(f"\n3. the same tiny GPT ({H} query heads, {L} layers), trained 2,000 steps with different K/V heads")
for kv in (8, 2, 1):
    torch.manual_seed(1337)
    model = TinyGPT(kv)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2000):
        x, y = batch(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    cache = 2 * L * kv * HD
    label = {8: "8 K/V heads (multi-head)", 2: "2 K/V heads (grouped)", 1: "1 K/V head (multi-query)"}.get(
        kv, f"{kv} K/V heads")
    print(f"   {label:<26} val loss {val_loss(model):.3f}   {params:,} parameters   "
          f"cache {cache} numbers per token")
