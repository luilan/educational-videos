"""Deep Dive, episode 12: Attention Sinks.

1. Many GPT-2 heads put most of their attention on the very first token, whatever that token is.
2. The first token's values are tiny: attending to it is close to "doing nothing" (softmax must sum to 1).
3. Why it matters: a sliding-window cache that drops the first tokens breaks; keeping 4 "sink" tokens fixes it.
GPT-2's forward pass is written out by hand so any attention mask can be used; it matches transformers exactly.
"""
import math
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = Path(__file__).parent
torch.set_num_threads(8)
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
tr = model.transformer
H, D = 12, 768


@torch.no_grad()
def forward(ids, allowed):
    """GPT-2 by hand. allowed: (T, T) bool, True where query i may attend to key j. Returns logits, weights, values."""
    T = ids.shape[1]
    x = tr.wte(ids) + tr.wpe(torch.arange(T))
    weights, values = [], []
    for blk in tr.h:
        q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(D, dim=2)
        q, k, v = (z.view(1, T, H, 64).transpose(1, 2) for z in (q, k, v))
        s = (q @ k.transpose(-2, -1) / 8).masked_fill(~allowed, -math.inf)
        w = s.softmax(-1)
        weights.append(w[0])
        values.append(v[0])
        x = x + blk.attn.c_proj((w @ v).transpose(1, 2).reshape(1, T, D))
        x = x + blk.mlp(blk.ln_2(x))
    return model.lm_head(tr.ln_f(x)), torch.stack(weights), torch.stack(values)   # (L, H, T, T), (L, H, T, 64)


def causal(T):
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


tok.model_max_length = 10**6
text = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()
ids = tok(text[100_000:110_000], return_tensors="pt").input_ids[:, :1024]
T = ids.shape[1]

# ---------------------------------------------------------------- check against transformers
logits, w, v = forward(ids[:, :64], causal(64))
ref = model(ids[:, :64]).logits
print(f"hand-written GPT-2 vs transformers: largest logit difference {(logits - ref).abs().max().item():.1e}")

# ---------------------------------------------------------------- 1. where the attention goes
_, w, v = forward(ids[:, :256], causal(256))
first = w[:, :, 16:, 0].mean(-1)                                  # attention on token 0, from queries 16-255
print(f"\n1. {first.numel()} heads: average attention on the first token: {first.mean():.3f}; "
      f"heads with more than half: {(first > 0.5).sum().item()}")
print("   per layer:", " ".join(f"{x:.2f}" for x in first.mean(1).tolist()))
for start in ("\n", " zebra", " The", " ,"):
    alt = torch.cat([tok(start, return_tensors="pt").input_ids[:, :1], ids[:, 1:256]], 1)
    _, wa, _ = forward(alt, causal(256))
    print(f"   first token {start!r:>9}: average attention on it {wa[:, :, 16:, 0].mean():.3f}")
mid = ids[:, 100:356]                                             # the same text, starting mid-sentence
_, wm, _ = forward(mid, causal(256))
print(f"   text cut mid-sentence: average attention on its first token {wm[:, :, 16:, 0].mean():.3f}")

# ---------------------------------------------------------------- 2. the sink's values are small
vn = v.norm(dim=-1)                                               # (L, H, T)
print(f"\n2. size of value vectors (average over layers 2-11 and heads): first token "
      f"{vn[2:, :, 0].mean():.2f}, other tokens {vn[2:, :, 1:].mean():.2f}")

# ---------------------------------------------------------------- 3. sliding-window cache, with and without sinks
W = 256
idx = torch.arange(T)
window = (idx[None] <= idx[:, None]) & (idx[:, None] - idx[None] < W)
sinks = window | ((idx[None] < 4) & (idx[None] <= idx[:, None]))  # also always see tokens 0-3
targets = ids[0, 1:]
print(f"\n3. GPT-2 on {T} tokens of Shakespeare; loss on positions 512-1022 (each sees at most {W} recent tokens)")
for name, mask in (("full attention", causal(T)), (f"window {W}", window), (f"window {W} + 4 sink tokens", sinks)):
    lg, _, _ = forward(ids, mask)
    loss = torch.nn.functional.cross_entropy(lg[0, 512:-1], targets[512:])
    print(f"   {name:<28} loss {loss:.2f}  (perplexity {loss.exp():.0f})")
