"""Deep Dive, episode 8: The Attention Matrix, Entry by Entry.

Recompute one GPT-2 attention head by hand, with the real weights, and check every step against transformers:
queries, keys and values; one score as a 64-number dot product; the scaling by 8; the mask; the softmax; the weighted
sum of values; and the 12 heads joined back together. Then survey all 144 heads for two striking patterns.
"""
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_printoptions(precision=3, sci_mode=False)
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager").eval()
D, H = 768, 12
HD = D // H                                                        # 64 numbers per head

text = "The cat sat on the mat because it was tired"
ids = tok(text, return_tensors="pt").input_ids
words = [tok.decode(t) for t in ids[0]]
T = len(words)
with torch.no_grad():
    out = model(ids, output_hidden_states=True, output_attentions=True)
print("tokens:", words)

LAYER, HEAD = 4, 3                                                 # a head where 'it' looks at 'cat'
block = model.transformer.h[LAYER]
with torch.no_grad():
    x = block.ln_1(out.hidden_states[LAYER][0])                    # (T, 768): the input to this layer's attention
    q, k, v = block.attn.c_attn(x).split(D, dim=1)                 # each (T, 768)
    qh, kh, vh = (z.view(T, H, HD).transpose(0, 1) for z in (q, k, v))   # (12, T, 64)

# ---------------------------------------------------------------- 1. one entry
i, j = words.index(" it"), words.index(" cat")
qi, kj = qh[HEAD, i], kh[HEAD, j]
dot = (qi * kj).sum()
print(f"\n1. layer {LAYER}, head {HEAD}: score of query '{words[i]}' (position {i}) on key '{words[j]}' (position {j})")
print("   first 4 products of the 64:", (qi * kj)[:4])
print(f"   dot product {dot:.3f}, divided by sqrt(64) = 8: {dot / 8:.3f}")

# ---------------------------------------------------------------- 2. the whole matrix
scores = qh[HEAD] @ kh[HEAD].T / HD ** 0.5                         # (T, T)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
weights = scores.masked_fill(mask, float("-inf")).softmax(-1)
ref = out.attentions[LAYER][0, HEAD]
print(f"\n2. {T} x {T} = {T * T} scores; largest difference from transformers' weights: "
      f"{(weights - ref).abs().max().item():.1e}")
print(f"   row '{words[i]}':", "  ".join(f"{w!r} {p:.3f}" for w, p in zip(words[:i + 1], weights[i, :i + 1].tolist())))

# ---------------------------------------------------------------- 3. why divide by 8
raw = (qh[HEAD] @ kh[HEAD].T).masked_fill(mask, float("-inf")).softmax(-1)
print(f"\n3. largest weight in row '{words[i]}': scaled {weights[i].max():.3f}, unscaled {raw[i].max():.3f}")
print(f"   average largest weight over all rows: scaled {weights.max(-1).values.mean():.3f}, "
      f"unscaled {raw.max(-1).values.mean():.3f}")
g = torch.Generator().manual_seed(0)
a, b = torch.randn(10_000, HD, generator=g), torch.randn(10_000, HD, generator=g)
print(f"   random 64-number vectors: dot products have std {(a * b).sum(-1).std():.2f} (about sqrt(64) = 8)")

# ---------------------------------------------------------------- 4. weights x values, heads joined
head_out = weights @ vh[HEAD]                                      # (T, 64)
print(f"\n4. head output for '{words[i]}': {HD} numbers, a weighted average of the values; first 4:", head_out[i, :4])
all_w = (qh @ kh.transpose(1, 2) / HD ** 0.5).masked_fill(mask, float("-inf")).softmax(-1)
joined = (all_w @ vh).transpose(0, 1).reshape(T, D)               # 12 heads x 64 = 768
with torch.no_grad():
    mine = block.attn.c_proj(joined)
    theirs = block.attn(x[None])[0][0]
print(f"   12 heads x 64 = {joined.shape[1]} numbers, mixed by one 768 x 768 matrix; "
      f"difference from transformers: {(mine - theirs).abs().max().item():.1e}")

# ---------------------------------------------------------------- 5. two patterns across all 144 heads
long_ids = tok("The quick brown fox jumps over the lazy dog while the farmer watches from the old red barn "
               "and the sun sets slowly behind the hills", return_tensors="pt").input_ids
with torch.no_grad():
    att = torch.stack(model(long_ids, output_attentions=True).attentions)[:, 0]   # (12 layers, 12 heads, T, T)
Tl = long_ids.shape[1]
prev = torch.stack([att[..., r, r - 1] for r in range(1, Tl)], -1).mean(-1)   # weight on the previous token
first = att[..., 1:, 0].mean(-1)                                               # weight on token 0
pl, ph = divmod(prev.argmax().item(), H)
print(f"\n5. over {Tl} tokens: strongest previous-token head: layer {pl}, head {ph}, "
      f"{prev[pl, ph]:.2f} of its attention on the token just before")
print(f"   heads putting more than half their attention on the first token: {(first > 0.5).sum().item()} of 144 "
      f"(average over all heads {first.mean():.2f})")
print(f"   attention matrices per forward pass: 12 layers x 12 heads = 144; at 1,024 tokens, "
      f"{144 * 1024 * 1024:,} weights")
