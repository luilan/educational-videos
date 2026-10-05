"""How LLMs Work: Deep Dive, episode 5 — RoPE: rotating vectors to encode order.

1. RoPE from scratch: split a query (or key) into pairs of numbers, and rotate pair i by angle position × freq_i.
2. The key property: the attention score between a query at position m and a key at position n depends only on m − n.
3. The frequencies of a real model (Qwen2.5-0.5B: head size 64, base 1,000,000).
4. Check: our rotation matches the implementation in Hugging Face transformers.

    pip install -r requirements.txt
    python rope.py
Downloads only Qwen2.5-0.5B's config (a few KB).
"""
import math

import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, apply_rotary_pos_emb

cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
head_dim = cfg.hidden_size // cfg.num_attention_heads
base = cfg.rope_theta
print(f"Qwen2.5-0.5B: head size {head_dim}, RoPE base {base:,.0f}, trained context {cfg.max_position_embeddings:,}")

# 1. RoPE from scratch (pairs = (x[i], x[i + d/2]), the layout most implementations use)
freqs = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)      # one frequency per pair


def rope(x, position):
    half = x.shape[-1] // 2
    angle = position * freqs
    cos, sin = torch.cos(angle), torch.sin(angle)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


torch.manual_seed(0)
q = torch.randn(head_dim, dtype=torch.float64)
k = torch.randn(head_dim, dtype=torch.float64)

# Rotation keeps the length
print(f"\n|q| = {q.norm():.6f}, |rope(q, 7)| = {rope(q, 7).norm():.6f}")

# 2. Relative position: same distance, same score, wherever the pair sits
print("\nscore = rope(q, m) · rope(k, n)")
for m, n in [(5, 2), (105, 102), (10_005, 10_002), (5, 4), (105, 104)]:
    s = torch.dot(rope(q, m), rope(k, n)).item()
    print(f"  m = {m:>6}, n = {n:>6}, m - n = {m - n}: score {s:+.6f}")

# 3. The frequencies: fast pairs turn quickly, slow pairs barely move
wave = 2 * math.pi / freqs
print(f"\n{len(freqs)} rotating pairs; wavelength (tokens per full turn):")
for i in (0, 1, 8, 16, 24, 31):
    print(f"  pair {i:>2}: frequency {freqs[i]:.2e} rad/token, one turn every {wave[i]:,.1f} tokens")

# 4. Check against Hugging Face's Qwen2 implementation
rot = Qwen2RotaryEmbedding(config=cfg)
pos = torch.tensor([[7]])
cos, sin = rot(torch.zeros(1, 1, head_dim), pos)
q_hf, _ = apply_rotary_pos_emb(q.float()[None, None, None], k.float()[None, None, None], cos, sin)
print(f"\nmax difference from transformers' Qwen2 RoPE at position 7: "
      f"{(q_hf.flatten().double() - rope(q, 7)).abs().max():.2e}")
