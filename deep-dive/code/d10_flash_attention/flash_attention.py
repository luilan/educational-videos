"""Deep Dive, episode 10: FlashAttention: Same Math, Less Memory.

1. Online softmax: a weighted average computed one chunk of scores at a time, with a running max and running sum.
2. Tiled attention built on it: never stores the T x T matrix, gives the same output as standard attention.
3. Memory and time at longer lengths: standard attention vs PyTorch's fused kernel (scaled_dot_product_attention).
"""
import math
import time

import torch
from torch.nn import functional as F

torch.manual_seed(0)
torch.set_num_threads(8)

# ---------------------------------------------------------------- 1. online softmax on one row
scores = torch.tensor([1.0, 3.0, 0.5, 2.0, 4.0, 1.5, 0.0, 2.5])
values = torch.arange(8.0) * 10                                   # one number per value, to keep it readable
print("1. one row of 8 scores, values 0, 10, ..., 70")
print(f"   standard: softmax, then weighted average = {(scores.softmax(0) * values).sum():.4f}")
m, l, acc = -math.inf, 0.0, 0.0                                   # running max, running sum, running output
for c in range(2):
    s, v = scores[4 * c:4 * c + 4], values[4 * c:4 * c + 4]
    m_new = max(m, s.max().item())
    fix = math.exp(m - m_new)                                     # rescale what we had under the new max
    p = torch.exp(s - m_new)
    l = l * fix + p.sum().item()
    acc = acc * fix + (p * v).sum().item()
    m = m_new
    print(f"   after chunk {c + 1}: max {m:.1f}, sum {l:.4f}, output so far {acc / l:.4f}")
print(f"   online result = {acc / l:.4f}")


# ---------------------------------------------------------------- 2. tiled attention
def standard(q, k, v):
    T = q.shape[-2]
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])         # the full T x T matrix
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), -math.inf)
    return s.softmax(-1) @ v


def tiled(q, k, v, B=64):
    """FlashAttention's algorithm: blocks of B queries against blocks of B keys, online softmax per row."""
    T, d = q.shape[-2:]
    out = torch.empty_like(q)
    biggest = 0
    for i in range(0, T, B):
        qi = q[..., i:i + B, :]
        m = torch.full(qi.shape[:-1], -math.inf)
        l = torch.zeros(qi.shape[:-1])
        acc = torch.zeros_like(qi)
        for j in range(0, i + B, B):                              # causal: skip blocks entirely in the future
            s = qi @ k[..., j:j + B, :].transpose(-2, -1) / math.sqrt(d)   # only a B x B block
            biggest = max(biggest, s.numel())
            if j + B > i:                                         # the diagonal block needs the mask
                rows, cols = torch.arange(i, i + B)[:, None], torch.arange(j, j + B)[None]
                s = s.masked_fill(cols > rows, -math.inf)
            m_new = torch.maximum(m, s.amax(-1))
            fix = torch.exp(m - m_new)
            p = torch.exp(s - m_new[..., None])
            l = l * fix + p.sum(-1)
            acc = acc * fix[..., None] + p @ v[..., j:j + B, :]
            m = m_new
        out[..., i:i + B, :] = acc / l[..., None]
    return out, biggest


T, d = 1024, 64
q, k, v = (torch.randn(T, d) for _ in range(3))
a = standard(q, k, v)
b, biggest = tiled(q, k, v)
print(f"\n2. one head, {T} tokens, head size {d}")
print(f"   largest difference, tiled vs standard: {(a - b).abs().max().item():.1e}")
print(f"   biggest score tensor: standard {T * T:,} numbers ({T * T * 4 / 2**20:.0f} MiB), "
      f"tiled {biggest:,} ({biggest * 4 / 2**10:.0f} KiB)")


# ---------------------------------------------------------------- 3. memory and time
def peak_rss_mib(fn):
    with open("/proc/self/clear_refs", "w") as f:                 # reset the peak-memory counter (Linux)
        f.write("5")
    base = rss("VmRSS")
    t0 = time.perf_counter()
    fn()
    return rss("VmHWM") - base, time.perf_counter() - t0


def rss(field):
    for line in open("/proc/self/status"):
        if line.startswith(field):
            return int(line.split()[1]) / 1024


H = 12
print(f"\n3. {H} heads, head size {d}, causal (extra memory above the inputs, and time)")
for T in (1024, 2048, 4096):
    q, k, v = (torch.randn(1, H, T, d) for _ in range(3))
    mem_f, t_f = peak_rss_mib(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))   # fused first, so it
    mem_s, t_s = peak_rss_mib(lambda: standard(q, k, v))                                     # can't reuse freed memory
    same = (standard(q, k, v) - F.scaled_dot_product_attention(q, k, v, is_causal=True)).abs().max().item()
    print(f"   T={T:>5}: standard {mem_s:6.0f} MiB {t_s:5.2f} s | fused {mem_f:5.0f} MiB {t_f:5.2f} s | "
          f"difference {same:.0e}")
for T in (16_384, 131_072):
    print(f"   T={T:,}: the full score matrices would need {H * T * T * 4 / 2**30:,.0f} GiB per layer")
