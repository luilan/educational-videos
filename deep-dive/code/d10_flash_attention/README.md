# Deep Dive, episode 10 · FlashAttention: Same Math, Less Memory

Online softmax and tiled attention in plain PyTorch, checked against standard attention, then measured against PyTorch's
fused kernel.

```bash
pip install -r requirements.txt
python flash_attention.py
```

No downloads. The memory measurement reads `/proc/self/status`, so it needs Linux. It prints:

1. **online softmax** on 8 scores in two chunks: running max 3.0 → 4.0, sum 1.5853 → 1.9067, output 14.3052 → 36.2742,
   the same as the full softmax;
2. **tiled attention** (blocks of 64, causal) on one head of 1,024 tokens: largest difference from standard attention
   4.8 × 10⁻⁷; biggest score tensor 4,096 numbers (16 KiB) instead of 1,048,576 (4 MiB);
3. **12 heads, standard vs `F.scaled_dot_product_attention`** (extra memory, time; CPU, 8 threads):

| tokens | standard | fused |
|---|---|---|
| 1,024 | 99 MiB, 0.09 s | 8 MiB, 0.01 s |
| 2,048 | 386 MiB, 0.17 s | 4 MiB, 0.02 s |
| 4,096 | 1,557 MiB, 0.74 s | 11 MiB, 0.06 s |

Times and memory vary with the machine; the differences between outputs stay at float rounding (about 10⁻⁷).
