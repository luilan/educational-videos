# Deep Dive, episode 9 · Multi-Query and Grouped-Query Attention

Sharing keys and values between attention heads: measured on a real model, bolted onto GPT-2, and trained in from
scratch.

```bash
pip install -r requirements.txt
python gqa.py
```

Downloads Qwen2.5-0.5B-Instruct (about 1 GB) and GPT-2 small (about 500 MB), and trains three tiny models (a few minutes
each on a CPU). It prints:

1. **Qwen2.5-0.5B**: 24 layers, 14 query heads, 2 key/value heads; cached keys of shape (1, 2, tokens, 64); 12,288
   bytes of cache per token (bfloat16), against 86,016 with one K/V head per query head: 384 MiB vs 2,688 MiB at 32,768
   tokens, 1,536 MiB vs 10,752 MiB at 131,072;
2. **GPT-2, keys and values averaged per group, no retraining** (loss on 1,024 tokens of Shakespeare): 3.80 original,
   6.65 with 4 K/V heads, 6.41 with 2, 6.52 with 1;
3. **a tiny GPT trained from scratch** (8 query heads, 2,000 steps):

| K/V heads | val loss | parameters | cache per token |
|---|---|---|---|
| 8 (multi-head) | 1.673 | 818,241 | 1,024 numbers |
| 2 (grouped) | 1.685 | 719,169 | 256 |
| 1 (multi-query) | 1.704 | 702,657 | 128 |
