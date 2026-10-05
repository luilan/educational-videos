# Deep Dive, episode 5 · RoPE: Rotating Vectors to Encode Order

Rotary position embedding from scratch, checked against Hugging Face's Qwen2 implementation.

```bash
pip install -r requirements.txt
python rope.py
```

Only Qwen2.5-0.5B's config is downloaded (head size 64, RoPE base 1,000,000). It prints:

1. that rotation keeps a vector's length;
2. **the key property**: `rope(q, m) · rope(k, n)` depends only on `m − n` (+5.924725 for (5, 2), (105, 102) and
   (10,005, 10,002));
3. the 32 rotation speeds: a full turn every 6.3 tokens for the fastest pair, every 4,080,185 for the slowest;
4. the difference from `transformers`' Qwen2 rotary embedding at position 7: about 2 × 10⁻⁷ (float rounding).
