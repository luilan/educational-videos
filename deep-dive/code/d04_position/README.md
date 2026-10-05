# Deep Dive, episode 4 · Why Attention Needs Position

Proof with GPT-2 small's real weights (about 500 MB, CPU) that attention is blind to word order.

```bash
pip install -r requirements.txt
python position.py
```

It runs all 12 GPT-2 blocks on " dog bites man" and " man bites dog" and prints:

1. **position embeddings off, no mask**: the final vector for "dog" is identical in both sentences (cosine similarity
   1.000000; largest difference 0.0002, rounding);
2. **position embeddings on**: dog vs dog 0.96;
3. **no position embeddings, causal mask on**: 0.987, because the causal mask alone leaks some order;
4. GPT-2's learned position table (1,024 × 768) and how similar position 100 is to 101, 110, 150, 300 (0.999, 0.98,
   0.76, −0.23).
