# Deep Dive, episode 33 · Quantization, Deeper

Why naive 4-bit rounding hurts, and what fixes it: per-group scales, the NF4 grid, and per-token scales for activations.

```bash
pip install -r requirements.txt
python quantization.py
```

Downloads Qwen2.5-0.5B (about 1 GB) and reports mean loss on 4 × 512 tokens of `eval_text.txt` (a Tiny Shakespeare
slice). Results on CPU, float32 (deterministic):

1. Weight outliers: one row of layer 0's down_proj has median |w| 0.0125 and largest 0.170 (14x); 4 bits with one scale
   round 49% of it to 0.
2. 4-bit weights: full precision 3.393; one scale per row 4.186; groups of 128 / 64 / 32: 3.756 / 3.622 / 3.570
   (4.12 / 4.25 / 4.50 bits per weight). 3 bits: per row 12.478, groups of 32 4.518.
3. Groups of 64: evenly spaced levels 3.622, NF4 3.533 (+0.229 vs +0.140 over full precision).
4. 8-bit inputs to every linear layer: the first token is the largest input in 84 of 168 layers (up to 1,808 vs 1.86,
   969x). One scale per tensor 4.709; first token kept in 16 bits 3.475; one scale per token 3.415.
