# Deep Dive, episode 7 · Causal Masking, in Detail

How the causal mask works, proved on GPT-2 and by training a tiny GPT with and without it.

```bash
pip install -r requirements.txt
python causal_mask.py
```

It downloads GPT-2 small (about 500 MB) and trains two 818,241-parameter models on Tiny Shakespeare (a few minutes on a
CPU). It prints:

1. **the mask by hand**: 4 tokens' scores with −∞ above the diagonal, and the softmax weights (each row sums to 1);
   with no mask, token 1 would give 0.631 of its attention to token 4;
2. **GPT-2**: change “mat” to “moon”, and every earlier position's output differs by exactly 0.0; layer 0, head 0's
   attention weights are 0 above the diagonal;
3. **with and without the mask** (1,500 steps each):

| | training loss | honest loss (past only) |
|---|---|---|
| mask on | 1.52 | 1.73 |
| mask off | 0.04 | 7.36 (uniform guessing: 4.17) |

plus a text sample from each model. The unmasked one learned to copy the next character, not predict it.
