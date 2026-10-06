# Deep Dive, episode 42 · Induction Heads

GPT-2 small on 8 sequences of 50 random tokens, each repeated twice: copying loss, the induction score of every head,
and ablations with `head_mask`.

```bash
pip install -r requirements.txt
python induction.py
```

Results (seeded):

1. Loss on the first copy 12.79, on the second copy 0.25.
2. Induction score (attention to the token after the earlier occurrence): 5.5 0.94, 7.10 0.92, 6.9 0.91, 5.1 0.91,
   7.2 0.85; 15 of 144 heads above 0.3.
3. Second-copy loss with the top 2 / 4 / 6 induction heads removed: 0.35 / 1.78 / 5.81; with 2 / 4 / 6 random heads
   removed (mean of 5): 0.29 / 0.38 / 0.66.
