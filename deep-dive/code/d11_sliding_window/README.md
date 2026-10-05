# Deep Dive, episode 11 · Sliding Windows and Sparse Attention

Sliding-window attention: its mask, how far stacked layers can reach, and what it costs on two different tasks.

```bash
pip install -r requirements.txt
python sliding_window.py
```

PyTorch only; it trains six tiny GPTs (4 layers, 1,500 steps each), about 20 minutes on a CPU. It prints:

1. **the mask**: a window of 4 on 8 tokens (a band along the diagonal); a window of 4,096 at 131,072 tokens keeps 6.2%
   of the pairs;
2. **reach**, measured with gradients, window 16: 15, 30, 45, 60 tokens after 1–4 layers;
3. **Shakespeare** (256-character texts), validation loss: full attention 1.780, window 64 1.750, window 16 1.693;
4. **copy task** (128 random symbols, then the same again), loss on the second half: full attention 0.000, window 64
   2.775, window 16 2.775 (guessing: ln 16 = 2.77).

Windows help on local patterns and fail on exact long-range recall, even when their theoretical reach (252 tokens for
window 64) covers the distance.
