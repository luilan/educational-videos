# Deep Dive, episode 19 · Adam and AdamW

Adam written by hand and checked against PyTorch, then compared with SGD on a tiny GPT, and AdamW's decoupled weight
decay compared with L2 in the gradient.

```bash
pip install -r requirements.txt
python adam.py
```

PyTorch only; trains eight 4-layer tiny GPTs (about 25 minutes on a CPU). It prints:

1. `MyAdam` vs `torch.optim.Adam` after 10 steps: largest difference 3.0 × 10⁻⁸;
2. first step size with the loss × 1 and × 1,000: SGD 0.0244 → 24.3951, Adam 0.0100 → 0.0100;
3. validation loss after 1,500 steps: SGD lr 0.1 2.476, SGD lr 1.0 NaN, SGD + momentum 1.940, Adam 1.708, MyAdam 1.708;
4. weight decay 0.1 at lr 0.003: no decay 1.666 (weight size 173.8), Adam + L2 3.307 (4.0), AdamW 1.682 (127.0);
5. GPT-2 small: 475 MiB of float32 weights, plus 949 MiB for Adam's m and v.
