# Deep Dive, episode 16 · Pre-Norm, Post-Norm, and Stability

The same 12-layer tiny GPT with the normalization before each block (pre-norm: GPT-2, Llama, Qwen) or after it
(post-norm: the original Transformer, BERT).

```bash
pip install -r requirements.txt
python pre_post_norm.py
```

PyTorch only; trains ten 12-layer models (about an hour on a CPU). It prints:

1. **gradients at initialization**: similar at every layer in both (last / first = 0.9);
2. **validation loss** at 100 / 500 / 1,500 steps:

| learning rate | warmup | pre-norm | post-norm |
|---|---|---|---|
| 0.001 | none | 2.47 / 1.95 / 1.67 | 2.51 / 1.94 / 1.68 |
| 0.001 | 300 steps | 2.72 / 2.03 / 1.69 | 2.69 / 2.01 / 1.68 |
| 0.003 | none | 2.41 / 1.91 / 1.67 | **3.37 / 3.37 / 3.36** (stuck) |
| 0.003 | 300 steps | 2.56 / 1.91 / 1.66 | 2.55 / 1.92 / 1.72 |

3. **the first 100 steps** at 0.003 without warmup: post-norm's training loss stalls at 3.38 by step 10; pre-norm keeps
   falling (2.40 at step 100).

`train(pre=False, lr=3e-3, warmup=50)` shows that a short warmup is already enough (1.75 after 1,500 steps).
