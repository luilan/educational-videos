# Deep Dive, episode 14 · LayerNorm vs RMSNorm

Both normalizations computed by hand and checked against real models, then compared in training.

```bash
pip install -r requirements.txt
python norms.py
```

Downloads GPT-2 small and Qwen2.5-0.5B-Instruct (about 1.5 GB) and trains six 8-layer tiny GPTs (about 25 minutes on a
CPU). It prints:

1. **LayerNorm** on “ sat” entering GPT-2 layer 6: mean 0.088, standard deviation 3.465; the hand version matches
   GPT-2's to 4.8 × 10⁻⁷; 768 + 768 learned numbers (scale and shift);
2. **RMSNorm** in Qwen2.5 layer 6: mean 0.022, root mean square 0.445; matches `Qwen2RMSNorm` to 4.8 × 10⁻⁷; 896 learned
   numbers (scale only);
3. **the stream vs what a block reads**, GPT-2: 5.2 → 253.3 across layers, always √768 = 27.7 after normalization;
4. **training**, validation loss at 100 / 500 / 1,500 steps:

| learning rate | LayerNorm | RMSNorm | no norm |
|---|---|---|---|
| 0.001 | 2.46 / 1.96 / 1.69 | 2.46 / 1.95 / 1.69 | 2.50 / 1.95 / 1.71 |
| 0.01 | 2.55 / 2.01 / 1.81 | 2.54 / 2.01 / 1.81 | NaN (diverged) |

and the time of PyTorch's built-in LayerNorm and RMSNorm on this machine (CPU: about 15 ms vs 46 ms for 4,096 × 4,096;
speed depends on the implementation and hardware).
