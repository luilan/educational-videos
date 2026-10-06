# Deep Dive, episode 29 · State-Space Models

Attention replaced by a recurrence with a fixed-size state, h ← a·h + (1 − a)·u, with a fixed or a selective
(input-dependent) decay.

```bash
pip install -r requirements.txt
python state_space.py
```

PyTorch only (about 10 minutes on a CPU). It prints:

1. **validation loss**, 4 layers, 2,000 steps: attention 1.644 (818,241 parameters); recurrence with fixed decay 1.648
   (744,513); recurrence with selective decay 1.591 (810,049);
2. **memory per layer** while generating: attention KV cache 0.98 MiB after 1,000 tokens, 97.66 MiB after 100,000;
   recurrent state 0.50 KiB;
3. **time per new token per layer** (1 thread): attention 37.9 µs (1,000 tokens of context), 312.6 µs (10,000),
   11,334 µs (100,000); recurrence about 6.4 µs. Times depend on the machine.
