# Deep Dive, episode 21 · Batch Size and Gradient Accumulation

Gradient accumulation checked against a full batch, gradient noise by batch size, and the same data trained at three
batch sizes.

```bash
pip install -r requirements.txt
python batch_size.py
```

PyTorch only (about 15 minutes on a CPU). It prints:

1. **accumulation**: one batch of 32 vs 4 micro-batches of 8 (each loss ÷ 4): largest gradient difference 1.5 × 10⁻⁸;
2. **gradient noise** after 300 steps, cosine similarity to the gradient of 4,096 sequences: batch 1 0.17, 4 0.19,
   16 0.50, 64 0.72, 256 0.90;
3. **48,000 sequences, AdamW**:

| batch | lr | steps | val loss | time per sequence (8-thread CPU) |
|---|---|---|---|---|
| 8 | 0.001 | 6,000 | 1.660 | 2.82 ms |
| 32 | 0.001 | 1,500 | 1.709 | 1.56 ms |
| 128 | 0.001 | 375 | 1.930 | 1.68 ms |
| 128 | 0.002 | 375 | 1.811 | 1.67 ms |

Times depend on the machine.
