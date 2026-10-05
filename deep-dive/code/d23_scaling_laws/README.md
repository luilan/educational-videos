# Deep Dive, episode 23 · Scaling Laws

Power laws for loss vs model size and loss vs data, measured on tiny GPTs.

```bash
pip install -r requirements.txt
python scaling_laws.py
```

PyTorch only (about 45 minutes on a CPU). It prints:

1. **six model sizes**, 2,000 steps on 4,096,000 tokens each (AdamW, lr 0.002):

| width × layers | parameters (excluding embeddings) | val loss |
|---|---|---|
| 16 × 2 | 7,697 | 2.280 |
| 32 × 2 | 27,617 | 2.070 |
| 48 × 3 | 88,097 | 1.879 |
| 64 × 3 | 154,305 | 1.774 |
| 96 × 4 | 453,857 | 1.684 |
| 128 × 6 | 1,198,273 | 1.612 |

   fit: loss ≈ 4.24 × N^−0.071 (the largest model sits above the line: 1.577 predicted);
2. **data**: the 96 × 4 model at 256,000 … 8,192,000 tokens: 2.406, 2.165, 1.987, 1.798, 1.684, 1.586;
   fit: loss ≈ 10.68 × tokens^−0.121.
