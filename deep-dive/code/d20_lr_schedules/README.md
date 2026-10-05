# Deep Dive, episode 20 · Learning-Rate Warmup and Schedules

The same tiny GPT trained for 3,000 steps with five learning-rate schedules.

```bash
pip install -r requirements.txt
python lr_schedules.py
```

PyTorch only; five 4-layer tiny GPTs (about 25 minutes on a CPU). AdamW, peak 0.003, 200 warmup steps, decays ending at
10% of the peak. Validation loss at steps 500 / 1,000 / 1,500 / 2,000 / 2,500 / 3,000:

| schedule | loss |
|---|---|
| constant 0.001 | 2.000 / 1.810 / 1.709 / 1.644 / 1.630 / 1.605 |
| constant 0.003 | 1.899 / 1.731 / 1.661 / 1.620 / 1.603 / 1.598 |
| warmup + cosine | 1.915 / 1.733 / 1.640 / 1.588 / 1.564 / 1.551 |
| warmup + linear | 1.912 / 1.728 / 1.643 / 1.600 / 1.579 / 1.554 |
| warmup-stable-decay | 1.917 / 1.740 / 1.670 / 1.623 / 1.606 / 1.542 (1.606 at step 2,400, when the decay starts) |
