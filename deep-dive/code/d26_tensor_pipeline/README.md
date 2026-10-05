# Deep Dive, episode 26 · Tensor and Pipeline Parallelism

Two ways to split one model across workers, run with 2 local processes (`torch.distributed`, gloo).

```bash
pip install -r requirements.txt
python tensor_pipeline.py
```

PyTorch only (seconds on a CPU). It prints:

1. **tensor parallelism**: an MLP 128 → 512 → 128; each worker keeps half the columns of the first matrix and half the
   rows of the second (65,792 of 131,584 parameters); one all-reduce of the partial outputs matches the full MLP to
   1.2 × 10⁻⁶; 256 KiB exchanged per layer;
2. **pipeline parallelism**: 4 layers split 2 + 2; activations sent between stages (64 KiB per micro-batch); output
   identical to the full model; idle "bubble" (stages − 1) / (micro-batches + stages − 1): 2 stages: 50% (1), 20% (4),
   6% (16); 8 stages: 47% (8), 18% (32).
