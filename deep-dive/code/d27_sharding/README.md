# Deep Dive, episode 27 · Sharding the Optimizer State

ZeRO-style optimizer sharding with PyTorch's `ZeroRedundancyOptimizer`, on 4 local processes (`torch.distributed`, gloo).

```bash
pip install -r requirements.txt
python sharding.py
```

PyTorch only (a few minutes on a CPU). It prints:

1. plain AdamW vs `ZeroRedundancyOptimizer(AdamW)`, 4 workers, 100 steps: largest loss difference 0.0 (2.4467 in both);
2. Adam state per worker for the 818,241-parameter tiny GPT: 6.24 MiB plain, 1.56 MiB sharded;
3. per-worker memory for 7 billion parameters (Adam, float32) on 64 workers: 112.0 GB copied, 56.9 GB with Adam state
   sharded, 1.8 GB with weights, gradients and state sharded.
