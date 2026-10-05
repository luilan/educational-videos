# Deep Dive, episode 25 · Data Parallelism

Real distributed training on one machine: 4 processes with `torch.distributed` (gloo) and `DistributedDataParallel`,
checked against a single process.

```bash
pip install -r requirements.txt
python data_parallel.py
```

PyTorch only; starts 4 local processes (a few minutes on a CPU). It prints:

1. 4 workers × 8 sequences vs 1 process × 32: first gradient within 1.5 × 10⁻⁸; weights after one AdamW step within
   6.9 × 10⁻⁶;
2. losses over 200 steps within 4.8 × 10⁻⁷ (4.3335 → 2.3006 in both runs);
3. the gradients exchanged every step: 3.1 MiB for the 818,241-parameter tiny model (4.7 MiB sent per worker in a ring
   all-reduce), 0.5 GiB for GPT-2 small, 26.1 GiB for 7 billion parameters.

Set `WORLD, PER_WORKER, STEPS = 2, 16, 50` to try 2 workers.
