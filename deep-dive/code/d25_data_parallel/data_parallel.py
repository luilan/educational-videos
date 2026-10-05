"""Deep Dive, episode 25: Data Parallelism.

Real distributed training on one machine: 4 worker processes talk through PyTorch's "gloo" backend.
1. Every worker holds a full copy of the model and takes a quarter of the batch.
2. After backward, DistributedDataParallel averages the gradients across workers (an all-reduce).
3. Check: 4 workers x 8 sequences give the same gradients, the same weights after a step, and the same losses for
   200 steps as one process with batches of 32.
4. What it costs: the bytes every step must exchange.
"""
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).parent
WORLD, PER_WORKER, STEPS = 4, 8, 200
text = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
train_data = data[:int(0.9 * len(data))]
V, D, H, L, T = len(chars), 128, 4, 4, 64


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        B, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.Sequential(*[Block() for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)

    def forward(self, idx):
        return self.head(self.ln(self.blocks(self.emb(idx) + self.pos(torch.arange(idx.shape[1])))))


def global_batch(step):
    """The same 32 sequences for a given step, whoever asks."""
    g = torch.Generator().manual_seed(step)
    ix = torch.randint(len(train_data) - T - 1, (WORLD * PER_WORKER,), generator=g)
    return torch.stack([train_data[i:i + T] for i in ix]), torch.stack([train_data[i + 1:i + T + 1] for i in ix])


def lm_loss(model, x, y):
    return F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))


def single_process():
    torch.manual_seed(0)
    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses, first_grad = [], None
    for step in range(STEPS):
        x, y = global_batch(step)
        opt.zero_grad()
        loss = lm_loss(model, x, y)
        loss.backward()
        if step == 0:
            first_grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
            after_one = None
        opt.step()
        if step == 0:
            after_one = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
        losses.append(loss.item())
    return first_grad, after_one, losses


def worker(rank, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29511")
    torch.set_num_threads(2)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    torch.manual_seed(0)                                        # every worker starts from the same weights
    model = DDP(TinyGPT())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(STEPS):
        x, y = global_batch(step)
        mine = slice(rank * PER_WORKER, (rank + 1) * PER_WORKER)  # this worker's quarter of the batch
        opt.zero_grad()
        loss = lm_loss(model, x[mine], y[mine])
        loss.backward()                                         # DDP all-reduces (averages) the gradients here
        if step == 0 and rank == 0:
            results["grad"] = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
        opt.step()
        if step == 0 and rank == 0:
            results["after_one"] = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
        full = loss.detach().clone()
        dist.all_reduce(full)                                   # average the workers' losses, for reporting only
        losses.append(full.item() / WORLD)
    if rank == 0:
        results["losses"] = losses
        results["n_params"] = sum(p.numel() for p in model.parameters())
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_num_threads(8)
    g1, w1, l1 = single_process()
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(worker, args=(results,), nprocs=WORLD, join=True)
        g4, w4, l4 = results["grad"], results["after_one"], results["losses"]
        n_params = results["n_params"]
    print(f"1-2. {WORLD} workers x {PER_WORKER} sequences vs 1 process x {WORLD * PER_WORKER} sequences")
    print(f"   step 1 gradient: largest difference {(g1 - g4).abs().max().item():.1e} (gradient size {g1.norm():.3f})")
    print(f"   weights after step 1: largest difference {(w1 - w4).abs().max().item():.1e}")
    print(f"3. losses over {STEPS} steps: largest difference {max(abs(a - b) for a, b in zip(l1, l4)):.1e}; "
          f"step 1: {l1[0]:.4f} vs {l4[0]:.4f}; step {STEPS}: {l1[-1]:.4f} vs {l4[-1]:.4f}")
    size = n_params * 4
    ring = 2 * (WORLD - 1) / WORLD * size
    print(f"4. every step, each worker exchanges the gradients of {n_params:,} parameters = {size / 2**20:.1f} MiB "
          f"(a ring all-reduce sends about 2 x (N-1)/N of that: {ring / 2**20:.1f} MiB per worker)")
    for name, n in (("GPT-2 small", 124_439_808), ("a 7-billion-parameter model", 7_000_000_000)):
        print(f"   {name}: {n * 4 / 2**30:,.1f} GiB of float32 gradients per step")
