"""Deep Dive, episode 27: Sharding the Optimizer State.

In data parallelism every worker stores the same Adam state. ZeRO-style sharding gives each worker only its own slice:
it updates that slice of the weights, then the workers share the updated weights (an all-gather).
4 local processes (torch.distributed, gloo): plain DDP + AdamW vs DDP + ZeroRedundancyOptimizer(AdamW).
1. Same training? Compare the losses over 100 steps.
2. Memory: Adam state stored per worker.
3. The arithmetic for a 7-billion-parameter model.
"""
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).parent
WORLD, PER_WORKER, STEPS = 4, 8, 100
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
    g = torch.Generator().manual_seed(step)
    ix = torch.randint(len(train_data) - T - 1, (WORLD * PER_WORKER,), generator=g)
    return torch.stack([train_data[i:i + T] for i in ix]), torch.stack([train_data[i + 1:i + T + 1] for i in ix])


def state_bytes(opt):
    """Bytes of optimizer state (Adam's m and v) held by this worker."""
    inner = opt.optim if isinstance(opt, ZeroRedundancyOptimizer) else opt
    return sum(t.numel() * t.element_size() for s in inner.state.values() for k, t in s.items()
               if torch.is_tensor(t) and t.dim() > 0)


def worker(rank, sharded, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29641" if sharded else "29642")
    torch.set_num_threads(2)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    torch.manual_seed(0)
    model = DDP(TinyGPT())
    if sharded:
        opt = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr=1e-3)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(STEPS):
        x, y = global_batch(step)
        mine = slice(rank * PER_WORKER, (rank + 1) * PER_WORKER)
        opt.zero_grad()
        loss = F.cross_entropy(model(x[mine]).reshape(-1, V), y[mine].reshape(-1))
        loss.backward()
        opt.step()                     # sharded: update my slice, then all-gather the weights
        full = loss.detach().clone()
        dist.all_reduce(full)
        losses.append(full.item() / WORLD)
    per_rank = torch.tensor([state_bytes(opt)], dtype=torch.float64)
    gathered = [torch.zeros(1, dtype=torch.float64) for _ in range(WORLD)]
    dist.all_gather(gathered, per_rank)
    if rank == 0:
        key = "sharded" if sharded else "plain"
        results[key] = (losses, [int(g.item()) for g in gathered])
        results["n_params"] = sum(p.numel() for p in model.parameters())
    dist.destroy_process_group()


if __name__ == "__main__":
    with mp.Manager() as manager:
        r = manager.dict()
        for sharded in (False, True):
            mp.spawn(worker, args=(sharded, r), nprocs=WORLD, join=True)
        r = dict(r)
    (lp, bp), (ls, bs), n = r["plain"], r["sharded"], r["n_params"]
    print(f"1. {WORLD} workers, {STEPS} steps: plain AdamW vs sharded (ZeroRedundancyOptimizer)")
    print(f"   largest loss difference {max(abs(a - b) for a, b in zip(lp, ls)):.1e}; "
          f"step {STEPS}: {lp[-1]:.4f} vs {ls[-1]:.4f}")
    print(f"\n2. {n:,} parameters = {n * 4 / 2**20:.2f} MiB of float32 weights")
    print(f"   Adam state per worker, plain:   " + ", ".join(f"{b / 2**20:.2f}" for b in bp) + " MiB")
    print(f"   Adam state per worker, sharded: " + ", ".join(f"{b / 2**20:.2f}" for b in bs) + " MiB")
    n7 = 7_000_000_000
    print(f"\n3. 7 billion parameters, Adam in float32, {WORLD * 16} workers")
    plain = 16 * n7
    zero1 = 8 * n7 + 8 * n7 / (WORLD * 16)
    zero3 = 16 * n7 / (WORLD * 16)
    print(f"   everything replicated:                      {plain / 1e9:6.1f} GB per worker")
    print(f"   Adam state sharded (weights, grads full):   {zero1 / 1e9:6.1f} GB per worker")
    print(f"   everything sharded (weights, grads, state): {zero3 / 1e9:6.1f} GB per worker")
