"""Deep Dive, episode 26: Tensor and Pipeline Parallelism.

Two ways to split ONE model across workers (2 local processes, torch.distributed "gloo"):
1. Tensor parallelism: split each matrix of an MLP. Worker k holds half the columns of the first matrix and half the
   rows of the second; one all-reduce (a sum) gives exactly the full MLP's output.
2. Pipeline parallelism: worker 0 runs the first half of the layers and sends its activations to worker 1, which runs
   the rest. Micro-batches keep both workers busy; the "bubble" is the idle time at the start and end.
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn import functional as F

D, HID, T, B = 128, 512, 64, 8


def setup(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(2)
    dist.init_process_group("gloo", rank=rank, world_size=2)


# ---------------------------------------------------------------- 1. tensor parallelism
def tensor_parallel(rank, results):
    setup(rank, 29621)
    torch.manual_seed(0)                                     # both workers build the same full MLP...
    w1, b1 = torch.randn(D, HID) / D ** 0.5, torch.randn(HID) * 0.1
    w2, b2 = torch.randn(HID, D) / HID ** 0.5, torch.randn(D) * 0.1
    x = torch.randn(B, T, D)
    half = slice(rank * HID // 2, (rank + 1) * HID // 2)    # ...but keep only their half
    my_w1, my_b1, my_w2 = w1[:, half].clone(), b1[half].clone(), w2[half, :].clone()
    h = F.gelu(x @ my_w1 + my_b1)                            # column-parallel: half of the hidden units
    partial = h @ my_w2                                      # row-parallel: a partial sum of the output
    dist.all_reduce(partial)                                 # sum the two partial outputs
    out = partial + b2
    if rank == 0:
        full = F.gelu(x @ w1 + b1) @ w2 + b2
        results["tp_diff"] = (out - full).abs().max().item()
        results["tp_params"] = (my_w1.numel() + my_b1.numel() + my_w2.numel(), w1.numel() + b1.numel() + w2.numel())
        results["tp_bytes"] = partial.numel() * 4
    dist.destroy_process_group()


# ---------------------------------------------------------------- 2. pipeline parallelism
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, HID), nn.GELU(), nn.Linear(HID, D))

    def forward(self, x):
        return x + self.mlp(self.ln(x))


def pipeline_parallel(rank, results):
    setup(rank, 29622)
    torch.manual_seed(0)
    blocks = nn.ModuleList([Block() for _ in range(4)])     # the full 4-layer model (same on both, for the check)
    mine = blocks[:2] if rank == 0 else blocks[2:]           # worker 0: layers 0-1, worker 1: layers 2-3
    torch.manual_seed(1)
    x = torch.randn(B, T, D)
    micro = x.chunk(4)                                       # 4 micro-batches of 2 sequences
    outs = []
    with torch.no_grad():
        for mb in micro:
            if rank == 0:
                h = mb
                for blk in mine:
                    h = blk(h)
                dist.send(h.contiguous(), dst=1)             # pass the activations down the pipeline
            else:
                h = torch.empty(mb.shape)
                dist.recv(h, src=0)
                for blk in mine:
                    h = blk(h)
                outs.append(h)
        if rank == 1:
            out = torch.cat(outs)
            full = x
            for blk in blocks:
                full = blk(full)
            results["pp_diff"] = (out - full).abs().max().item()
            results["pp_bytes"] = micro[0].numel() * 4
            results["pp_params"] = (sum(p.numel() for p in mine.parameters()), sum(p.numel() for p in blocks.parameters()))
    dist.destroy_process_group()


def bubble(stages, micro_batches):
    """Fraction of time a stage sits idle in a simple forward pipeline (GPipe-style)."""
    return (stages - 1) / (micro_batches + stages - 1)


if __name__ == "__main__":
    with mp.Manager() as manager:
        r = manager.dict()
        mp.spawn(tensor_parallel, args=(r,), nprocs=2, join=True)
        mp.spawn(pipeline_parallel, args=(r,), nprocs=2, join=True)
        r = dict(r)
    mine, total = r["tp_params"]
    print(f"1. tensor parallelism, MLP {D} -> {HID} -> {D} split over 2 workers")
    print(f"   each worker holds {mine:,} of the {total:,} parameters; output vs the full MLP: largest difference "
          f"{r['tp_diff']:.1e}")
    print(f"   communication: one all-reduce of the layer output, {r['tp_bytes'] / 1024:,.0f} KiB, in every layer")
    mine, total = r["pp_params"]
    print(f"\n2. pipeline parallelism, 4 layers split 2 + 2 over 2 workers, 4 micro-batches")
    print(f"   each worker holds {mine:,} of the {total:,} parameters; output vs the full model: largest difference "
          f"{r['pp_diff']:.1e}")
    print(f"   communication: one send of activations per micro-batch, {r['pp_bytes'] / 1024:,.0f} KiB, between stages only")
    print("   idle 'bubble' per stage:")
    for stages, m in ((2, 1), (2, 4), (2, 16), (8, 8), (8, 32)):
        print(f"     {stages} stages, {m:>2} micro-batches: {bubble(stages, m):.0%}")
