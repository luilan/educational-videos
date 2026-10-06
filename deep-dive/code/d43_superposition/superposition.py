"""Deep Dive, episode 43: Superposition.

The toy model from "Toy Models of Superposition" (Elhage et al., 2022): n features are squeezed into m < n dimensions
and read back out, x' = ReLU(W^T W x + b). Each feature is zero with probability S (the sparsity), otherwise uniform in
[0, 1]. In part 1 feature i has importance 0.8^i in the loss; in part 2 all features matter equally.
1. 5 features, 2 dimensions: what W looks like as features get sparser.
2. 100 features, 20 dimensions: how many features the model represents, and how much they interfere.
"""
import json
import sys

import torch

torch.set_num_threads(4)


def train(n, m, S, steps=6000, seed=0, decay=0.8):
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(m, n) * 0.1)
    b = torch.nn.Parameter(torch.zeros(n))
    imp = decay ** torch.arange(n, dtype=torch.float)
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for step in range(steps):
        x = torch.rand(1024, n) * (torch.rand(1024, n) > S)
        out = torch.relu(x @ W.T @ W + b)
        loss = (imp * (out - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach(), b.detach(), loss.item()


print("1. 5 features → 2 dimensions (norm of each feature's direction; > 0.5 = represented)")
shapes = {}
for S in (0.0, 0.7, 0.9, 0.97):
    W, b, loss = train(5, 2, S)
    norms = W.norm(dim=0)
    shapes[S] = W.T.tolist()
    ang = torch.atan2(W[1], W[0]) * 180 / torch.pi
    rep = (norms > 0.5).sum().item()
    print(f"   sparsity {S:.2f}: {rep} features represented; norms " + " ".join(f"{v:.2f}" for v in norms.tolist())
          + "; angles " + " ".join(f"{v:.0f}°" for v in ang.tolist()))
json.dump({str(k): v for k, v in shapes.items()}, open("toy_5x2.json", "w"))

print("\n2. 100 features → 20 dimensions, all equally important")
for S in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
    W, b, loss = train(100, 20, S, steps=10000, decay=1.0)
    norms = W.norm(dim=0)
    rep = norms > 0.5
    unit = W / norms.clamp_min(1e-8)
    overlap = (unit.T @ unit).abs()
    overlap.fill_diagonal_(0)
    inter = overlap[rep][:, rep].max(1).values.mean().item() if rep.sum() > 1 else 0.0
    signed = (unit.T @ unit)[rep][:, rep].fill_diagonal_(0)
    partner = signed.gather(1, signed.abs().argmax(1, keepdim=True)).mean().item() if rep.sum() > 1 else 0.0
    print(f"   sparsity {S:.2f}: {rep.sum().item():>3} features represented in 20 dimensions "
          f"({rep.sum().item() / 20:.1f} per dimension); mean largest overlap with another feature {inter:.2f} "
          f"(signed cosine with that partner {partner:+.2f})", flush=True)
