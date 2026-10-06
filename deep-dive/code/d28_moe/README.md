# Deep Dive, episode 28 · Mixture of Experts

A mixture-of-experts layer (router + 8 expert MLPs, top-k routing, load-balancing loss) in a tiny GPT, compared with
dense models.

```bash
pip install -r requirements.txt
python moe.py
```

PyTorch only; five 4-layer tiny GPTs, 2,000 steps each (about an hour on a CPU; the expert loop is not optimized). It
prints:

| model | val loss | parameters | active per token |
|---|---|---|---|
| dense MLP | 1.644 | 818,241 | 818,241 |
| MoE top-1, no balancing | 1.687 | 4,510,273 | 822,337 |
| MoE top-1, balancing loss (0.01) | 1.695 | 4,510,273 | 822,337 |
| MoE top-2, balancing loss | 1.596 | 4,510,273 | 1,349,185 |
| dense MLP twice as wide | 1.628 | 1,344,577 | 1,344,577 |

and the share of tokens per expert in layer 2: 4%–20% without balancing, 11%–14% with it.
