"""Deep Dive, episode 18: Backprop Through a Transformer.

1. The chain rule on a tiny two-step function, by hand vs autograd.
2. Check autograd on a real GPT-2 weight: nudge it and watch the loss (finite differences, float64).
3. One backward pass through GPT-2: how big is the gradient at each layer?
4. The cost: time of the backward pass vs the forward pass, and the memory kept for it.
"""
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = Path(__file__).parent
torch.set_num_threads(8)
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tok.model_max_length = 10 ** 6
text = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()

# ---------------------------------------------------------------- 1. chain rule by hand
w = torch.tensor(0.5, requires_grad=True)
x, target = torch.tensor(2.0), torch.tensor(3.0)
h = torch.tanh(w * x)                 # step 1
loss = (h - target) ** 2              # step 2
loss.backward()
by_hand = 2 * (h - target) * (1 - h ** 2) * x       # dloss/dh * dh/d(wx) * d(wx)/dw
print(f"1. loss = (tanh(w·x) − 3)², w = 0.5, x = 2")
print(f"   chain rule by hand: {by_hand.item():.6f}   autograd: {w.grad.item():.6f}")

# ---------------------------------------------------------------- 2. finite differences on GPT-2
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").double().eval()
ids = tok(text[100_000:103_000], return_tensors="pt").input_ids[:, :256]


def lm_loss(m, ids):
    """Our own cross-entropy, so it stays in the model's precision (float64 here)."""
    return F.cross_entropy(m(ids).logits[0, :-1], ids[0, 1:])


loss = lm_loss(model, ids)
loss.backward()
W = model.transformer.h[5].mlp.c_fc.weight                          # one weight in layer 5's MLP
i, j = 100, 200
grad = W.grad[i, j].item()
eps = 1e-4
with torch.no_grad():
    W[i, j] += eps
    up = lm_loss(model, ids).item()
    W[i, j] -= 2 * eps
    down = lm_loss(model, ids).item()
    W[i, j] += eps
print(f"\n2. GPT-2 (float64), loss {loss.item():.6f}; weight [{i}, {j}] of layer 5's MLP")
print(f"   autograd gradient {grad:.10f}   nudge ±{eps:g}: (loss up − loss down) / 2ε = {(up - down) / (2 * eps):.10f}")

# ---------------------------------------------------------------- 3. gradient size per layer
print("\n3. size of the gradient for each layer's weights (one backward pass)")
for l, blk in enumerate(model.transformer.h):
    a = torch.cat([p.grad.flatten() for p in blk.attn.parameters()]).norm().item()
    m = torch.cat([p.grad.flatten() for p in blk.mlp.parameters()]).norm().item()
    print(f"   layer {l:>2}: attention {a:.3f}   MLP {m:.3f}")
emb = model.transformer.wte.weight.grad
print(f"   token embeddings (shared with the output layer): {emb.norm().item():.3f}; rows with a nonzero gradient: "
      f"{(emb.abs().sum(1) > 0).sum().item():,} of {emb.shape[0]:,}")
total = sum(p.numel() for p in model.parameters())
print(f"   every one of the {total:,} parameters gets a gradient from a single loss number")

# ---------------------------------------------------------------- 4. time and memory
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").float().eval()
ids = tok(text[100_000:110_000], return_tensors="pt").input_ids[:, :1024]


saved, param_ptrs = [], {p.data_ptr() for p in model.parameters()}


def pack(t):
    if t.data_ptr() not in param_ptrs:                             # weights are kept anyway; count activations only
        saved.append(t.numel() * t.element_size())
    return t


with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):       # count what autograd keeps for backward
    loss = lm_loss(model, ids)
weights = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"\n4. GPT-2, 1,024 tokens (float32): weights {weights / 2**20:,.0f} MiB; tensors saved during the forward pass "
      f"for the backward pass (activations, not weights): {len(saved):,}, {sum(saved) / 2**20:,.0f} MiB")
del loss

fwd, bwd = [], []
for _ in range(5):
    model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    loss = lm_loss(model, ids)
    t1 = time.perf_counter()
    loss.backward()
    t2 = time.perf_counter()
    fwd.append(t1 - t0)
    bwd.append(t2 - t1)
f, b = sorted(fwd[1:])[1], sorted(bwd[1:])[1]          # skip the first (warm-up) run, take the median
print(f"   CPU time: forward {f:.2f} s, backward {b:.2f} s (backward / forward = {b / f:.1f})")
