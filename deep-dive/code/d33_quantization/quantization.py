"""Deep Dive, episode 33: Quantization, Deeper.

Qwen2.5-0.5B, loss on 4 × 512 tokens of Tiny Shakespeare (eval_text.txt, a fixed slice):
1. Outliers: how far the largest weight in a row is from a typical one.
2. 4-bit weights with one scale per row vs one scale per group of 128, 64 or 32 weights, and the cost in bits.
3. A better grid: NF4 (levels placed at normal-distribution quantiles) vs evenly spaced levels.
4. Activations: 8-bit inputs to every linear layer, one scale per tensor vs per token; the first token's huge
   values (the attention sink of episode 12) and what keeping it in 16 bits does.
"""
import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
torch.set_num_threads(8)
NAME = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
text = open(__file__.replace("quantization.py", "eval_text.txt")).read()
ids = tok(text, return_tensors="pt").input_ids[0]
CHUNKS = [ids[i * 512:(i + 1) * 512] for i in range(4)]
assert all(len(c) == 512 for c in CHUNKS), len(ids)
linears = [(n, m) for n, m in model.model.layers.named_modules() if isinstance(m, torch.nn.Linear)]
original = {n: m.weight.data.clone() for n, m in linears}


@torch.no_grad()
def loss():
    tot = 0.0
    for c in CHUNKS:
        logits = model(c[None]).logits[0, :-1].double()
        tot += F.cross_entropy(logits, c[1:]).item()
    return tot / len(CHUNKS)


def restore():
    for n, m in linears:
        m.weight.data.copy_(original[n])


def quant_uniform(w, bits, group):
    """Symmetric absmax rounding: one scale per group of `group` weights (group=None: one per row)."""
    rows, cols = w.shape
    g = cols if group is None else group
    x = w.reshape(rows, cols // g, g)
    levels = 2 ** (bits - 1) - 1                                  # 4 bits: -7..7
    scale = x.abs().amax(-1, keepdim=True) / levels
    return (torch.round(x / scale).clamp(-levels, levels) * scale).reshape(rows, cols)


# NF4: 16 levels at normal quantiles, rescaled to [-1, 1] (the QLoRA construction)
_n = torch.distributions.Normal(0, 1)
pos = _n.icdf(torch.linspace(0.5, 0.9677, 9))[1:]                 # 8 positive levels
neg = -_n.icdf(torch.linspace(0.5, 0.9677, 8))[1:]                # 7 negative levels
NF4 = torch.cat([neg.flip(0), torch.zeros(1), pos])
NF4 = NF4 / NF4.abs().max()


def quant_nf4(w, group):
    rows, cols = w.shape
    x = w.reshape(rows, cols // group, group)
    scale = x.abs().amax(-1, keepdim=True)
    idx = ((x / scale)[..., None] - NF4).abs().argmin(-1)
    return (NF4[idx] * scale).reshape(rows, cols)


def apply(fn):
    for n, m in linears:
        m.weight.data.copy_(fn(original[n]))


# ---------------------------------------------------------------- 1. outliers in weights
w = original["0.mlp.down_proj"]
ratios = torch.cat([(o.abs().amax(1) / o.abs().median(1).values) for o in original.values()])
print("1. weight outliers")
r0 = w[0].abs()
print(f"   one row of layer 0 down_proj ({w.shape[1]} weights): median |w| {r0.median():.4f}, largest {r0.max():.4f} "
      f"({r0.max() / r0.median():.0f}x)")
print(f"   all {len(ratios):,} rows: largest/median, median over rows {ratios.median():.0f}x, "
      f"90th percentile {ratios.quantile(0.9):.0f}x, max {ratios.max():.0f}x")
steps = r0.max() / 7
print(f"   with 4 bits and one scale for this row, the step is {steps:.4f}: "
      f"{(r0 < steps / 2).float().mean() * 100:.0f}% of its weights round to zero")

base = loss()
print(f"\n   full precision loss: {base:.3f}")

# ---------------------------------------------------------------- 2. group size
print("\n2. 4-bit weights, symmetric rounding (+ one 16-bit scale per group)")
for group in (None, 128, 64, 32):
    apply(lambda x: quant_uniform(x, 4, group))
    l = loss()
    eff = 4 + (16 / group if group else 0)
    name = "per row" if group is None else f"groups of {group}"
    rel = sum(((quant_uniform(original[n], 4, group) - original[n]).norm() / original[n].norm()).item()
              for n, _ in linears) / len(linears)
    print(f"   {name:>13}: loss {l:.3f}   weight error {rel * 100:4.1f}%   {eff:.2f} bits per weight")
restore()

print("\n   3 bits:")
for group in (None, 32):
    apply(lambda x: quant_uniform(x, 3, group))
    print(f"   {'per row' if group is None else f'groups of {group}':>13}: loss {loss():.3f}")
restore()

# ---------------------------------------------------------------- 3. NF4
print("\n3. the grid: 16 levels, groups of 64")
print("   NF4 levels:", " ".join(f"{v:+.2f}" for v in NF4.tolist()))
apply(lambda x: quant_uniform(x, 4, 64)); u = loss()
apply(lambda x: quant_nf4(x, 64)); n4 = loss()
restore()
print(f"   evenly spaced (15 used levels): loss {u:.3f}")
print(f"   NF4 (16 levels, denser near 0): loss {n4:.3f}")
print(f"   loss increase over full precision: {u - base:+.3f} vs {n4 - base:+.3f}")

# ---------------------------------------------------------------- 4. activations
print("\n4. activations: 8-bit inputs to every linear layer (weights kept in full precision)")
stats = {}


def grab(name):
    def hook(mod, inp):
        stats[name] = inp[0][0].abs().amax(-1)                    # largest input value, per token
    return hook


hs = [m.register_forward_pre_hook(grab(n)) for n, m in linears]
with torch.no_grad():
    model(CHUNKS[0][None])
for h in hs:
    h.remove()
first = sum(1 for s in stats.values() if s.argmax().item() == 0)
ratio = {n: (s[0] / s[1:].median()).item() for n, s in stats.items()}
worst = max(ratio, key=ratio.get)
s = stats[worst]
print(f"   the first token has the largest input in {first} of {len(stats)} linear layers")
print(f"   worst, {worst}: first token {s[0]:.1f}, typical token {s[1:].median():.2f} ({ratio[worst]:.0f}x)")
print(f"   with one scale for the whole tensor, the 8-bit step there is {s.max() / 127:.2f}: "
      f"{(s[1:] < s.max() / 254).float().mean() * 100:.0f}% of the other tokens round to all zeros")


def act_quant(mode):
    def hook(mod, inp):
        x = inp[0]
        if mode == "token":
            sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
            return (torch.round(x / sc).clamp(-127, 127) * sc,)
        rest = x[:, 1:] if mode == "keep first" else x
        sc = rest.abs().amax() / 127
        q = torch.round(x / sc).clamp(-127, 127) * sc
        if mode == "keep first":
            q[:, 0] = x[:, 0]                                       # the first token stays in 16 bits
        return (q,)
    return hook


for mode, label in [("tensor", "one scale per tensor"), ("keep first", "per tensor, first token kept"),
                    ("token", "one scale per token")]:
    hs = [m.register_forward_pre_hook(act_quant(mode)) for n, m in linears]
    print(f"   {label:>29}: loss {loss():.3f}")
    for h in hs:
        h.remove()
