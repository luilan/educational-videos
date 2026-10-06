"""Deep Dive, episode 45: Circuits.

Indirect object identification (IOI) in GPT-2 small: "When Mary and John went to the store, John gave a drink to" → Mary.
1. Behaviour: how often GPT-2 prefers the right name, and the logit difference logit(IO) - logit(S).
2. Activation patching, residual stream: run a corrupted prompt (the second name swapped, so the answer flips), copy in
   the clean run's residual stream at one layer and one position, and measure how much of the clean logit difference
   comes back.
3. Activation patching, heads: the same with each attention head's output at the final position.
4. Knockout: replace the top heads' outputs at the final position with their average over all prompts.
"""
import itertools
import json
import random

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

random.seed(0)
torch.set_num_threads(8)
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
L, H, DH = 12, 12, 64
NAMES = [" Mary", " John", " Tom", " James", " Anna", " Sarah", " David", " Paul", " Kate", " Mark", " Lisa", " Peter"]
TEMPLATES = ["When{A} and{B} went to the store,{S} gave a drink to",
             "When{A} and{B} got a snack at the cafe,{S} decided to give it to",
             "After{A} and{B} went to the park,{S} handed a ball to"]
assert all(len(tok(n).input_ids) == 1 for n in NAMES)
pairs = random.sample(list(itertools.permutations(NAMES, 2)), 30)
data = []                                                          # (clean ids, corrupted ids, IO token, S token)
for t in TEMPLATES:
    for a, b in pairs:
        clean = t.format(A=a, B=b, S=b)                            # answer: a (the indirect object)
        corrupt = t.format(A=a, B=b, S=a)                          # second mention swapped: answer becomes b
        data.append((tok(clean).input_ids, tok(corrupt).input_ids, tok(a).input_ids[0], tok(b).input_ids[0]))


def batch(template_idx):
    rows = data[template_idx * len(pairs):(template_idx + 1) * len(pairs)]
    c = torch.tensor([r[0] for r in rows])
    x = torch.tensor([r[1] for r in rows])
    return c, x, torch.tensor([r[2] for r in rows]), torch.tensor([r[3] for r in rows])


def logit_diff(logits, io, s):
    last = logits[:, -1]
    return (last.gather(1, io[:, None]) - last.gather(1, s[:, None])).squeeze(1)


# ---------------------------------------------------------------- 1. behaviour
with torch.no_grad():
    diffs, corr = [], []
    for ti in range(len(TEMPLATES)):
        c, x, io, s = batch(ti)
        diffs.append(logit_diff(model(c).logits, io, s))
        corr.append(logit_diff(model(x).logits, io, s))
    diffs, corr = torch.cat(diffs), torch.cat(corr)
print(f"1. {len(diffs)} prompts ({len(TEMPLATES)} templates × {len(pairs)} name pairs)")
print(f"   clean: right name preferred {(diffs > 0).float().mean() * 100:.0f}%, mean logit difference {diffs.mean():.2f}")
print(f"   corrupted (second name swapped): logit difference {corr.mean():.2f}")
c, x, io, s = batch(0)
print(f"   e.g. {tok.decode(c[0])!r} → {tok.decode(io[0])!r}")


# ---------------------------------------------------------------- 2. residual stream patching
@torch.no_grad()
def run_with_patch(ids, patch):
    """patch(layer_index) → hook function or None; hooks on block outputs."""
    hooks = []
    for l in range(L):
        h = patch(l)
        if h:
            hooks.append(model.transformer.h[l].register_forward_hook(h))
    out = model(ids).logits
    for h in hooks:
        h.remove()
    return out


labels = ["When", "A", "and", "B", "…", "S2", "…", "end"]
print("\n2. residual stream patching (clean → corrupted), template 1: share of the clean logit difference recovered")
c, x, io, s = batch(0)
block_out = {}                                                     # raw output of each block on the clean prompts
hooks = [model.transformer.h[l].register_forward_hook(lambda mod, inp, out, l=l: block_out.__setitem__(l, out[0].clone()))
         for l in range(L)]                                        # (hidden_states[-1] already has the final norm)
with torch.no_grad():
    clean_logits = model(c).logits
for h in hooks:
    h.remove()
with torch.no_grad():
    base_c, base_x = logit_diff(clean_logits, io, s).mean(), logit_diff(model(x).logits, io, s).mean()
toks = tok.convert_ids_to_tokens(c[0])
pos = {"A": 1, "B": 3, "S2": toks.index(tok.convert_ids_to_tokens(c[0])[3], 4), "end": len(toks) - 1}
print("   positions: " + ", ".join(f"{k}={v} ({toks[v]})" for k, v in pos.items()))
grid = torch.zeros(L, len(pos))
for l in range(L):
    for j, p in enumerate(pos.values()):
        def patch(layer, l=l, p=p):
            if layer != l:
                return None

            def hook(mod, inp, out):
                h = out[0].clone()
                h[:, p] = block_out[l][:, p]
                return (h,) + tuple(out[1:])
            return hook
        d = logit_diff(run_with_patch(x, patch), io, s).mean()
        grid[l, j] = (d - base_x) / (base_c - base_x)
print("   layer  " + "  ".join(f"{k:>5}" for k in pos))
for l in range(L):
    print(f"   {l:>5}  " + "  ".join(f"{v:5.2f}" for v in grid[l].tolist()))
json.dump([[round(v, 2) for v in row] for row in grid.tolist()], open("residual.json", "w"))


# ---------------------------------------------------------------- 3. head patching at the end position
def head_outputs(ids):
    """Inputs to each layer's c_proj (concatenated head outputs), per layer."""
    store = {}
    hooks = [model.transformer.h[l].attn.c_proj.register_forward_pre_hook(
        lambda mod, inp, l=l: store.__setitem__(l, inp[0].detach().clone())) for l in range(L)]
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    return store


heads_clean = head_outputs(c)
end = pos["end"]
hp = torch.zeros(L, H)
for l in range(L):
    for h in range(H):
        def pre(mod, inp, h=h, l=l):
            z = inp[0].clone()
            z[:, end, h * DH:(h + 1) * DH] = heads_clean[l][:, end, h * DH:(h + 1) * DH]
            return (z,)
        hk = model.transformer.h[l].attn.c_proj.register_forward_pre_hook(pre)
        with torch.no_grad():
            d = logit_diff(model(x).logits, io, s).mean()
        hk.remove()
        hp[l, h] = (d - base_x) / (base_c - base_x)
flat = sorted([(hp[l, h].item(), l, h) for l in range(L) for h in range(H)], reverse=True)
print("\n3. head patching at the end position: share recovered, top 8")
print("   " + ", ".join(f"{l}.{h} {v:+.2f}" for v, l, h in flat[:8]))
neg = sorted(flat)[:3]
print("   most negative: " + ", ".join(f"{l}.{h} {v:+.2f}" for v, l, h in neg))

# ---------------------------------------------------------------- 4. knockout
all_c = [batch(ti)[0] for ti in range(len(TEMPLATES))]
means = {}
for ti, cc in enumerate(all_c):
    ho = head_outputs(cc)
    for l in range(L):
        means.setdefault(l, []).append(ho[l][:, -1].mean(0))
means = {l: torch.stack(v).mean(0) for l, v in means.items()}


def knockout(heads):
    hooks = []
    for l in {l for l, _ in heads}:
        hs = [h for ll, h in heads if ll == l]

        def pre(mod, inp, hs=hs, l=l):
            z = inp[0].clone()
            for h in hs:
                z[:, -1, h * DH:(h + 1) * DH] = means[l][h * DH:(h + 1) * DH]
            return (z,)
        hooks.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(pre))
    with torch.no_grad():
        d = torch.cat([logit_diff(model(batch(ti)[0]).logits, batch(ti)[2], batch(ti)[3]) for ti in range(len(TEMPLATES))])
    for h in hooks:
        h.remove()
    return d


print("\n4. knockout (mean-ablate heads at the end position), all prompts")
for k in (3, 6):
    top = [(l, h) for _, l, h in flat[:k]]
    d = knockout(top)
    rnd = [knockout(random.sample([(l, h) for l in range(L) for h in range(H)], k)).mean().item() for _ in range(5)]
    print(f"   top {k} heads {', '.join(f'{l}.{h}' for l, h in top)}: logit difference {diffs.mean():.2f} → {d.mean():.2f}, "
          f"right name {(d > 0).float().mean() * 100:.0f}%;  {k} random heads: {sum(rnd) / 5:.2f}")
