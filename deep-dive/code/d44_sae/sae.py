"""Deep Dive, episode 44: Sparse Autoencoders.

Unpack GPT-2 small's residual stream (after layer 6, 768 numbers per token) into many sparse features:
  f = ReLU(W_enc (x - b_dec) + b_enc),  x_hat = W_dec f + b_dec,  loss = |x - x_hat|^2 + L1 * sum_i f_i |W_dec[:, i]|
trained on Tiny Shakespeare activations.
1. How well it reconstructs (variance explained), how sparse it is (features active per token), dead features.
2. Splice the reconstruction back into GPT-2: how much does the next-token loss suffer?
3. What features mean: the tokens (with context) where some features fire most.
4. Compare with the 768 raw dimensions: how many of the top activations of a feature vs a raw neuron share the same token.
"""
import time

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.manual_seed(0)
torch.set_num_threads(8)
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
gpt = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
LAYER, D, M, L1, STEPS, BATCH = 6, 768, 768 * 8, 2.0, 3000, 4096   # L1 from a sweep: 1 → ~290 active, 3 → ~24
text = open("../../../how-llms-work/tiny_gpt/input.txt").read()
ids = tok(text, return_tensors="pt").input_ids[0]
CTX = 128
N_SHOW = 100_000                                                   # tokens used to interpret features
chunks = ids[: len(ids) // CTX * CTX].view(-1, CTX)
train_chunks, test_chunks = chunks[:-100], chunks[-100:]


@torch.no_grad()
def activations(ch):
    out = []
    for i in range(0, len(ch), 32):
        hs = gpt(ch[i:i + 32], output_hidden_states=True).hidden_states[LAYER]   # after LAYER blocks
        out.append(hs[:, 1:].reshape(-1, D))                       # skip position 0 (the attention sink)
    return torch.cat(out)


t0 = time.time()
X = activations(train_chunks)
Xt = activations(test_chunks)
scale = X.norm(dim=1).mean() / D ** 0.5                            # normalise so the average squared entry is ~1
X, Xt = X / scale, Xt / scale
print(f"{len(X):,} training activations, {len(Xt):,} test ({time.time() - t0:.0f} s)")


class SAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.randn(M, D) / D ** 0.5)
        self.W_dec = torch.nn.Parameter(self.W_enc.data.T.clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(M))
        self.b_dec = torch.nn.Parameter(X.mean(0).clone())

    def forward(self, x):
        f = torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)
        return f, f @ self.W_dec.T + self.b_dec


sae = SAE()
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
t0 = time.time()
for step in range(STEPS):
    x = X[torch.randint(len(X), (BATCH,))]
    f, xh = sae(x)
    mse = ((xh - x) ** 2).sum(1).mean()
    l1 = (f * sae.W_dec.norm(dim=0)).sum(1).mean()
    loss = mse + L1 * l1
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0 or step == STEPS - 1:
        print(f"   step {step + 1:>5}: reconstruction error {mse.item() / D:.3f} per dim, active features per token "
              f"{(f > 0).float().sum(1).mean():.0f} ({time.time() - t0:.0f} s)", flush=True)

with torch.no_grad():
    f, xh = sae(Xt)
    fve = 1 - ((xh - Xt) ** 2).sum() / ((Xt - Xt.mean(0)) ** 2).sum()
    l0 = (f > 0).float().sum(1).mean()
    fired = torch.zeros(M, dtype=torch.bool)
    for i in range(0, 200_000, 20_000):                            # in chunks: 200,000 × 6,144 floats would be 4.9 GB
        fired |= (sae(X[i:i + 20_000])[0] > 0).any(0)
print(f"\n1. test activations: variance explained {fve * 100:.1f}%, {l0:.1f} of {M} features active per token on "
      f"average, {(~fired).sum().item()} features never fire ({(~fired).float().mean() * 100:.1f}%)")


# ---------------------------------------------------------------- 2. splice the reconstruction into GPT-2
def splice(mode):
    def hook(mod, inp, out):
        h = out[0]
        x = h[:, 1:] / scale
        rec = sae(x)[1] if mode == "sae" else torch.zeros_like(x) + X.mean(0)
        h = torch.cat([h[:, :1], rec * scale], 1)
        return (h,) + tuple(out[1:])
    return hook


@torch.no_grad()
def lm_loss(mode=None):
    hk = gpt.transformer.h[LAYER - 1].register_forward_hook(splice(mode)) if mode else None
    tot = 0.0
    for i in range(0, len(test_chunks), 20):
        c = test_chunks[i:i + 20]
        lg = gpt(c).logits[:, 1:-1]
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), c[:, 2:].reshape(-1)).item()
    if hk:
        hk.remove()
    return tot / (len(test_chunks) / 20)


base, spliced, ablated = lm_loss(), lm_loss("sae"), lm_loss("mean")
print(f"\n2. next-token loss on test text: GPT-2 {base:.3f}; with the SAE reconstruction at layer {LAYER}: {spliced:.3f}; "
      f"with the layer replaced by its mean: {ablated:.3f}")
print(f"   the SAE recovers {(ablated - spliced) / (ablated - base) * 100:.0f}% of the loss gap")


# ---------------------------------------------------------------- 3. what features mean
def show(acts, flat_ids, k, label):
    top = acts.topk(k).indices
    toks = [tok.decode(flat_ids[i]) for i in top]
    ctx = [tok.decode(flat_ids[max(0, i - 6):i + 1]).replace("\n", "⏎") for i in top]
    return toks, ctx


with torch.no_grad():
    Xs = X[:N_SHOW]
    F_all = torch.cat([sae(Xs[i:i + 20_000])[0] for i in range(0, N_SHOW, 20_000)])
flat = train_chunks[:, 1:].reshape(-1)[:N_SHOW]
freq = (F_all > 0).float().mean(0)
alive = ((freq > 1e-4) & (freq < 0.05)).nonzero().squeeze(1)
torch.manual_seed(1)
pick = alive[torch.randperm(len(alive))[:8]]
print("\n3. eight random live features: where they fire most (token, with the 6 tokens before it)")
for j in pick.tolist():
    toks, ctx = show(F_all[:, j], flat, 5, f"feature {j}")
    print(f"   feature {j:>4} (active on {freq[j] * 100:.2f}% of tokens): " + " | ".join(repr(c[-40:]) for c in ctx))


# ---------------------------------------------------------------- 4. features vs raw dimensions
def purity(acts, n_units, k=20):
    """For each unit: the share of its top-k activations that are the same token as its most common top token."""
    out = []
    for j in range(n_units):
        top = acts[:, j].topk(k).indices
        t = flat[top]
        out.append(t.bincount().max().item() / k)
    return torch.tensor(out)


with torch.no_grad():
    n_feat = min(300, len(alive))
    sae_p = purity(F_all[:, alive[:n_feat]], n_feat)
    raw_p = purity(Xs.abs(), D)
print(f"\n4. top-20 activations that share one token: SAE features ({n_feat} live) {sae_p.mean() * 100:.0f}% on average, "
      f"raw residual dimensions (768) {raw_p.mean() * 100:.0f}%")
print(f"   units whose top 20 are all one token: SAE {(sae_p == 1).float().mean() * 100:.0f}%, raw {(raw_p == 1).float().mean() * 100:.0f}%")
