"""Deep Dive, episode 28: Mixture of Experts.

Replace each MLP of a tiny GPT with 8 expert MLPs and a router that sends every token to its top-k experts.
1. Train: dense MLP vs MoE (top-1 and top-2), same steps; count total and active parameters per token.
2. Load balancing: without an auxiliary loss, does the router use all experts?
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = Path(__file__).parent
torch.set_num_threads(8)
text = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
V, D, H, L, T, E, HID = len(chars), 128, 4, 4, 64, 8, 512


class MoE(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.router = nn.Linear(D, E, bias=False)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(D, HID), nn.GELU(), nn.Linear(HID, D)) for _ in range(E)])
        self.aux = torch.tensor(0.0)
        self.counts = torch.zeros(E)

    def forward(self, x):
        flat = x.reshape(-1, D)
        probs = self.router(flat).softmax(-1)                      # (tokens, 8): how much each expert suits each token
        top_p, top_i = probs.topk(self.k, dim=-1)                  # keep the k best experts per token
        top_p = top_p / top_p.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e in range(E):
            hit = (top_i == e)                                     # (tokens, k)
            rows = hit.any(-1).nonzero().flatten()
            if len(rows):
                w = (top_p * hit)[rows].sum(-1, keepdim=True)
                out[rows] += w * self.experts[e](flat[rows])       # only the chosen tokens run through expert e
        frac = F.one_hot(top_i[:, 0], E).float().mean(0)           # share of tokens whose first choice is each expert
        self.aux = E * (frac * probs.mean(0)).sum()                # load-balancing loss (1.0 when perfectly balanced)
        self.counts += F.one_hot(top_i, E).sum((0, 1)).float().detach()
        return out.reshape(x.shape)


class Block(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        if k == "wide":                                             # dense, twice as wide: same active size as top-2
            self.mlp = nn.Sequential(nn.Linear(D, 2 * HID), nn.GELU(), nn.Linear(2 * HID, D))
        else:
            self.mlp = MoE(k) if k else nn.Sequential(nn.Linear(D, HID), nn.GELU(), nn.Linear(HID, D))

    def forward(self, x):
        B, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.Sequential(*[Block(k) for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)

    def forward(self, idx):
        return self.head(self.ln(self.blocks(self.emb(idx) + self.pos(torch.arange(idx.shape[1])))))


def batch(d, bs=32, g=None):
    ix = torch.randint(len(d) - T - 1, (bs,), generator=g)
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


@torch.no_grad()
def val_loss(model):
    g = torch.Generator().manual_seed(0)
    return sum(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()
               for x, y in (batch(val_data, g=g) for _ in range(40))) / 40


def params(model, k):
    total = sum(p.numel() for p in model.parameters())
    if not k or k == "wide":
        return total, total
    expert = sum(p.numel() for p in model.blocks[0].mlp.experts[0].parameters())
    return total, total - L * (E - k) * expert                     # active: only k experts run per token


print(f"tiny GPT, {L} layers, 2,000 steps; MoE layers have {E} experts of {D} -> {HID} -> {D}")
for name, k, aux_w in (("dense MLP", 0, 0.0), ("MoE top-1, no balancing", 1, 0.0), ("MoE top-1, balancing", 1, 0.01),
                       ("MoE top-2, balancing", 2, 0.01), ("dense MLP, twice as wide", "wide", 0.0)):
    torch.manual_seed(1337)
    model = TinyGPT(k)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2000):
        x, y = batch(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        if aux_w:
            loss = loss + aux_w * sum(b.mlp.aux for b in model.blocks)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    if k and k != "wide":
        for b in model.blocks:
            b.mlp.counts.zero_()
    vl = val_loss(model)
    total, active = params(model, k)
    line = f"   {name:<26} val loss {vl:.3f}   {total:>9,} parameters, {active:>9,} active per token"
    print(line)
    if k and k != "wide":
        share = model.blocks[2].mlp.counts / model.blocks[2].mlp.counts.sum()
        print("      layer 2, share of tokens per expert: " + " ".join(f"{s:.0%}" for s in share.tolist()))
