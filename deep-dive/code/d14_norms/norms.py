"""Deep Dive, episode 14: LayerNorm vs RMSNorm.

1. LayerNorm by hand on GPT-2's residual stream: subtract the mean, divide by the spread, scale and shift.
2. RMSNorm by hand on Qwen2.5's: divide by the root mean square, scale. No mean, no shift.
3. What normalization does: the stream grows layer after layer, but every block reads it at the same size.
4. Train the same 8-layer tiny GPT with LayerNorm, RMSNorm, and no normalization; time both norms.
"""
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast

HERE = Path(__file__).parent
torch.set_num_threads(8)
torch.set_printoptions(precision=3, sci_mode=False)
text = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()

# ---------------------------------------------------------------- 1. LayerNorm, GPT-2
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
ids = tok("The cat sat on the mat because it was tired", return_tensors="pt").input_ids
with torch.no_grad():
    hs = gpt2(ids, output_hidden_states=True).hidden_states
x = hs[6][0, 2]                                                    # " sat", entering layer 6
ln = gpt2.transformer.h[6].ln_1
mean, var = x.mean(), x.var(unbiased=False)
mine = (x - mean) / torch.sqrt(var + ln.eps) * ln.weight + ln.bias
print(f"1. LayerNorm on ' sat' entering GPT-2 layer 6 (768 numbers)")
print(f"   mean {mean:.3f}, standard deviation {var.sqrt():.3f}, size (norm) {x.norm():.1f}")
print(f"   first 4 in: {x[:4]}")
print(f"   normalized (before scale and shift): {((x - mean) / torch.sqrt(var + ln.eps))[:4]}")
print(f"   by hand vs GPT-2's LayerNorm: largest difference {(mine - ln(x)).abs().max().item():.1e}")
print(f"   learned scale (gamma) and shift (beta): {ln.weight.numel()} + {ln.bias.numel()} numbers")

# ---------------------------------------------------------------- 2. RMSNorm, Qwen2.5
name = "Qwen/Qwen2.5-0.5B-Instruct"
qtok = AutoTokenizer.from_pretrained(name)
qwen = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
qids = qtok("The cat sat on the mat because it was tired", return_tensors="pt").input_ids
with torch.no_grad():
    qh = qwen(qids, output_hidden_states=True).hidden_states
xq = qh[6][0, 2]
rms_layer = qwen.model.layers[6].input_layernorm
rms = torch.sqrt(xq.pow(2).mean() + rms_layer.variance_epsilon)
mine_q = xq / rms * rms_layer.weight
print(f"\n2. RMSNorm in {name}, layer 6 ({xq.numel()} numbers): {type(rms_layer).__name__}")
print(f"   mean {xq.mean():.3f}, root mean square {rms:.3f}")
print(f"   by hand vs Qwen's RMSNorm: largest difference {(mine_q - rms_layer(xq)).abs().max().item():.1e}")
print(f"   learned scale only: {rms_layer.weight.numel()} numbers, no shift")

# ---------------------------------------------------------------- 3. the stream grows, the input to each block does not
print("\n3. GPT-2, average token: size of the stream vs size of what layer l's attention reads")
with torch.no_grad():
    for l in (0, 3, 6, 9, 11):
        s = hs[l][0, 1:]
        n = (s - s.mean(-1, keepdim=True)) / s.std(-1, unbiased=False, keepdim=True)
        print(f"   layer {l:>2}: stream {s.norm(dim=-1).mean():6.1f}   normalized {n.norm(dim=-1).mean():5.1f}"
              f"   (sqrt(768) = {768 ** 0.5:.1f})")

# ---------------------------------------------------------------- 4. train with each norm
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
V, D, H, L, T = len(chars), 128, 4, 8, 64


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


NORMS = {"LayerNorm": lambda: nn.LayerNorm(D), "RMSNorm": lambda: RMSNorm(D), "no norm": lambda: nn.Identity()}


class Block(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.n1, self.n2 = NORMS[norm](), NORMS[norm]()
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        B, t, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, dim=2)
        q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, D))
        return x + self.mlp(self.n2(x))


class TinyGPT(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.Sequential(*[Block(norm) for _ in range(L)])
        self.ln, self.head = NORMS[norm](), nn.Linear(D, V)

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


print(f"\n4. tiny GPT, {L} layers, 1,500 steps; validation loss at 100 / 500 / 1,500 steps, for two learning rates")
for lr in (1e-3, 1e-2):
    for norm in NORMS:
        torch.manual_seed(1337)
        model = TinyGPT(norm)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        curve = []
        for step in range(1, 1501):
            x, y = batch(train_data)
            loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in (100, 500, 1500):
                model.eval()
                curve.append(f"{val_loss(model):.2f}")
                model.train()
        print(f"   lr {lr:g}  {norm:<10} " + " / ".join(curve))

big = torch.randn(4096, 4096)
print("   speed depends on the implementation: PyTorch's built-in versions, CPU (times vary between runs)")
for name, norm in (("LayerNorm", nn.LayerNorm(4096)), ("RMSNorm", nn.RMSNorm(4096, eps=1e-6))):
    with torch.no_grad():
        norm(big)
        t0 = time.perf_counter()
        for _ in range(50):
            norm(big)
    print(f"   {name}: {(time.perf_counter() - t0) / 50 * 1000:.1f} ms for 4,096 vectors of 4,096 numbers")
