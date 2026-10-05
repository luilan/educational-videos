"""Deep Dive, episode 15: Activations: ReLU, GELU, SwiGLU.

1. The three functions at a few points.
2. GPT-2's MLP uses GELU: how its 3,072 hidden values are distributed, and what happens if we swap in ReLU.
3. Qwen2.5's MLP uses SwiGLU: down(silu(gate(x)) * up(x)). Recompute it by hand.
4. Train the same tiny GPT with ReLU, GELU and SwiGLU (same parameter count); count ReLU neurons that never fire.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast

HERE = Path(__file__).parent
torch.set_num_threads(8)
torch.set_printoptions(precision=3, sci_mode=False)
text = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()

# ---------------------------------------------------------------- 1. the functions
xs = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
print("1.        x:", xs)
print("       ReLU:", F.relu(xs))
print("       GELU:", F.gelu(xs, approximate="tanh"))
print("  SiLU/Swish:", F.silu(xs))
print(f"   smallest GELU value: {F.gelu(torch.linspace(-3, 0, 30001), approximate='tanh').min():.3f}")

# ---------------------------------------------------------------- 2. GPT-2: GELU, and swapping in ReLU
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tok.model_max_length = 10 ** 6
gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
ids = tok(text[100_000:106_000], return_tensors="pt").input_ids[:, :512]
hidden = []
hook = gpt2.transformer.h[6].mlp.act.register_forward_hook(lambda m, i, o: hidden.append((i[0], o)))
with torch.no_grad():
    base = gpt2(ids, labels=ids).loss.item()
hook.remove()
pre, post = hidden[0]
print(f"\n2. GPT-2 layer 6 MLP: 768 -> {pre.shape[-1]:,} -> 768, GELU in the middle")
print(f"   hidden values before GELU: {(pre < 0).float().mean():.0%} negative; "
      f"after GELU: {(post.abs() < 0.01).float().mean():.0%} within 0.01 of zero")
print(f"   loss with GELU {base:.3f}")
for blk in gpt2.transformer.h:
    blk.mlp.act = nn.ReLU()
with torch.no_grad():
    print(f"   swap every GELU for ReLU, no retraining: loss {gpt2(ids, labels=ids).loss.item():.3f}")

# ---------------------------------------------------------------- 3. Qwen2.5: SwiGLU by hand
name = "Qwen/Qwen2.5-0.5B-Instruct"
qtok = AutoTokenizer.from_pretrained(name)
qwen = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
mlp = qwen.model.layers[6].mlp
x = torch.randn(4, qwen.config.hidden_size)
with torch.no_grad():
    gate, up = mlp.gate_proj(x), mlp.up_proj(x)
    mine = mlp.down_proj(F.silu(gate) * up)
    print(f"\n3. {name} MLP: {qwen.config.hidden_size} -> {qwen.config.intermediate_size:,} (gate and up) -> "
          f"{qwen.config.hidden_size}, activation {qwen.config.hidden_act}")
    print(f"   by hand down(silu(gate(x)) * up(x)) vs the model: largest difference {(mine - mlp(x)).abs().max():.1e}")
    print(f"   three matrices: {sum(p.numel() for p in mlp.parameters()):,} parameters in one MLP")

# ---------------------------------------------------------------- 4. train with each activation
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
V, D, H, L, T = len(chars), 128, 4, 4, 64


class MLP(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "SwiGLU":                                       # 3 matrices, so 2/3 of the width: same parameters
            hid = 344
            self.gate, self.up, self.down = nn.Linear(D, hid, bias=False), nn.Linear(D, hid, bias=False), \
                nn.Linear(hid, D, bias=False)
        else:
            self.up, self.down = nn.Linear(D, 4 * D, bias=False), nn.Linear(4 * D, D, bias=False)

    def forward(self, x):
        if self.kind == "SwiGLU":
            return self.down(F.silu(self.gate(x)) * self.up(x))
        h = self.up(x)
        self.last = h
        return self.down(F.relu(h) if self.kind == "ReLU" else F.gelu(h, approximate="tanh"))


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = MLP(kind)

    def forward(self, x):
        B, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.Sequential(*[Block(kind) for _ in range(L)])
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


print(f"\n4. tiny GPT, {L} layers, 3,000 steps, about the same MLP parameters, three seeds")
for kind in ("ReLU", "GELU", "SwiGLU"):
    losses, extra = [], ""
    for seed in (1337, 1, 2):
        torch.manual_seed(seed)
        model = TinyGPT(kind)
        mlp_params = sum(p.numel() for p in model.blocks[0].mlp.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(3000):
            x, y = batch(train_data)
            loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        losses.append(val_loss(model))
        if kind == "ReLU" and seed == 1337:
            with torch.no_grad():
                g = torch.Generator().manual_seed(1)
                fired = torch.zeros(L, 4 * D, dtype=torch.bool)
                for _ in range(20):
                    model(batch(val_data, g=g)[0])
                    for l, b in enumerate(model.blocks):
                        fired[l] |= (b.mlp.last > 0).flatten(0, 1).any(0)
            extra = f"   (neurons that never fired on 40,960 tokens: {(~fired).sum().item()} of {fired.numel()})"
    print(f"   {kind:<7} {mlp_params:,} MLP parameters   val loss " + " / ".join(f"{v:.3f}" for v in losses)
          + f"   mean {sum(losses) / 3:.3f}{extra}")
