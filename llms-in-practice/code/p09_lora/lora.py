"""LLMs in Practice, episode 9: Fine-tuning vs prompting, and LoRA in one picture.

Teach the tiny GPT from How LLMs Work (trained on Shakespeare) a new style, made-up recipes, in two ways:
full fine-tuning (every weight changes) and LoRA (the weights stay frozen; two small matrices per layer learn the
change). Compare trainable parameters, losses and samples.

    pip install -r requirements.txt
    python lora.py
CPU only. The first run trains the Shakespeare base model (about 25 minutes) and saves base.pt; later runs reuse it.
"""
import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = Path(__file__).parent
torch.manual_seed(1337)
torch.set_num_threads(8)

# ---------------------------------------------------------------- the tiny GPT (How LLMs Work, episode 12)
shakespeare = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()
recipes = (HERE / "recipes.txt").read_text()
chars = sorted(set(shakespeare))
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: torch.tensor([stoi[c] for c in s])
decode = lambda t: "".join(chars[i] for i in t)
vocab_size, block_size, batch_size = len(chars), 64, 32
n_embd, n_head, n_layer = 128, 4, 4


def split(text):
    d = encode(text)
    n = int(0.9 * len(d))
    return d[:n], d[n:]


sh_train, sh_val = split(shakespeare)
rc_train, rc_val = split(recipes)


def get_batch(d):
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    return (torch.stack([d[i:i + block_size] for i in ix]), torch.stack([d[i + 1:i + block_size + 1] for i in ix]))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q, k, v = (t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // n_head)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf")).softmax(-1)
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up, self.down = nn.Linear(n_embd, 4 * n_embd), nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.attn, self.ln2, self.mlp = nn.LayerNorm(n_embd), Attention(), nn.LayerNorm(n_embd), MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb, self.pos_emb = nn.Embedding(vocab_size, n_embd), nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f, self.head = nn.LayerNorm(n_embd), nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.shape[1]))
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, n_new=160, temperature=0.8, seed=7):
        torch.manual_seed(seed)
        idx = encode(prompt)[None]
        for _ in range(n_new):
            logits, _ = self(idx[:, -block_size:])
            idx = torch.cat([idx, torch.multinomial((logits[:, -1] / temperature).softmax(-1), 1)], 1)
        return decode(idx[0].tolist())


def train(model, data, steps, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    for _ in range(steps):
        x, y = get_batch(data)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


@torch.no_grad()
def val_loss(model, data, batches=40):
    g = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(batches):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=g)
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        losses.append(model(x, y)[1].item())
    return sum(losses) / len(losses)


# ---------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """y = W x + (B A x) * alpha / r, with W frozen and only A (r x in) and B (out x r) trained."""

    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # starts at zero: no change at first
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        return out + (x @ self.A.T @ self.B.T) * self.scale if self.enabled else out


def add_lora(model, r=4):
    for p in model.parameters():
        p.requires_grad = False                                      # freeze every original weight
    for block in model.blocks:
        block.attn.qkv, block.attn.proj = LoRALinear(block.attn.qkv, r), LoRALinear(block.attn.proj, r)
        block.mlp.up, block.mlp.down = LoRALinear(block.mlp.up, r), LoRALinear(block.mlp.down, r)
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


count = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------------------------------------------- 1. the base model: Shakespeare
base = TinyGPT()
ckpt = HERE / "base.pt"
if ckpt.exists():
    base.load_state_dict(torch.load(ckpt))
else:
    print("training the Shakespeare base model (3,000 steps, about 25 minutes on a CPU) ...")
    train(base, sh_train, 3000, 1e-3)
    torch.save(base.state_dict(), ckpt)
base.eval()
print(f"base model: {count(base):,} parameters")
print(f"  loss  Shakespeare {val_loss(base, sh_val):.2f}   recipes {val_loss(base, rc_val):.2f}")
print("  sample:", repr(base.generate("RECIPE:")))

STEPS = 600
# ---------------------------------------------------------------- 2. full fine-tuning: every weight changes
full = copy.deepcopy(base)
train(full, rc_train, STEPS, 1e-3)
print(f"\nfull fine-tuning: {count(full):,} trainable parameters (100%)")
print(f"  loss  Shakespeare {val_loss(full, sh_val):.2f}   recipes {val_loss(full, rc_val):.2f}")
print("  sample:", repr(full.generate("RECIPE:")))

# ---------------------------------------------------------------- 3. LoRA: freeze everything, train small A and B
tuned = copy.deepcopy(base)
adapters = add_lora(tuned, r=4)
print(f"\nLoRA r=4: {count(tuned):,} trainable parameters ({count(tuned) / count(base):.1%} of the model), "
      f"adapter file {count(tuned) * 4 / 1024:.0f} KB vs model {count(base) * 4 / 1024:.0f} KB")
print(f"  before training (B = 0): recipes {val_loss(tuned, rc_val):.2f}  (identical to the base model)")
train(tuned, rc_train, STEPS, 3e-3)
print(f"  loss  Shakespeare {val_loss(tuned, sh_val):.2f}   recipes {val_loss(tuned, rc_val):.2f}")
print("  sample:", repr(tuned.generate("RECIPE:")))
for a in adapters:
    a.enabled = False
print(f"  adapter switched off: Shakespeare {val_loss(tuned, sh_val):.2f}  (the base weights never changed)")
