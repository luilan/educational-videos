"""A tiny character-level GPT, built from the pieces in the "How LLMs Work" series."""
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# --- Tokens: every character is a token (video 2) --------------------------------------
text = open("input.txt").read()
chars = sorted(set(text))
vocab_size = len(chars)                                   # 65
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text))
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

# --- Hyperparameters -----------------------------------------------------------------------
block_size, batch_size = 64, 32                           # context length, sequences per step
n_embd, n_head, n_layer = 128, 4, 4
max_steps, lr = 5000, 1e-3


def get_batch(d):
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])   # the next character
    return x, y


class Attention(nn.Module):                                 # videos 5-7
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q, k, v = (t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in (q, k, v))
        att = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))   # no peeking ahead
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)            # merge the heads
        return self.proj(y)


class MLP(nn.Module):                                       # video 8
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(n_embd, 4 * n_embd)
        self.down = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):                                     # video 9
    def __init__(self):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(n_embd), Attention()
        self.ln2, self.mlp = nn.LayerNorm(n_embd), MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))                      # tokens talk
        x = x + self.mlp(self.ln2(x))                       # each token thinks
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)     # video 3
        self.pos_emb = nn.Embedding(block_size, n_embd)     # video 4 (learned, like GPT-2)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)           # video 10: logits

    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.shape[1]))
        return self.head(self.ln_f(self.blocks(x)))

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8):       # video 1: predict, pick, append
        for _ in range(n_new):
            logits = self(idx[:, -block_size:])[:, -1] / temperature
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


model = TinyGPT()
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params:,} parameters")
opt = torch.optim.AdamW(model.parameters(), lr=lr)


@torch.no_grad()
def val_loss(batches=20):
    losses = [F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1)).item()
              for x, y in (get_batch(val_data) for _ in range(batches))]
    return sum(losses) / len(losses)


def sample(n=400, seed=7):
    g = torch.random.get_rng_state()
    torch.manual_seed(seed)
    out = decode(model.generate(torch.tensor([encode("\n")]), n)[0].tolist())
    torch.random.set_rng_state(g)
    return out


log = {"params": n_params, "train": [], "val": [], "samples": {}}
t0 = time.time()
for step in range(max_steps + 1):                           # video 11: the training loop
    if step % 250 == 0:
        log["val"].append((step, val_loss()))
    if step in (0, 250, 1000, max_steps):
        log["samples"][step] = sample()
        print(f"--- step {step}, val loss {log['val'][-1][1]:.3f}, {time.time() - t0:.0f}s\n{log['samples'][step]}\n")
    x, y = get_batch(train_data)
    logits = model(x)                                       # forward pass
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    opt.zero_grad()
    loss.backward()                                         # backpropagation
    opt.step()                                              # nudge every weight
    log["train"].append((step, loss.item()))

log["seconds"] = time.time() - t0
json.dump(log, open("training_log.json", "w"))
print(f"done in {log['seconds'] / 60:.1f} min")
