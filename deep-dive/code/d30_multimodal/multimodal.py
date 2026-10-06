"""Deep Dive, episode 30: Multimodal: Images as Tokens.

A tiny vision-language model, built from scratch:
1. A 32 x 32 color image is cut into 16 patches of 8 x 8; each patch (8 x 8 x 3 = 192 numbers) is projected by one linear
   layer into the same 128-number space as the text tokens.
2. The sequence is [16 image tokens] + [caption characters]; one causal transformer is trained to write the caption.
3. Test on new images: does it describe shape, color and position correctly?
"""
import math
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.set_num_threads(8)
SHAPES, COLORS = ["circle", "square", "triangle", "cross"], {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1)}
PLACES = {"top left": (8, 8), "top right": (8, 24), "bottom left": (24, 8), "bottom right": (24, 24)}
S, P = 32, 8                                   # image size and patch size: (32 / 8)^2 = 16 patches
N_PATCH, PATCH_DIM = (S // P) ** 2, P * P * 3


def draw(shape, color, place):
    img = torch.zeros(3, S, S) + 0.1 * torch.rand(3, S, S)          # a little noise
    cy, cx = PLACES[place]
    cy, cx = cy + random.randint(-2, 2), cx + random.randint(-2, 2)
    r = random.randint(4, 6)
    ys, xs = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    dy, dx = ys - cy, xs - cx
    mask = {"circle": dy ** 2 + dx ** 2 <= r * r,
            "square": (dy.abs() <= r) & (dx.abs() <= r),
            "triangle": (dy <= r) & (dy >= -r) & (dx.abs() <= (dy + r) / 2),
            "cross": ((dy.abs() <= 1) & (dx.abs() <= r)) | ((dx.abs() <= 1) & (dy.abs() <= r))}[shape]
    for c, v in enumerate(COLORS[color]):
        img[c][mask] = v
    return img


def patches(img):
    """(3, 32, 32) -> (16, 192): the image as a sequence of 16 patch tokens."""
    return img.unfold(1, P, P).unfold(2, P, P).permute(1, 2, 0, 3, 4).reshape(N_PATCH, PATCH_DIM)


def caption(shape, color, place):
    return f"a {color} {shape}, {place}."


chars = sorted(set("".join(caption(s, c, p) for s in SHAPES for c in COLORS for p in PLACES)) | {"|"})
stoi = {ch: i for i, ch in enumerate(chars)}
V, D, H, L = len(chars), 128, 4, 4
CAP = max(len(caption(s, c, p)) for s in SHAPES for c in COLORS for p in PLACES) + 1
T = N_PATCH + CAP


def example():
    s, c, p = random.choice(SHAPES), random.choice(list(COLORS)), random.choice(list(PLACES))
    text = ("|" + caption(s, c, p)).ljust(CAP, ".")
    return patches(draw(s, c, p)), torch.tensor([stoi[ch] for ch in text]), (s, c, p)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        B, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, D))
        return x + self.mlp(self.ln2(x))


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = nn.Linear(PATCH_DIM, D)          # image patch -> token vector (the "vision encoder")
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.Sequential(*[Block() for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)

    def forward(self, img_tokens, text):
        x = torch.cat([self.patch_proj(img_tokens), self.emb(text)], 1)   # one sequence: image, then text
        x = x + self.pos(torch.arange(x.shape[1]))
        return self.head(self.ln(self.blocks(x)))[:, N_PATCH:]            # predictions at the text positions


def batch(bs=32):
    ex = [example() for _ in range(bs)]
    return torch.stack([e[0] for e in ex]), torch.stack([e[1] for e in ex])


@torch.no_grad()
def describe(model, img_tokens):
    text = torch.tensor([[stoi["|"]]])
    for _ in range(CAP - 1):
        nxt = model(img_tokens[None], text)[:, -1].argmax(-1, keepdim=True)
        text = torch.cat([text, nxt], 1)
        if chars[nxt.item()] == "." and text.shape[1] > 6:
            break
    return "".join(chars[i] for i in text[0, 1:])


random.seed(0)
torch.manual_seed(0)
print(f"1. a {S} x {S} image = {N_PATCH} patches of {P} x {P} x 3 = {PATCH_DIM} numbers; each becomes one token of {D}")
model = TinyVLM()
print(f"   {sum(p.numel() for p in model.parameters()):,} parameters, of which {PATCH_DIM * D + D:,} in the patch "
      f"projection; sequence: {N_PATCH} image tokens + {CAP} text tokens")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(1, 1501):
    img, txt = batch()
    logits = model(img, txt[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, V), txt[:, 1:].reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step in (1, 250, 500, 1500):
        print(f"   step {step:>4}: caption loss {loss.item():.3f}")

model.eval()
random.seed(123)
right = {"shape": 0, "color": 0, "place": 0, "all": 0}
samples, confusions = [], {}
N = 200
for i in range(N):
    img, _, (s, c, p) = example()
    out = describe(model, img)
    ok = {"shape": s in out, "color": c in out, "place": p in out}
    for k, v in ok.items():
        right[k] += v
    right["all"] += all(ok.values())
    if i < 4:
        samples.append((caption(s, c, p), out))
    if not ok["shape"]:
        said = next((x for x in SHAPES if x in out), "?")
        confusions[(s, said)] = confusions.get((s, said), 0) + 1
print(f"\n2. {N} new images: shape right {right['shape'] / N:.0%}, color {right['color'] / N:.0%}, "
      f"position {right['place'] / N:.0%}, all three {right['all'] / N:.0%}")
for truth, out in samples:
    print(f"   true: {truth!r:<32} model: {out!r}")
print("   shape mistakes (true -> said): " + ", ".join(f"{a} -> {b} x{n}" for (a, b), n in
                                                   sorted(confusions.items(), key=lambda kv: -kv[1])))
