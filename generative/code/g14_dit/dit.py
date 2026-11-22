"""Episode 14: a Diffusion Transformer (DiT) in place of episode 13's latent U-Net.

    PATCH=2 python3 dit.py     # 16 tokens of 2×2×4 latent numbers → out/dit_p2_*, results_dit_p2.json
    PATCH=1 python3 dit.py     # 64 tokens of 1×1×4 → out/dit_p1_*, results_dit_p1.json

Same data, autoencoder (../g13_latent_diffusion/out/autoencoder.pt), frozen CLIP text encoder, schedule, loss,
sampler, caption dropout and number of training steps (40,000) as episode 13's cross-attention U-Net; only the
denoiser changes. The DiT (Peebles & Xie 2023):
  - patchify: cut the 8×8×4 latent into PATCH×PATCH squares, each flattened and linearly mapped to a token;
    add a learned position embedding;
  - N transformer blocks: self-attention over the tokens, cross-attention to CLIP's word vectors (as in PixArt-α
    and episode 13), an MLP; time and the pooled caption vector steer every block through adaLN-Zero: they set the
    scale and shift of each LayerNorm and a gate on each residual branch, the gates starting at zero;
  - a final adaLN + linear layer turns each token back into its patch of predicted noise.
"""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("AE_PATH", "../g13_latent_diffusion/out/autoencoder.pt")
import diffusion as D                                                        # noqa: E402
import latent_diffusion as L                                                 # noqa: E402
from clip_model import ALL, CAPTIONS                                         # noqa: E402
from judge import load_judge, to_tensor                                      # noqa: E402
from shapes import make_dataset, save_grid                                   # noqa: E402
from train_autoencoder import HELD_OUT, LATENT                               # noqa: E402

PATCH = int(os.environ.get("PATCH", 2))
DIM, DEPTH, HEADS = int(os.environ.get("DIM", 128)), int(os.environ.get("DEPTH", 8)), 4


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.nx = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, kdim=L.TEXT_DIM, vdim=L.TEXT_DIM, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)                                     # adaLN-Zero: every block starts as
        nn.init.zeros_(self.ada[1].bias)                                       # the identity

    def forward(self, x, c, words, pad):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=1)
        h = modulate(self.n1(x), s1, sc1)
        x = x + g1[:, None] * self.attn(h, h, h, need_weights=False)[0]
        x = x + self.cross(self.nx(x), words, words, key_padding_mask=pad, need_weights=False)[0]
        x = x + g2[:, None] * self.mlp(modulate(self.n2(x), s2, sc2))
        return x


class DiT(nn.Module):
    def __init__(self, words, pad, pooled, patch=PATCH, d=DIM, depth=DEPTH, heads=HEADS):
        super().__init__()
        self.patch, self.side = patch, 8 // patch
        self.tokens = self.side ** 2
        self.embed = nn.Linear(LATENT * patch * patch, d)
        self.pos = nn.Parameter(torch.randn(1, self.tokens, d) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.text = nn.Linear(L.TEXT_DIM, d)
        self.null_pooled = nn.Parameter(torch.zeros(L.TEXT_DIM))
        self.null_word = nn.Parameter(torch.zeros(L.TEXT_DIM))
        self.register_buffer("words", words)
        self.register_buffer("pad", pad)
        self.register_buffer("pooled", pooled)
        self.blocks = nn.ModuleList([DiTBlock(d, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.final = nn.Linear(d, LATENT * patch * patch)
        for m in (self.final_ada[1], self.final):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.d = d

    caption = L.LatentUNet.caption                                             # same "no caption" handling as episode 13

    def patchify(self, x):                                                     # (N, 4, 8, 8) → (N, tokens, 4·p·p)
        n, p, s = len(x), self.patch, self.side
        return x.reshape(n, LATENT, s, p, s, p).permute(0, 2, 4, 1, 3, 5).reshape(n, s * s, LATENT * p * p)

    def unpatchify(self, t):
        n, p, s = len(t), self.patch, self.side
        return t.reshape(n, s, s, LATENT, p, p).permute(0, 3, 1, 4, 2, 5).reshape(n, LATENT, 8, 8)

    def forward(self, x, t, cap):
        pooled, words, pad = self.caption(cap)
        c = self.t_mlp(D.timestep_embedding(t, self.d)) + self.text(pooled)
        h = self.embed(self.patchify(x)) + self.pos
        for b in self.blocks:
            h = b(h, c, words, pad)
        shift, scale = self.final_ada(c).chunk(2, dim=1)
        return self.unpatchify(self.final(modulate(self.final_norm(h), shift, scale)))


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("THREADS", min(8, os.cpu_count()))))
    NAME = f"dit_p{PATCH}" + os.environ.get("SUFFIX", "")
    STEPS, BATCH, LR = int(os.environ.get("STEPS", 40_000)), 128, 2e-4
    ae, clip, _ = L.load_models("words")
    model = DiT(*L.text_features(clip))
    params = sum(p.numel() for p in model.parameters())
    print(f"{NAME}: {model.tokens} tokens of {LATENT * PATCH * PATCH} numbers, width {DIM}, depth {DEPTH}, "
          f"{params:,} parameters", flush=True)

    images, _, labels = make_dataset(20_000, seed=0)
    keep = np.array([tuple(l) not in set(HELD_OUT) for l in labels.tolist()])
    x_all = to_tensor(images[keep])
    cap_idx = torch.tensor([ALL.index(tuple(l)) for l in labels[keep].tolist()])
    with torch.no_grad():
        z_all = torch.cat([ae.encode(x_all[i:i + 1000])[0] for i in range(0, len(x_all), 1000)])

    # cost of one training step next to episode 13's cross-attention U-Net, same batch, same moment
    def time_step(net, reps=10):
        o = torch.optim.Adam(net.parameters(), lr=0)
        x, t, c = torch.randn(BATCH, LATENT, 8, 8), torch.randint(1, D.T + 1, (BATCH,)), torch.randint(0, len(ALL), (BATCH,))
        for k in range(reps + 2):
            if k == 2:
                s = time.time()
            loss = F.mse_loss(net(x, t, c), torch.randn_like(x))
            o.zero_grad()
            loss.backward()
            o.step()
        return (time.time() - s) / reps * 1000
    unet = L.LatentUNet(*L.text_features(clip), cross=True)
    cost = {"dit_params": params, "unet_params": sum(p.numel() for p in unet.parameters()),
            "dit_ms_per_step": time_step(copy.deepcopy(model)), "unet_ms_per_step": time_step(unet)}
    print(cost, flush=True)

    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
    warm = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000))
    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(z_all), (BATCH,))
        z0, cap = z_all[idx], cap_idx[idx].clone()
        cap[torch.rand(BATCH) < 0.1] = -1
        t = torch.randint(1, D.T + 1, (BATCH,))
        eps = torch.randn_like(z0)
        loss = F.mse_loss(model(D.noise(z0, t, eps), t, cap), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        warm.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.lerp_(p, 1 - 0.999)
        log.append(loss.item())
        if step % 2000 == 0 or step == STEPS:
            print(f"step {step:5d}: loss {np.mean(log[-200:]):.4f}  {(time.time() - t0) / 60:.0f} min", flush=True)
            torch.save(ema.state_dict(), f"out/{NAME}.pt")
    train_minutes = (time.time() - t0) / 60

    judge = load_judge()

    @torch.no_grad()
    def evaluate(combos, per, seed, w):
        cap = torch.tensor([ALL.index(c) for c in combos for _ in range(per)])
        z = L.sample(ema, cap, seed, w)
        x = torch.cat([ae.decode(z[i:i + 500]) for i in range(0, len(z), 500)])
        probs = [l.softmax(-1) for l in judge(x)]
        ok = torch.stack([p.max(-1).values >= 0.9 for p in probs[:3]] + [probs[3][:, 1] >= 0.5], 1).all(1)
        lab = torch.stack([p.argmax(-1) for p in probs[:3]], 1)
        want = torch.tensor([c for c in combos for _ in range(per)])
        return x, {"recognizable": ok.float().mean().item(), "follows": ((lab == want).all(1) & ok).float().mean().item()}

    trained = [c for c in ALL if c not in set(HELD_OUT)]
    res = {"name": NAME, "patch": PATCH, "tokens": model.tokens, "width": DIM, "depth": DEPTH, "parameters": params,
           "steps": STEPS, "train_minutes": train_minutes, "cost": cost,
           "loss_curve": [float(np.mean(log[i:i + 200])) for i in range(0, STEPS, 200)], "eval": {}}
    for w in (1, 3):
        x, r = evaluate(trained, 4, 1, w)                                       # same seeds as episode 13
        xh, rh = evaluate(HELD_OUT, 32, 2, w)
        res["eval"][f"w{w}"] = {"trained": r, "held_out": rh}
        print(f"w = {w}: trained {r}  held-out {rh}", flush=True)
        save_grid((x[:16] * 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8), f"out/{NAME}_samples_w{w}.png")
        save_grid((torch.cat([xh[k * 32:k * 32 + 8] for k in range(4)]) * 255).round().permute(0, 2, 3, 1).numpy()
                  .astype(np.uint8), f"out/{NAME}_held_out_w{w}.png")
    prompts = ["a red triangle in the center", "a blue square at the top left", "a yellow circle at the bottom",
               "a purple triangle at the top left"]
    cap = torch.tensor([CAPTIONS.index(p) for p in prompts for _ in range(8)])
    with torch.no_grad():
        x = ae.decode(L.sample(ema, cap, 5, 3))
    save_grid((x * 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8), f"out/{NAME}_prompts.png")
    res["prompts"] = prompts
    with open(f"results_{NAME}.json", "w") as f:
        json.dump(res, f, indent=1)
