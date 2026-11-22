"""Episode 13, part 2: text-to-image latent diffusion on Shapes.

    TEXT=words  python3 latent_diffusion.py   # default: cross-attention over CLIP's word vectors → out/words_*, results_latent_words.json
    TEXT=pooled python3 latent_diffusion.py   # one averaged caption vector added to the time embedding → out/pooled_*
    EVAL=1 TEXT=words python3 latent_diffusion.py   # re-score a trained model without training again

Needs out/autoencoder.pt (train_autoencoder.py) and ../g12_clip/out/clip.pt (episode 12).
Diffusion runs on the autoencoder's 8×8×4 latents instead of 32×32×3 pixels, conditioned on episode 12's CLIP text
encoder (frozen). Two ways to feed the caption in:
  pooled: CLIP's single 64-number caption vector is added to the time embedding, like episode 10's labels.
  words:  the U-Net also gets cross-attention layers (at 8×8, 4×4 and 2×2) in which every latent position looks at
          CLIP's vectors for the individual words, as Stable Diffusion does. The pooled vector is still added too.
10% of captions are dropped during training, so classifier-free guidance works (episode 11). Episode 6's schedule and
loss, episode 7's fresh-noise sampler. Episode 10's eight held-out combinations are never seen by the autoencoder,
CLIP or this model.
"""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import diffusion as D
from clip_model import ALL, CAPTIONS, CLIP, tokenize
from judge import load_judge, to_tensor
from shapes import make_dataset, save_grid
from train_autoencoder import HELD_OUT, LATENT, Autoencoder

TEXT_DIM = 64


@torch.no_grad()
def text_features(clip):
    """For all 135 captions: CLIP's per-word vectors (N, 8, 64), the padding mask, and the pooled caption vector."""
    te = clip.text
    ids = tokenize(CAPTIONS)
    pad = ids == 0
    words = te.tf(te.tok(ids) + te.pos, src_key_padding_mask=pad)
    words = torch.nan_to_num(words) * ~pad[..., None]
    pooled = te(ids) * 8                                        # unit vector × √64: entries of size ~1
    return words, pad, pooled


class CrossAttention(nn.Module):
    """Every latent position (query) looks at the caption's words (keys and values)."""

    def __init__(self, c, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.attn = nn.MultiheadAttention(c, heads, kdim=TEXT_DIM, vdim=TEXT_DIM, batch_first=True)

    def forward(self, h, words, pad):
        n, c, hh, ww = h.shape
        q = self.norm(h).flatten(2).transpose(1, 2)               # (N, H·W, C)
        out, _ = self.attn(q, words, words, key_padding_mask=pad, need_weights=False)
        return h + out.transpose(1, 2).reshape(n, c, hh, ww)


class LatentUNet(D.UNet):
    """Episode 6's U-Net on 8×8×4 latents (8×8 → 4×4 → 2×2). cap: caption index into ALL, or -1 for "no caption"."""

    def __init__(self, words, pad, pooled, cross=True, ch=(64, 128, 128), t_dim=128):
        super().__init__(ch=ch, t_dim=t_dim)
        self.inp = nn.Conv2d(LATENT, ch[0], 3, padding=1)
        self.out[-1] = nn.Conv2d(ch[0], LATENT, 3, padding=1)
        self.text = nn.Linear(TEXT_DIM, t_dim)
        self.null_pooled = nn.Parameter(torch.zeros(TEXT_DIM))     # "no caption"
        self.null_word = nn.Parameter(torch.zeros(TEXT_DIM))
        self.register_buffer("words", words)
        self.register_buffer("pad", pad)
        self.register_buffer("pooled", pooled)
        self.cross = cross
        if cross:
            self.x1, self.x2, self.xm, self.xu2, self.xu1 = (CrossAttention(c) for c in (ch[0], ch[1], ch[2], ch[1], ch[0]))

    def caption(self, cap):
        none = cap < 0
        c = cap.clamp(min=0)
        pooled = torch.where(none[:, None], self.null_pooled.expand(len(cap), -1), self.pooled[c])
        words, pad = self.words[c].clone(), self.pad[c].clone()
        words[none] = 0
        words[none, 0] = self.null_word                          # no caption: a single learned "empty" word
        pad[none] = True
        pad[none, 0] = False
        return pooled, words, pad

    def attend(self, name, h, words, pad):
        return getattr(self, name)(h, words, pad) if self.cross else h

    def forward(self, x, t, cap):
        pooled, w, pad = self.caption(cap)
        temb = self.t_mlp(D.timestep_embedding(t, self.t_dim)) + self.text(pooled)
        h1 = self.attend("x1", self.down1(self.inp(x), temb), w, pad)
        h2 = self.attend("x2", self.down2(self.pool1(h1), temb), w, pad)
        h = self.mid2(self.attend("xm", self.mid1(self.pool2(h2), temb), w, pad), temb)
        h = self.attend("xu2", self.up2(torch.cat([self.upc2(h), h2], 1), temb), w, pad)
        h = self.attend("xu1", self.up1(torch.cat([self.upc1(h), h1], 1), temb), w, pad)
        return self.out(h)


@torch.no_grad()
def predict(model, x, t, cap, w):
    tt = torch.full((len(x),), t, dtype=torch.long)
    if w == 1:
        return model(x, tt, cap)
    both = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([cap, torch.full_like(cap, -1)]))
    cond, uncond = both.chunk(2)
    return uncond + w * (cond - uncond)


X0_CLIP = float(os.environ.get("X0_CLIP", 16))


@torch.no_grad()
def sample(model, cap, seed, w=1.0, steps=50, shape=(LATENT, 8, 8)):
    """Fresh-noise sampler (η = 1) from episode 7, in latent space.

    The clean-latent guess is clipped to ±16, only as a guard: real latents reach about ±12 (unit std overall, but the
    shape's own pixels are large). An earlier version clipped to ±4, like pixels' ±1 scaled up; clipping real
    latents at ±4 leaves only 46% of them decoding to the right picture (figures.py), and it capped every model at ~44%."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(cap), *shape, generator=g)
    ts = torch.linspace(D.T, 1, steps).round().long().tolist()
    for i, t in enumerate(ts):
        eps = predict(model, x, t, cap, w)
        ab = D.ALPHA_BAR[t - 1]
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-X0_CLIP, X0_CLIP)
        ab_s = D.ALPHA_BAR[ts[i + 1] - 1] if i + 1 < len(ts) else torch.tensor(1.0)
        sigma = ((1 - ab_s) / (1 - ab)).sqrt() * (1 - ab / ab_s).sqrt()
        x = ab_s.sqrt() * x0 + (1 - ab_s - sigma ** 2).clamp(min=0).sqrt() * eps
        if i + 1 < len(ts):
            x = x + sigma * torch.randn(x.shape, generator=g)
    return x


def load_models(mode="words"):
    ae = Autoencoder()
    ae.load_state_dict(torch.load("out/autoencoder.pt"))
    ae.eval()
    clip = CLIP()
    clip.load_state_dict(torch.load("../g12_clip/out/clip.pt"))
    clip.eval()
    model = LatentUNet(*text_features(clip), cross=(mode == "words"))
    return ae, clip, model


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("THREADS", min(8, os.cpu_count()))))
    MODE = os.environ.get("TEXT", "words")
    STEPS, BATCH, LR = int(os.environ.get("STEPS", 40_000 if MODE == "words" else 20_000)), 128, 2e-4
    ae, clip, model = load_models(MODE)
    images, _, labels = make_dataset(20_000, seed=0)
    keep = np.array([tuple(l) not in set(HELD_OUT) for l in labels.tolist()])
    x_all = to_tensor(images[keep])
    cap_idx = torch.tensor([ALL.index(tuple(l)) for l in labels[keep].tolist()])
    t0 = time.time()
    with torch.no_grad():
        z_all = torch.cat([ae.encode(x_all[i:i + 1000])[0] for i in range(0, len(x_all), 1000)])
    print(f"encoded {len(z_all):,} pictures to {tuple(z_all.shape[1:])} latents in {time.time() - t0:.0f} s "
          f"(latent std {z_all.std():.2f})", flush=True)

    EVAL_ONLY = os.environ.get("EVAL") == "1"     # re-score out/{MODE}_latent_unet.pt without training
    if EVAL_ONLY:
        STEPS = 0
        prev = json.load(open(f"results_latent_{MODE}.json"))
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    params = sum(p.numel() for p in model.parameters())
    print(f"{MODE}: latent U-Net with {params:,} parameters", flush=True)

    # cost per training step, latent vs pixel U-Net (episode 6's), same batch
    def time_step(net, x, extra=(), reps=10):
        o = torch.optim.Adam(net.parameters(), lr=0)
        t = torch.randint(1, D.T + 1, (len(x),))
        for k in range(reps + 2):
            if k == 2:
                s = time.time()
            loss = F.mse_loss(net(x, t, *extra), torch.randn_like(x))
            o.zero_grad()
            loss.backward()
            o.step()
        return (time.time() - s) / reps * 1000
    pixel = D.UNet(ch=(32, 64, 128))
    cost = prev["cost"] if EVAL_ONLY else {"pixel_unet_params": sum(p.numel() for p in pixel.parameters()), "latent_unet_params": params,
            "pixel_ms_per_step": time_step(pixel, torch.randn(BATCH, 3, 32, 32)),
            "latent_ms_per_step": time_step(copy.deepcopy(model), torch.randn(BATCH, LATENT, 8, 8),
                                            (torch.randint(0, len(ALL), (BATCH,)),))}
    print(cost, flush=True)

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
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.lerp_(p, 1 - 0.999)
        log.append(loss.item())
        if step % 2000 == 0 or step == STEPS:
            print(f"step {step:5d}: loss {np.mean(log[-200:]):.4f}  {(time.time() - t0) / 60:.0f} min", flush=True)
            torch.save(ema.state_dict(), f"out/{MODE}_latent_unet.pt")
    train_minutes = (time.time() - t0) / 60
    if EVAL_ONLY:
        ema.load_state_dict(torch.load(f"out/{MODE}_latent_unet.pt"))
        STEPS, train_minutes, log = prev["steps"], prev["train_minutes"], []

    # evaluation: follow the caption?
    judge = load_judge()

    @torch.no_grad()
    def evaluate(combos, per, seed, w):
        cap = torch.tensor([ALL.index(c) for c in combos for _ in range(per)])
        s = time.time()
        z = sample(ema, cap, seed, w)
        x = torch.cat([ae.decode(z[i:i + 500]) for i in range(0, len(z), 500)])
        secs = time.time() - s
        probs = [l.softmax(-1) for l in judge(x)]
        ok = torch.stack([p.max(-1).values >= 0.9 for p in probs[:3]] + [probs[3][:, 1] >= 0.5], 1).all(1)
        lab = torch.stack([p.argmax(-1) for p in probs[:3]], 1)
        want = torch.tensor([c for c in combos for _ in range(per)])
        right = ((lab == want) & ok[:, None]).float().mean(0).tolist()
        return x, {"recognizable": ok.float().mean().item(), "follows": ((lab == want).all(1) & ok).float().mean().item(),
                   "shape": right[0], "color": right[1], "position": right[2], "ms_per_picture": secs / len(x) * 1000}

    trained = [c for c in ALL if c not in set(HELD_OUT)]
    res = {"text": MODE, "parameters": params, "steps": STEPS, "train_minutes": train_minutes, "cost": cost,
           "loss_curve": prev["loss_curve"] if EVAL_ONLY else [float(np.mean(log[i:i + 200])) for i in range(0, STEPS, 200)],
           "x0_clip": X0_CLIP, "eval": {}}
    for w in (1, 3):
        x, r = evaluate(trained, 4, 1, w)
        xh, rh = evaluate(HELD_OUT, 32, 2, w)
        res["eval"][f"w{w}"] = {"trained": r, "held_out": rh}
        print(f"w = {w}: trained {r}  held-out {rh}", flush=True)
        save_grid((x[:16] * 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8), f"out/{MODE}_samples_w{w}.png")
        save_grid((torch.cat([xh[k * 32:k * 32 + 8] for k in range(4)]) * 255).round().permute(0, 2, 3, 1).numpy()
                  .astype(np.uint8), f"out/{MODE}_held_out_w{w}.png")
    # prompts for the video: one row of 8 per caption
    prompts = ["a red triangle in the center", "a blue square at the top left", "a yellow circle at the bottom",
               "a purple triangle at the top left"]
    cap = torch.tensor([CAPTIONS.index(p) for p in prompts for _ in range(8)])
    with torch.no_grad():
        x = ae.decode(sample(ema, cap, 5, 3))
    save_grid((x * 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8), f"out/{MODE}_prompts.png")
    res["prompts"] = prompts
    with open(f"results_latent_{MODE}.json", "w") as f:
        json.dump(res, f, indent=1)
