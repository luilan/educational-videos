"""Episode 12: a tiny CLIP trained on Shapes pictures and their captions.

    python3 train_clip.py      # ~15 minutes on a 4-core CPU → out/clip.pt, results.json, out/*.png

An image encoder (a small CNN) and a text encoder (a 2-layer transformer over the caption's words) each map their
input to a 64-number vector. Contrastive training (Radford et al. 2021): in a batch of N picture-caption pairs, the
N matching pairs should have the highest similarity, the N² − N mismatched pairs low. The loss is cross-entropy over
each row (picture → which caption?) and each column (caption → which picture?) of the N × N similarity table.
Eight shape-color-position combinations (episode 10's) are left out of training, to test captions never seen.
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from judge import to_tensor
from shapes import COLOR_NAMES, POSITIONS, SHAPES, caption, make_dataset, save_grid

HELD_OUT = [(2, 4, 0), (0, 3, 8), (1, 0, 4), (2, 1, 6), (0, 2, 2), (1, 4, 7), (2, 3, 5), (0, 0, 3)]
ALL = [(s, c, p) for s in range(3) for c in range(5) for p in range(9)]
CAPTIONS = [caption(SHAPES[s], COLOR_NAMES[c], p) for s, c, p in ALL]
VOCAB = ["<pad>"] + sorted({w for cap in CAPTIONS for w in cap.split()})
MAXLEN = 8
DIM = 64


def tokenize(caps):
    ids = torch.zeros(len(caps), MAXLEN, dtype=torch.long)
    for i, cap in enumerate(caps):
        w = [VOCAB.index(x) for x in cap.split()]
        ids[i, :len(w)] = torch.tensor(w)
    return ids


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),       # 16×16
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),      # 8×8
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, DIM))

    def forward(self, x):
        return F.normalize(self.body(x), dim=-1)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), DIM)
        self.pos = nn.Parameter(torch.randn(MAXLEN, DIM) * 0.02)
        layer = nn.TransformerEncoderLayer(DIM, 4, 128, dropout=0.0, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(DIM, DIM)

    def forward(self, ids, normalize=True):
        pad = ids == 0
        h = self.tf(self.tok(ids) + self.pos, src_key_padding_mask=pad)
        h = (h * ~pad[..., None]).sum(1) / (~pad).sum(1, keepdim=True)       # average over the real words
        h = self.out(h)
        return F.normalize(h, dim=-1) if normalize else h


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.image, self.text = ImageEncoder(), TextEncoder()
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))      # temperature, learned

    def forward(self, x, ids):
        return self.log_scale.exp() * self.image(x) @ self.text(ids).T


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("THREADS", min(8, os.cpu_count()))))
    os.makedirs("out", exist_ok=True)
    STEPS, BATCH, LR = int(os.environ.get("STEPS", 4000)), 128, 1e-3
    images, _, labels = make_dataset(20_000, seed=0)
    keep = np.array([tuple(l) not in set(HELD_OUT) for l in labels.tolist()])
    x_all = to_tensor(images[keep])
    c_all = torch.tensor([ALL.index(tuple(l)) for l in labels[keep].tolist()])
    test_img, _, test_lab = make_dataset(2000, seed=99)
    x_test = to_tensor(test_img)
    c_test = torch.tensor([ALL.index(tuple(l)) for l in test_lab.tolist()])
    held_test = torch.tensor([tuple(l) in set(HELD_OUT) for l in test_lab.tolist()])
    # extra test pictures of the eight held-out combinations only (25 each)
    rng = np.random.default_rng(5)
    from shapes import draw
    x_held = to_tensor(np.stack([draw(SHAPES[s], COLOR_NAMES[c], p, rng) for s, c, p in HELD_OUT for _ in range(25)]))
    c_held = torch.tensor([ALL.index(h) for h in HELD_OUT for _ in range(25)])
    all_ids = tokenize(CAPTIONS)
    model = CLIP()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    params = {"image": sum(p.numel() for p in model.image.parameters()),
              "text": sum(p.numel() for p in model.text.parameters())}
    print(f"vocabulary {len(VOCAB)} words; parameters {params}", flush=True)

    @torch.no_grad()
    def evaluate():
        """Zero-shot reading: for each test picture, which of all 135 captions is most similar?"""
        model.eval()
        t = model.text(all_ids)
        out = {}
        for name, x, c in (("trained", x_test[~held_test], c_test[~held_test]), ("held_out", x_held, c_held)):
            pred = (model.image(x) @ t.T).argmax(1)
            P, C = torch.tensor([ALL[i] for i in pred.tolist()]), torch.tensor([ALL[i] for i in c.tolist()])
            out[name] = {"exact": (pred == c).float().mean().item(),
                         **{k: (P[:, j] == C[:, j]).float().mean().item() for j, k in enumerate(("shape", "color", "position"))}}
        # caption → picture: the 5 most similar test pictures for each trained caption; how many match?
        # (every caption has at least 5 matching test pictures)
        sim = t @ model.image(x_test[~held_test]).T
        top = sim.topk(5, dim=1).indices
        hits = (c_test[~held_test][top] == torch.arange(len(ALL))[:, None]).float().mean(1)
        trained_caps = torch.tensor([a not in set(HELD_OUT) for a in ALL])
        out["caption_to_picture_top5"] = hits[trained_caps].mean().item()
        model.train()
        return out

    CHECK = [0, 10, 20, 30, 40, 50, 75, 100, 150, 250, 500, 1000, 2000, STEPS]
    history, log, t0 = [], [], time.time()
    for step in range(0, STEPS + 1):
        if step in CHECK:
            h = {"step": step, "loss_last_100": float(np.mean(log[-100:])) if log else None, **evaluate(),
                 "temperature": 1 / model.log_scale.exp().item(), "minutes": (time.time() - t0) / 60}
            history.append(h)
            print(f"step {step:4d}: loss {h['loss_last_100'] or float('nan'):.3f}  trained {h['trained']['exact']:.1%}  "
                  f"(shape {h['trained']['shape']:.0%} color {h['trained']['color']:.0%} position {h['trained']['position']:.0%})  "
                  f"held-out {h['held_out']['exact']:.1%}  caption→picture top-5 {h['caption_to_picture_top5']:.1%}", flush=True)
        if step == STEPS:
            break
        idx = torch.randint(0, len(x_all), (BATCH,))
        logits = model(x_all[idx], tokenize([CAPTIONS[i] for i in c_all[idx].tolist()]))
        target = torch.arange(BATCH)
        loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.log_scale.clamp_(0, math.log(100))
        log.append(loss.item())
    torch.save(model.state_dict(), "out/clip.pt")

    model.eval()
    with torch.no_grad():
        # the similarity table for 8 test pictures and their 8 captions (for the video)
        pick = torch.arange(8)
        sim8 = (model.image(x_test[pick]) @ model.text(tokenize([CAPTIONS[i] for i in c_test[pick].tolist()])).T)
        save_grid(test_img[:8], "out/table_pictures.png")
        # the captions closest to one caption, and the held-out combinations read wrong
        t = model.text(all_ids)
        q = CAPTIONS.index("a red circle at the top left")
        near = (t @ t[q]).topk(6)
        pred = (model.image(x_held) @ t.T).argmax(1)
        wrong = [{"true": CAPTIONS[c], "read": CAPTIONS[p]} for c, p in zip(c_held.tolist(), pred.tolist()) if c != p]
        # caption → picture search: top 8 test pictures for two captions (one trained, one held out)
        for name, cap in (("search_trained", "a green square at the bottom left"),
                          ("search_held_out", "a purple triangle at the top left")):
            pool = torch.cat([x_test, x_held])
            top = (model.text(tokenize([cap])) @ model.image(pool).T)[0].topk(8).indices
            save_grid((pool[top].permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8), f"out/{name}.png")

    with open("results.json", "w") as f:
        json.dump({"vocabulary": VOCAB, "parameters": params, "steps": STEPS, "batch": BATCH, "lr": LR,
                   "dim": DIM, "held_out": [CAPTIONS[ALL.index(h)] for h in HELD_OUT],
                   "loss_curve": [float(np.mean(log[i:i + 50])) for i in range(0, STEPS, 50)], "history": history,
                   "table_captions": [CAPTIONS[i] for i in c_test[:8].tolist()], "table": sim8.tolist(),
                   "nearest_to_red_circle_top_left": [{"caption": CAPTIONS[i], "similarity": v}
                                                      for v, i in zip(near.values.tolist(), near.indices.tolist())],
                   "held_out_wrong": wrong, "minutes": (time.time() - t0) / 60}, f, indent=1)
    print(json.dumps(history[-1], indent=1))
