"""Episode 5: PPO, small safe steps.

Same task and starting model as episodes 2-4 (episode 1's base.pt, 32 questions x 8 tries per batch). Every learner
gets the same 400 batches (102,400 sampled answers):

  reinforce  one update per batch, per-question baseline (episode 3)
  ppo        each batch reused for 4 epochs x 4 minibatches (16 updates); per-token probability ratio
             new/old clipped to [0.8, 1.2]; a value head (one linear layer on the '=' position) learns the baseline
  noclip     the same 16 updates per batch with the ratio but without the clip

Runs (3 seeds each): reinforce at learning rates 0.0003 (episode 2's) and 0.001; ppo and noclip at 0.0003 (the same
rate as REINFORCE) and 0.0001. Writes results.json. About 8 minutes on 16 threads:  python ppo.py
"""
import json
import random
import time
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F

from arith import BLOCK, Q_LEN, TinyGPT, accuracy, reward, sample, split

BATCHES, QUESTIONS, TRIES, LR = 400, 32, 8, 3e-4    # LR: default when a job gives none
EPOCHS, MINIBATCHES, CLIP, VF_COEF = 4, 4, 0.2, 0.5
BASE = "../r01_learning_from_a_grade/base.pt"
SEEDS = (1, 2, 3)


class PolicyValue(nn.Module):
    """tiny_gpt plus a value head: one linear layer reading the hidden state at the '=' (the question's end)."""

    def __init__(self):
        super().__init__()
        self.gpt = TinyGPT()
        self.gpt.load_state_dict(torch.load(BASE))
        self.value = nn.Linear(64, 1)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, idx):
        g = self.gpt
        h = g.ln(g.blocks(g.tok(idx) + g.pos(torch.arange(idx.shape[1]))))
        return g.head(h), self.value(h[:, Q_LEN - 1, :])[:, 0]


def token_logps(logits, idx, lengths):
    """Per-token log-probabilities of the answer tokens and their mask, shape [B, BLOCK-1]."""
    lp = F.log_softmax(logits, -1).gather(-1, idx[:, 1:, None])[..., 0]
    pos = torch.arange(BLOCK - 1)[None]
    mask = ((pos >= Q_LEN - 1) & (pos < Q_LEN - 1 + lengths[:, None])).float()
    return lp, mask


def batch(model, train, rng):
    qs = rng.sample(train, QUESTIONS)
    idx, answers, lengths, flat = sample(model.gpt, qs, n=TRIES)
    r = torch.tensor([reward(a, b, x) for (a, b), x in zip(flat, answers)])
    return idx, lengths, r


def train_run(job):
    method, seed = job[:2]
    lr = job[2] if len(job) > 2 else LR
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train, test = split()
    model = PolicyValue()
    opt = torch.optim.Adam(model.parameters(), lr)
    curve, stats = [], []
    t0 = time.time()
    for b in range(BATCHES + 1):
        if b % 20 == 0:
            model.eval()
            curve.append([b, accuracy(model.gpt, test)])
            model.train()
        if b == BATCHES:
            break
        idx, lengths, r = batch(model, train, rng)
        if method == "reinforce":
            logits, _ = model(idx[:, :-1])
            lp, mask = token_logps(logits, idx, lengths)
            g = r.view(QUESTIONS, TRIES)
            adv = (g - g.mean(1, keepdim=True)).view(-1)
            loss = -(adv * (lp * mask).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            continue
        with torch.no_grad():
            logits, v = model(idx[:, :-1])
            old, mask = token_logps(logits, idx, lengths)
            adv = r - v                                      # the value head is the baseline
        n = len(r)
        ratios, clipped, kls, vls = [], [], [], []
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for mb in perm.chunk(MINIBATCHES):
                logits, v = model(idx[mb, :-1])
                lp, m = token_logps(logits, idx[mb], lengths[mb])
                ratio = torch.exp(lp - old[mb]) * m + (1 - m)
                a = adv[mb, None]
                if method == "ppo":
                    surr = torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a)
                else:
                    surr = ratio * a
                pg = -(surr * m).sum() / m.sum()
                vloss = F.mse_loss(v, r[mb])
                loss = pg + VF_COEF * vloss
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    sel = m.bool()
                    ratios.append(ratio[sel].max().item())
                    clipped.append(((ratio[sel] - 1).abs() > CLIP).float().mean().item())
                    kls.append(((old[mb] - lp) * m).sum().item() / m.sum().item())
                    vls.append(vloss.item())
        stats.append({"batch": b, "max_ratio": max(ratios), "clip_frac": sum(clipped) / len(clipped),
                      "kl_old": sum(kls) / len(kls), "value_loss": sum(vls) / len(vls), "mean_reward": r.mean().item(),
                      "mean_value": v.mean().item()})
    out = {"method": method, "seed": seed, "lr": lr, "curve": curve, "stats": stats, "seconds": round(time.time() - t0, 1)}
    print(method, seed, lr, curve[-1], f"{out['seconds']}s", flush=True)
    return out


RUNS = [("reinforce", 3e-4), ("reinforce", 1e-3), ("ppo", 3e-4), ("noclip", 3e-4), ("ppo", 1e-4), ("noclip", 1e-4)]


def main():
    jobs = [(m, s, lr) for m, lr in RUNS for s in SEEDS]
    t0 = time.time()
    with Pool(8) as pool:
        runs = pool.map(train_run, jobs, chunksize=1)
    json.dump({"batches": BATCHES, "questions": QUESTIONS, "tries": TRIES, "epochs": EPOCHS,
               "minibatches": MINIBATCHES, "clip": CLIP, "seconds": round(time.time() - t0, 1), "runs": runs},
              open("results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
