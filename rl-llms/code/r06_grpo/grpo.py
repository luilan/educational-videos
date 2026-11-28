"""Episode 6: GRPO, grading on a curve.

GRPO (group relative policy optimization, DeepSeekMath 2024) keeps PPO's ratio and clip but drops the value model:
each question gets a group of G answers, and an answer's advantage is how it did compared with the rest of its group,

    advantage = (reward - group mean) / (group standard deviation + 0.0001)

plus a KL penalty to the frozen starting model in the loss (beta 0.04, the "k3" estimator).

Same task, starting model and budget as episode 5 (400 batches of 256 answers, 4 epochs x 4 minibatches per batch,
step 0.0001, 3 seeds). Runs: PPO with a value head (episode 5's code), GRPO with groups of 2, 4, 8 and 16 answers
(256/G questions per batch), and GRPO with G = 8 without dividing by the standard deviation.
Then `python grpo.py extra` adds two G = 8 ablations: no KL term, and DAPO-style dynamic sampling (sample more
questions and keep only groups whose rewards differ, until the batch has 256 answers; counts every answer generated).
Writes results.json. About 11 + 8 minutes (4 processes):  python grpo.py; python grpo.py extra
"""
import json
import random
import time
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F

from arith import BLOCK, Q_LEN, TinyGPT, accuracy, reward, sample, split

BATCHES, ANSWERS, LR, EPOCHS, MINIBATCHES, CLIP, BETA, VF_COEF = 400, 256, 1e-4, 4, 4, 0.2, 0.04, 0.5
BASE = "../r01_learning_from_a_grade/base.pt"
SEEDS = (1, 2, 3)


class PolicyValue(nn.Module):
    """tiny_gpt plus a value head on the '=' position (used only by PPO)."""

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
    lp = F.log_softmax(logits, -1).gather(-1, idx[:, 1:, None])[..., 0]
    pos = torch.arange(BLOCK - 1)[None]
    mask = ((pos >= Q_LEN - 1) & (pos < Q_LEN - 1 + lengths[:, None])).float()
    return lp, mask


def train_run(job):
    method, group, seed = job
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train, test = split()
    model = PolicyValue()
    ref = TinyGPT()
    ref.load_state_dict(torch.load(BASE))
    ref.eval()
    n_params = sum(p.numel() for p in model.gpt.parameters())
    n_value = sum(p.numel() for p in model.value.parameters())
    params = list(model.gpt.parameters()) + (list(model.value.parameters()) if method == "ppo" else [])
    opt = torch.optim.Adam(params, LR)
    questions = ANSWERS // group
    curve, stats = [], []
    generated = 0
    t0 = time.time()
    for b in range(BATCHES + 1):
        if b % 20 == 0:
            model.eval()
            curve.append([b, accuracy(model.gpt, test)])
            model.train()
        if b == BATCHES:
            break
        if method == "grpo_dyn":                  # DAPO-style dynamic sampling: keep only groups whose rewards differ
            keep_idx, keep_len, keep_r = [], [], []
            while sum(len(x) for x in keep_r) < ANSWERS:
                qs = rng.sample(train, questions)
                idx, answers, lengths, flat = sample(model.gpt, qs, n=group)
                generated += len(answers)
                r = torch.tensor([reward(a, bb, x) for (a, bb), x in zip(flat, answers)]).view(questions, group)
                ok = r.std(1) > 0
                rows = ok.repeat_interleave(group)
                keep_idx.append(idx[rows]); keep_len.append(lengths[rows]); keep_r.append(r[ok].view(-1))
            idx, lengths, r = torch.cat(keep_idx)[:ANSWERS], torch.cat(keep_len)[:ANSWERS], torch.cat(keep_r)[:ANSWERS]
        else:
            qs = rng.sample(train, questions)
            idx, answers, lengths, flat = sample(model.gpt, qs, n=group)
            generated += len(answers)
            r = torch.tensor([reward(a, bb, x) for (a, bb), x in zip(flat, answers)])
        with torch.no_grad():
            logits, v = model(idx[:, :-1])
            old, mask = token_logps(logits, idx, lengths)
            ref_lp, _ = token_logps(ref(idx[:, :-1]), idx, lengths)
            if method == "ppo":
                adv = r - v
            else:
                g = r.view(questions, group)
                adv = g - g.mean(1, keepdim=True)
                if method != "grpo_nostd":
                    adv = adv / (g.std(1, keepdim=True) + 1e-4)
                adv = adv.view(-1)
        zero_groups = None if method == "ppo" else (r.view(questions, group).std(1) == 0).float().mean().item()
        for _ in range(EPOCHS):
            for mb in torch.randperm(len(r)).chunk(MINIBATCHES):
                logits, v = model(idx[mb, :-1])
                lp, m = token_logps(logits, idx[mb], lengths[mb])
                ratio = torch.exp(lp - old[mb])
                a = adv[mb, None]
                surr = torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a)
                if method == "ppo":
                    loss = -(surr * m).sum() / m.sum() + VF_COEF * F.mse_loss(v, r[mb])
                else:
                    d = ref_lp[mb] - lp                                   # k3 estimator of KL(model || reference)
                    kl = torch.exp(d) - d - 1
                    beta = 0.0 if method == "grpo_nokl" else BETA
                    loss = -((surr - beta * kl) * m).sum() / m.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        stats.append({"batch": b, "mean_reward": r.mean().item(), "zero_adv_groups": zero_groups, "generated": generated})
    out = {"method": method, "group": group, "seed": seed, "curve": curve, "stats": stats,
           "seconds": round(time.time() - t0, 1), "generated": generated, "policy_params": n_params,
           "value_params": n_value if method == "ppo" else 0}
    print(method, group, seed, curve[-1], f"{out['seconds']}s", flush=True)
    return out


def main(extra=False):
    if extra:
        jobs = [(m, 8, s) for m in ("grpo_nokl", "grpo_dyn") for s in SEEDS]
    else:
        jobs = [("ppo", 8, s) for s in SEEDS] + [("grpo", g, s) for g in (2, 4, 8, 16) for s in SEEDS] + \
               [("grpo_nostd", 8, s) for s in SEEDS]
    t0 = time.time()
    with Pool(4) as pool:
        runs = pool.map(train_run, jobs, chunksize=1)
    if extra:
        res = json.load(open("results.json"))
        res["runs"] += runs
        res["seconds_extra"] = round(time.time() - t0, 1)
    else:
        res = {"batches": BATCHES, "answers_per_batch": ANSWERS, "lr": LR, "epochs": EPOCHS, "minibatches": MINIBATCHES,
               "clip": CLIP, "beta": BETA, "seconds": round(time.time() - t0, 1), "runs": runs}
    json.dump(res, open("results.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    main(extra=sys.argv[1:] == ["extra"])
