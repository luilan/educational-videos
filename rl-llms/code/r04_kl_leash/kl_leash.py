"""Episode 4: The KL Leash.

REINFORCE with the per-question baseline (episode 3), from episode 1's model, plus a KL penalty that keeps the
model close to a frozen copy of where it started (the reference). Each sampled answer's reward becomes

    shaped = reward - beta * (log p_model(answer) - log p_reference(answer))

the classic RLHF recipe. Part 1 sweeps beta at the normal learning rate and measures accuracy, the KL to the
reference, and how varied the answers are. Part 2 repeats two of them at a 10x learning rate, where the run without
a leash collapses. Writes results.json. About 25 minutes on a CPU:  python kl_leash.py
"""
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from arith import TinyGPT, accuracy, answer_logprob, reward, sample, split

torch.set_num_threads(8)
STEPS, QUESTIONS, TRIES, SEEDS = 2000, 32, 8, (1, 2)
BETAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
BASE = "../r01_learning_from_a_grade/base.pt"
SHOW = [(47, 38), (19, 26), (73, 58)]          # questions whose sampled answers we print as we go


def load():
    m = TinyGPT()
    m.load_state_dict(torch.load(BASE))
    return m


@torch.no_grad()
def measure(model, ref, test, n_q=200, n=16, seed=0):
    """Sampled-answer statistics at temperature 1 on n_q test questions, n tries each."""
    g = torch.Generator().manual_seed(seed)
    qs = test[:n_q]
    idx, answers, lengths, flat = sample(model, qs, n=n, generator=g)
    lp, lr_ = answer_logprob(model, idx, lengths), answer_logprob(ref, idx, lengths)
    right = [reward(a, b, x) for (a, b), x in zip(flat, answers)]
    distinct = [len(set(answers[i * n:(i + 1) * n])) for i in range(n_q)]
    return {"kl": (lp - lr_).mean().item(),                      # KL(model || reference), in nats per answer
            "entropy": (-lp).mean().item(),                     # nats per answer
            "sample_acc": sum(right) / len(right),
            "distinct": sum(distinct) / n_q,                    # different answers among 16 tries
            "greedy_acc": accuracy(model, test)}


@torch.no_grad()
def show(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    _, answers, _, flat = sample(model, SHOW, n=8, generator=g)
    return {f"{a}+{b}=": answers[i * 8:(i + 1) * 8] for i, (a, b) in enumerate(SHOW)}


def train_run(beta, seed, train, test, lr=3e-4, every=100):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model, ref = load(), load()
    ref.eval()
    opt = torch.optim.Adam(model.parameters(), lr)
    curve, samples = [], {}
    for step in range(STEPS + 1):
        if step % every == 0:
            model.eval()
            curve.append({"step": step, **measure(model, ref, test)})
            if step in (0, 200, 500, 1000, 2000):
                samples[step] = show(model)
            model.train()
        if step == STEPS:
            break
        qs = rng.sample(train, QUESTIONS)
        idx, answers, lengths, flat = sample(model, qs, n=TRIES)
        r = torch.tensor([reward(a, b, x) for (a, b), x in zip(flat, answers)])
        logp = answer_logprob(model, idx, lengths)
        with torch.no_grad():
            kl = logp.detach() - answer_logprob(ref, idx, lengths)
        shaped = r - beta * kl
        g = shaped.view(QUESTIONS, TRIES)
        adv = (g - g.mean(1, keepdim=True)).view(-1)
        loss = -(adv * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"beta": beta, "seed": seed, "lr": lr, "curve": curve, "samples": samples}


def main(quick=False):
    global STEPS
    if quick:
        STEPS = 300
    t0 = time.time()
    train, test = split()
    ref = load()
    res = {"steps": STEPS, "questions": QUESTIONS, "tries": TRIES, "betas": list(BETAS), "seeds": list(SEEDS),
           "start": {**measure(ref, ref, test), "samples": show(ref)}, "sweep": [], "fast": []}
    print("start", {k: v for k, v in res["start"].items() if k != "samples"}, flush=True)
    for beta in BETAS:
        for s in SEEDS:
            run = train_run(beta, s, train, test)
            res["sweep"].append(run)
            print(f"beta {beta} seed {s}", {k: round(v, 3) for k, v in run["curve"][-1].items()},
                  f"{time.time() - t0:.0f}s", flush=True)
    for beta in (0.0, 0.1):
        run = train_run(beta, 1, train, test, lr=3e-3, every=50)
        res["fast"].append(run)
        print(f"fast beta {beta}", {k: round(v, 3) for k, v in run["curve"][-1].items()}, f"{time.time() - t0:.0f}s", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open("results_quick.json" if quick else "results.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    main(quick=sys.argv[1:] == ["quick"])
