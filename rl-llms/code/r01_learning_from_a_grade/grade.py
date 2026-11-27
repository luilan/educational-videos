"""Episode 1: Learning From a Grade.

1. Pre-train a tiny GPT on 500 worked examples of two-digit addition ("47+38=85.").
2. Measure it on 1,000 questions it has never seen: greedy, and 64 sampled answers per question.
3. Grade every sample with the checker (reward 1 or 0) and look at the spread.

Writes results.json and base.pt (the starting model for episode 2). About 30 seconds on a CPU.
"""
import collections
import json
import random
import time

import torch
import torch.nn.functional as F

from arith import (BLOCK, CHARS, MAX_ANS, Q_LEN, VOCAB, TinyGPT, accuracy, encode, prompt, reward, sample, split)

torch.manual_seed(0)
torch.set_num_threads(8)
N_WORKED, STEPS, N_SAMPLES = 500, 2000, 64


def pretrain(worked):
    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.1)
    seqs = torch.tensor([encode(prompt(a, b) + f"{a + b}.".ljust(MAX_ANS, ".")) for a, b in worked])
    for _ in range(STEPS):
        batch = seqs[torch.randint(len(seqs), (64,))]
        target = batch[:, 1:].clone()
        target[:, :Q_LEN - 1] = -100                     # learn the answer, not the question
        loss = F.cross_entropy(model(batch[:, :-1]).reshape(-1, VOCAB), target.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def main():
    t0 = time.time()
    train, test = split()
    worked = train[:N_WORKED]
    model = pretrain(worked)
    n_params = sum(p.numel() for p in model.parameters())
    res = {"params": n_params, "worked_examples": N_WORKED, "steps": STEPS, "test_questions": len(test),
           "acc_worked_greedy": accuracy(model, worked), "acc_test_greedy": accuracy(model, test)}

    # The policy: probabilities of each next character for one question.
    a, b = 47, 38
    ids = torch.tensor([encode(prompt(a, b))])
    steps = []
    with torch.no_grad():
        for _ in range(MAX_ANS):
            p = F.softmax(model(ids)[0, -1], -1)
            top = torch.topk(p, 3)
            steps.append([[CHARS[i], round(v, 3)] for v, i in zip(top.values.tolist(), top.indices.tolist())])
            ids = torch.cat([ids, top.indices[:1][None]], 1)
            if CHARS[top.indices[0]] == ".":
                break
    res["policy_example"] = {"question": prompt(a, b), "steps": steps}

    # 64 graded samples per test question.
    g = torch.Generator().manual_seed(1)
    _, answers, _, qs = sample(model, test, n=N_SAMPLES, generator=g)
    r = torch.tensor([reward(x, y, ans) for (x, y), ans in zip(qs, answers)]).view(len(test), N_SAMPLES)
    frac = r.mean(1)
    res["pass1_sampled"] = frac.mean().item()
    res["pass64"] = (frac > 0).float().mean().item()
    res["never_right"] = int((frac == 0).sum())
    res["always_right"] = int((frac == 1).sum())
    edges = [0, 1e-9, 0.25, 0.5, 0.75, 1 - 1e-9, 1.0001]
    labels = ["0", "1–25%", "25–50%", "50–75%", "75–99%", "100%"]
    res["spread"] = {lab: int(((frac >= lo) & (frac < hi)).sum()) for lab, lo, hi in zip(labels, edges, edges[1:])}
    greedy_right = torch.tensor([reward(x, y, ans) for (x, y), ans in
                                 zip(test, sample(model, test, greedy=True)[1])])
    res["greedy_wrong_but_some_sample_right"] = int(((greedy_right == 0) & (frac > 0)).sum())

    # One question in detail: the 64 answers and their grades.
    i = next(k for k, (x, y) in enumerate(test) if 0.2 < frac[k] < 0.6 and (x % 10) + (y % 10) >= 10)
    x, y = test[i]
    tally = collections.Counter(answers[i * N_SAMPLES:(i + 1) * N_SAMPLES])
    res["detail"] = {"question": prompt(x, y), "right": f"{x + y}.",
                     "tally": [[ans, n, reward(x, y, ans)] for ans, n in tally.most_common()]}

    # Where do mistakes come from? Carry vs no carry in the ones digit.
    carry = torch.tensor([(x % 10) + (y % 10) >= 10 for x, y in test])
    res["pass1_carry"] = frac[carry].mean().item()
    res["pass1_no_carry"] = frac[~carry].mean().item()
    res["n_carry"] = int(carry.sum())
    res["seconds"] = round(time.time() - t0, 1)
    torch.save(model.state_dict(), "base.pt")
    json.dump(res, open("results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
