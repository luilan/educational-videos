"""Episode 2: REINFORCE.

Start from episode 1's model (500 worked examples, 14% on new questions) and train it with REINFORCE: sample answers,
grade them, and raise the log-probability of each answer in proportion to its reward. No answer key, no baseline yet
(that is episode 3). Writes results.json and reinforced.pt. About 2 minutes on a CPU.

    python reinforce.py            # needs ../r01_learning_from_a_grade/base.pt (run grade.py there first)
"""
import collections
import json
import random
import time

import torch

from arith import TinyGPT, accuracy, answer_logprob, reward, sample, split

torch.manual_seed(1)
random.seed(1)
torch.set_num_threads(8)
STEPS, QUESTIONS, TRIES, LR = 2000, 32, 8, 3e-4
WATCH = [(47, 38), (27, 15)]


def tally(model, q, n=64, seed=5):
    g = torch.Generator().manual_seed(seed)
    _, answers, _, _ = sample(model, [q], n=n, generator=g)
    return collections.Counter(answers).most_common(4)


def per_question(model, test, n=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    _, answers, _, qs = sample(model, test, n=n, generator=g)
    r = torch.tensor([reward(a, b, x) for (a, b), x in zip(qs, answers)]).view(len(test), n)
    return r.mean(1)


def main():
    t0 = time.time()
    train, test = split()
    model = TinyGPT()
    model.load_state_dict(torch.load("../r01_learning_from_a_grade/base.pt"))
    opt = torch.optim.Adam(model.parameters(), LR)
    before = per_question(model.eval(), test)
    res = {"steps": STEPS, "questions_per_step": QUESTIONS, "tries": TRIES, "lr": LR,
           "watch_before": {f"{a}+{b}=": tally(model, (a, b)) for a, b in WATCH}, "curve": []}
    for step in range(STEPS + 1):
        if step % 50 == 0:
            model.eval()
            res["curve"].append([step, accuracy(model, test), accuracy(model, test, greedy=False)])
            model.train()
            print(step, res["curve"][-1], f"{time.time() - t0:.0f}s", flush=True)
        if step == STEPS:
            break
        qs = random.sample(train, QUESTIONS)                     # questions only: no answers are used
        idx, answers, lengths, flat = sample(model, qs, n=TRIES)
        r = torch.tensor([reward(a, b, x) for (a, b), x in zip(flat, answers)])
        logp = answer_logprob(model, idx, lengths)
        loss = -(r * logp).mean()                                # REINFORCE: reward × log-probability
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            res["first_batch"] = {"answers": len(answers), "rewarded": int(r.sum())}
    model.eval()
    after = per_question(model, test)
    res["watch_after"] = {f"{a}+{b}=": tally(model, (a, b)) for a, b in WATCH}
    res["acc_before"], res["acc_after"] = res["curve"][0][1], res["curve"][-1][1]
    never = before == 0
    res["never_right_before"] = int(never.sum())
    res["never_now_right_greedy"] = int(sum(reward(a, b, x) for (a, b), x, nv in
                                            zip(test, sample(model, test, greedy=True)[1], never.tolist()) if nv))
    res["sampled_before"], res["sampled_after"] = before.mean().item(), after.mean().item()
    res["seconds"] = round(time.time() - t0, 1)
    torch.save(model.state_dict(), "reinforced.pt")
    json.dump(res, open("results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
