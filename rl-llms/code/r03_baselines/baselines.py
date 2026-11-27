"""Episode 3: Baselines and Advantages.

REINFORCE with three choices of baseline, from episode 1's model, 3 seeds each:
  none      advantage = reward                      (episode 2)
  batch     advantage = reward - mean reward of the whole batch
  question  advantage = reward - mean reward of the 8 tries at the same question
Also measures the noise of each gradient estimate at the starting model (20 independent batches).
Writes results.json. About 12 minutes on a CPU. Then `python baselines.py offset` (about 6 minutes) repeats the
comparison with rewards of 1 (wrong) and 2 (right), and measures the gradient noise at episode 2's trained model.
"""
import json
import random
import time

import torch

from arith import TinyGPT, accuracy, answer_logprob, reward, sample, split

torch.set_num_threads(8)
STEPS, QUESTIONS, TRIES, LR, SEEDS = 2000, 32, 8, 3e-4, (1, 2, 3)
KINDS = ("none", "batch", "question")
BASE = "../r01_learning_from_a_grade/base.pt"


def advantages(r, kind):
    if kind == "none":
        return r
    if kind == "batch":
        return r - r.mean()
    g = r.view(QUESTIONS, TRIES)
    return (g - g.mean(1, keepdim=True)).view(-1)


def load():
    m = TinyGPT()
    m.load_state_dict(torch.load(BASE))
    return m


def batch_loss(model, train, kind, rng, offset=0.0):
    qs = rng.sample(train, QUESTIONS)
    idx, answers, lengths, flat = sample(model, qs, n=TRIES)
    r = torch.tensor([reward(a, b, x) for (a, b), x in zip(flat, answers)]) + offset
    adv = advantages(r, kind)
    return -(adv * answer_logprob(model, idx, lengths)).mean(), r, adv


def train_run(kind, seed, train, test, offset=0.0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = load()
    opt = torch.optim.Adam(model.parameters(), LR)
    curve = []
    for step in range(STEPS + 1):
        if step % 100 == 0:
            model.eval()
            curve.append([step, accuracy(model, test)])
            model.train()
        if step == STEPS:
            break
        loss, _, _ = batch_loss(model, train, kind, rng, offset)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return curve


def gradient_noise(train, n=20, path=BASE):
    """Signal-to-noise of each estimator at a given model: |mean gradient| / spread across batches."""
    out = {}
    for kind in KINDS:
        torch.manual_seed(0)
        rng = random.Random(0)
        model = TinyGPT()
        model.load_state_dict(torch.load(path))
        grads = []
        for _ in range(n):
            model.zero_grad()
            loss, _, _ = batch_loss(model, train, kind, rng)
            loss.backward()
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
        G = torch.stack(grads)
        mean, spread = G.mean(0), G.std(0)
        out[kind] = {"signal": mean.norm().item(), "noise": spread.norm().item(),
                     "snr": (mean.norm() / spread.norm()).item()}
    return out


def example_advantages():
    """The advantages of one question's 8 tries under each baseline (a made-up batch with known rewards)."""
    torch.manual_seed(5)
    r = torch.zeros(QUESTIONS * TRIES)
    r[torch.randperm(len(r))[:53]] = 1                    # 53 of 256 right, as in episode 2's first batch
    g = r.view(QUESTIONS, TRIES)
    i = int(((g.sum(1) == 3)).nonzero()[0])              # a question with 3 of 8 right
    return {"rewards": g[i].tolist(), "none": g[i].tolist(),
            "batch": (g[i] - r.mean()).tolist(), "batch_mean": r.mean().item(),
            "question": (g[i] - g[i].mean()).tolist(), "question_mean": g[i].mean().item()}


def main():
    t0 = time.time()
    train, test = split()
    res = {"steps": STEPS, "questions": QUESTIONS, "tries": TRIES, "lr": LR, "seeds": list(SEEDS),
           "noise": gradient_noise(train), "example": example_advantages(), "runs": {}}
    print(json.dumps(res["noise"], indent=1), flush=True)
    for kind in KINDS:
        res["runs"][kind] = [train_run(kind, s, train, test) for s in SEEDS]
        finals = [c[-1][1] for c in res["runs"][kind]]
        reach = []
        for c in res["runs"][kind]:
            hit = [s for s, a in c if a >= 0.6]
            reach.append(hit[0] if hit else None)
        res.setdefault("summary", {})[kind] = {"final": finals, "final_mean": sum(finals) / len(finals),
                                               "steps_to_60": reach}
        print(kind, res["summary"][kind], f"{time.time() - t0:.0f}s", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open("results.json", "w"), indent=1)


def offset_experiment():
    """Rewards 1 for a wrong answer and 2 for a right one ("a point for trying"): no baseline vs baselines.
    Adds res["offset"] and res["noise_late"] (gradient noise at episode 2's trained model, 73% right)."""
    train, test = split()
    res = json.load(open("results.json"))
    off = {"none": [train_run("none", s, train, test, 1.0) for s in SEEDS],
           "batch": [train_run("batch", SEEDS[0], train, test, 1.0)],
           "question": [train_run("question", SEEDS[0], train, test, 1.0)]}
    res["offset"] = {k: {"curves": v, "final": [c[-1][1] for c in v]} for k, v in off.items()}
    res["offset"]["batch_identical_to_unshifted"] = off["batch"][0] == res["runs"]["batch"][0]
    res["offset"]["question_identical_to_unshifted"] = off["question"][0] == res["runs"]["question"][0]
    res["noise_late"] = gradient_noise(train, path="../r02_reinforce/reinforced.pt")
    json.dump(res, open("results.json", "w"), indent=1)
    print(json.dumps({k: v["final"] if isinstance(v, dict) else v for k, v in res["offset"].items()}), res["noise_late"])


if __name__ == "__main__":
    import sys
    offset_experiment() if sys.argv[1:] == ["offset"] else main()
