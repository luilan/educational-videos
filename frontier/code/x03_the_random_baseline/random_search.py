"""Episode 3 of The Frontier: The Random Baseline.

Episode 2's loop (propose an edit, test it on 300 practice questions, keep it only if the score goes up, stop at 99%)
with the editor replaced by chance. Two random editors:

  edits   the same edit language Claude used (set one number, scale a row, swap two rows, replace a row), with every
          choice random: the tensor (uniform over the 30 weight tensors), the row or entry, the new value (drawn
          from that tensor's own mean and spread), the factor (uniform between -2 and 2)
  nudges  add a little Gaussian noise to every weight at once (1% of each tensor's spread): blind hill climbing

Same three damages as episode 2, up to 10,000 tries each, 3 seeds for `edits`, 1 for `nudges`. The 1,000 test
questions are only used to judge, at fixed checkpoints. About 20 minutes on 16 CPU threads:  python random_search.py
Writes results.json.
"""
import json
import math
import random
import time
from multiprocessing import Pool

import torch

from arith import TinyGPT, accuracy, reward, sample, split

MODEL_PT = "../x01_the_frozen_model/model.pt"
N_PRACTICE, BUDGET, TARGET = 300, 10_000, 0.99
CHECKS = (1, 4, 10, 100, 1000, 10_000)
CASES = ("zero", "swap", "flip")


def damage(case, s):           # identical to episode 2's wsp.py
    if case == "zero":
        s["blocks.0.ln1.weight"][13] = 0.0
    elif case == "flip":
        s["tok.weight"][7] = -s["tok.weight"][7]
    elif case == "swap":
        w = s["tok.weight"]
        w[[3, 8]] = w[[8, 3]]


def practice_score(model, qs):
    _, answers, _, flat = sample(model, qs, greedy=True)
    return sum(reward(a, b, x) for (a, b), x in zip(flat, answers)) / len(qs)


def random_edit(rng, state, names):
    name = rng.choice(names)
    t = state[name]
    op = rng.choice(["set", "scale_row", "swap_rows", "set_row"] if t.dim() == 2 else ["set", "scale_row"])
    mean, std = t.mean().item(), t.std().item()
    if op == "set":
        idx = tuple(rng.randrange(n) for n in t.shape)
        return {"op": op, "tensor": name, "index": list(idx), "value": round(rng.gauss(mean, std), 3)}
    row = rng.randrange(t.shape[0])
    if op == "scale_row":
        return {"op": op, "tensor": name, "row": row, "factor": round(rng.uniform(-2, 2), 3)}
    if op == "swap_rows":
        other = rng.choice([r for r in range(t.shape[0]) if r != row])
        return {"op": op, "tensor": name, "row": row, "row_b": other}
    return {"op": op, "tensor": name, "row": row, "values": [round(rng.gauss(mean, std), 3) for _ in range(t.shape[1])]}


def apply(state, e):
    s = dict(state)
    t = s[e["tensor"]].clone()
    if e["op"] == "set":
        t[tuple(e["index"])] = e["value"]
    elif e["op"] == "scale_row":
        t[e["row"]] = t[e["row"]] * e["factor"]
    elif e["op"] == "swap_rows":
        t[[e["row"], e["row_b"]]] = t[[e["row_b"], e["row"]]]
    else:
        t[e["row"]] = torch.tensor(e["values"])
    s[e["tensor"]] = t
    return s


def nudge(rng_t, state, names, scale=0.01):
    return {k: (v + scale * v.std() * torch.randn(v.shape, generator=rng_t) if k in names else v) for k, v in state.items()}


def run(job):
    case, method, seed = job
    torch.set_num_threads(2)
    train, test = split()
    practice = random.Random(7).sample(train, N_PRACTICE)
    base = torch.load(MODEL_PT)
    state = {k: v.clone() for k, v in base.items()}
    damage(case, state)
    names = [k for k in state if "mask" not in k]
    model = TinyGPT().eval()

    def score(s, qs):
        model.load_state_dict(s)
        return practice_score(model, qs) if qs is practice else accuracy(model, qs)

    rng, rng_t = random.Random(seed), torch.Generator().manual_seed(seed)
    acc = score(state, practice)
    log = {"case": case, "method": method, "seed": seed, "practice_damaged": acc, "test_damaged": score(state, test),
           "kept": [], "checks": {}, "solved_at": None}
    t0 = time.time()
    with torch.no_grad():
        for trial in range(1, BUDGET + 1):
            if method == "edits":
                e = random_edit(rng, state, names)
                new = apply(state, e)
            else:
                e, new = {"op": "nudge"}, nudge(rng_t, state, names)
            new_acc = score(new, practice)
            if new_acc > acc:
                changed = int(sum((new[k] != state[k]).sum() for k in names))
                state, acc = new, new_acc
                log["kept"].append({"trial": trial, "practice": acc, "edit": e if e["op"] != "set_row" else
                                    {k: v for k, v in e.items() if k != "values"}, "numbers_changed": changed})
            if trial in CHECKS or (acc >= TARGET and log["solved_at"] is None):
                if acc >= TARGET and log["solved_at"] is None:
                    log["solved_at"] = trial
                log["checks"][trial] = {"practice": acc, "test": score(state, test)}
                if acc >= TARGET:
                    break
    log["practice_final"], log["test_final"] = acc, score(state, test)
    log["trials"] = trial
    log["numbers_changed_vs_intact"] = int(sum((state[k] != base[k]).sum() for k in names))
    log["seconds"] = round(time.time() - t0, 1)
    print(f"{case:5s} {method:6s} seed {seed}: practice {log['practice_damaged']:.3f} -> {acc:.3f}, test "
          f"{log['test_damaged']:.3f} -> {log['test_final']:.3f}, solved at {log['solved_at']}, "
          f"{len(log['kept'])} kept, {log['seconds']}s", flush=True)
    return log


def main():
    jobs = [(c, "edits", s) for c in CASES for s in (1, 2, 3)] + [(c, "nudges", 1) for c in CASES]
    t0 = time.time()
    with Pool(6) as pool:
        logs = pool.map(run, jobs, chunksize=1)
    json.dump({"budget": BUDGET, "practice": N_PRACTICE, "target": TARGET, "seconds": round(time.time() - t0, 1),
               "runs": logs}, open("results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
