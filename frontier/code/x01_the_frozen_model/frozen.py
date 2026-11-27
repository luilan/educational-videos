"""Episode 1 of The Frontier: The Frozen Model.

1. Train a tiny GPT on two-digit addition until it is almost always right, and freeze it (save its weights).
2. Look at the weight file: how many numbers, how big.
3. Zero every one of the 102,413 numbers in turn and measure accuracy on 250 probe questions: how many matter?
4. Re-check the most damaging edits on all 1,000 test questions and show how one answer changes.

Writes model.pt, results.json and drops.npy. About 5 minutes on 12 CPU cores.
"""
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from arith import MAX_ANS, Q_LEN, VOCAB, TinyGPT, accuracy, encode, prompt, sample, split

WORKERS, STEPS = 12, 3000
_model = _probe = None


def train_model(train):
    torch.manual_seed(0)
    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.1)
    seqs = torch.tensor([encode(prompt(a, b) + f"{a + b}.".ljust(MAX_ANS, ".")) for a, b in train])
    for _ in range(STEPS):
        batch = seqs[torch.randint(len(seqs), (64,))]
        target = batch[:, 1:].clone()
        target[:, :Q_LEN - 1] = -100
        loss = F.cross_entropy(model(batch[:, :-1]).reshape(-1, VOCAB), target.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def _init(state, probe):
    global _model, _probe
    torch.set_num_threads(1)
    _model = TinyGPT()
    _model.load_state_dict(state)
    _model.eval()
    _probe = probe


@torch.no_grad()
def _scan(job):
    """Zero each number in params[name][start:end] (flattened) in turn; return accuracy on the probe set."""
    name, start, end = job
    flat = dict(_model.named_parameters())[name].data.view(-1)
    out = []
    for i in range(start, end):
        old = flat[i].item()
        flat[i] = 0.0
        out.append(accuracy(_model, _probe))
        flat[i] = old
    return name, start, out


def main():
    t0 = time.time()
    torch.set_num_threads(8)
    train, test = split()
    model = train_model(train)
    torch.save(model.state_dict(), "model.pt")
    params = dict(model.named_parameters())
    res = {"params": sum(p.numel() for p in params.values()), "file_bytes": os.path.getsize("model.pt"),
           "tensors": [[n, list(p.shape), p.numel()] for n, p in params.items()],
           "acc_test": accuracy(model, test), "example_weights": params["blocks.0.mlp.0.weight"][0, :8].tolist(),
           "weight_range": [min(p.min().item() for p in params.values()), max(p.max().item() for p in params.values())]}
    probe = test[:250]
    res["acc_probe"] = accuracy(model, probe)

    jobs = []
    for name, p in params.items():
        for s in range(0, p.numel(), 512):
            jobs.append((name, s, min(s + 512, p.numel())))
    state = {k: v.clone() for k, v in model.state_dict().items()}
    acc = {name: np.zeros(p.numel()) for name, p in params.items()}
    with mp.get_context("fork").Pool(WORKERS, _init, (state, probe)) as pool:
        for name, start, out in pool.imap_unordered(_scan, jobs, chunksize=4):
            acc[name][start:start + len(out)] = out
    names = list(params)
    drops = np.concatenate([res["acc_probe"] - acc[n] for n in names])
    np.save("drops.npy", drops)
    edges = [-1, 0.0005, 0.01, 0.1, 0.5, 1.01]
    labels = ["no change", "under 1 point", "1–10 points", "10–50 points", "over 50 points"]
    res["drop_hist"] = {lab: int(((drops >= lo) & (drops < hi)).sum()) for lab, lo, hi in zip(labels, edges, edges[1:])}
    res["improved"] = int((drops < -0.0005).sum())
    res["drop_hist"]["no change"] += 0       # improvements are counted separately in "improved"
    res["drop_hist"]["no change"] = int(((drops > -0.0005) & (drops < 0.0005)).sum())

    # Map flat index back to (tensor, index) and re-check the 10 most damaging on all 1,000 test questions.
    offsets = np.cumsum([0] + [params[n].numel() for n in names])
    worst = []
    for k in np.argsort(-drops)[:10]:
        t = int(np.searchsorted(offsets, k, side="right") - 1)
        name, i = names[t], int(k - offsets[t])
        flat = params[name].data.view(-1)
        old = flat[i].item()
        with torch.no_grad():
            flat[i] = 0.0
            a = accuracy(model, test)
            _, ans, _, _ = sample(model, [(47, 38), (27, 15), (63, 58)], greedy=True)
            flat[i] = old
        idx = np.unravel_index(i, tuple(params[name].shape))
        worst.append({"tensor": name, "index": [int(x) for x in idx], "value": old, "acc_test": a,
                      "answers": dict(zip(["47+38=", "27+15=", "63+58="], ans))})
    res["worst"] = worst
    _, ans, _, _ = sample(model, [(47, 38), (27, 15), (63, 58)], greedy=True)
    res["answers_before"] = dict(zip(["47+38=", "27+15=", "63+58="], ans))
    # Per tensor: share of numbers whose removal costs at least 1 point.
    res["per_tensor"] = {n: {"numbers": int(params[n].numel()),
                             "matter": int((drops[offsets[t]:offsets[t + 1]] >= 0.01).sum())}
                         for t, n in enumerate(names)}
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open("results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_tensor", "tensors")}, indent=1))


if __name__ == "__main__":
    main()
