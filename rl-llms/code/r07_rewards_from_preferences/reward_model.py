"""Episode 7, step 2: learn a reward model from preference pairs (Bradley-Terry).

The "people" are a script, so their rule is known: prefer the correct answer; if both are right or both wrong,
prefer the shorter one; and, like real labellers, disagree with that rule 10% of the time (the label is flipped).

The reward model is Qwen2.5-0.5B-Instruct, frozen, plus one trained linear layer on its last hidden state:
reward(question, answer) = w . h + b. It is trained on pairs from 300 questions so that, for every pair,
P(first preferred) = sigmoid(reward(first) - reward(second))  (the Bradley-Terry model), and tested on pairs from the
100 other questions. Writes features.pt (cache), rm_head.pt (used in episode 8) and results.json.
About 6 minutes:  python reward_model.py
"""
import json
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
NOISE, TRAIN_Q = 0.10, 300


def true_pref(a, b):
    """1 if the labelling rule prefers a over b, 0 if b, None for an exact tie."""
    if a["correct"] != b["correct"]:
        return int(a["correct"])
    if a["words"] != b["words"]:
        return int(a["words"] < b["words"])
    return None


def rm_input(tok, question, answer):
    return tok.apply_chat_template([{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
                                   tokenize=False)


@torch.no_grad()
def features(data):
    tok = AutoTokenizer.from_pretrained(NAME)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    texts = [rm_input(tok, d["question"], s["text"]) for d in data for s in d["samples"]]
    out = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], return_tensors="pt", padding=True)
        h = model(**enc, output_hidden_states=True).hidden_states[-1]
        out.append(h[:, -1, :])                                   # last token (left padding puts it at the end)
    return torch.cat(out)


def pairs(data, idx, rng, noisy):
    P = []
    for qi in idx:
        s = data[qi]["samples"]
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                p = true_pref(s[i], s[j])
                if p is None:
                    continue
                label = p if not noisy or rng.random() >= NOISE else 1 - p
                P.append((qi * 4 + i, qi * 4 + j, label, p, s[i]["correct"] != s[j]["correct"]))
    return P


def main():
    torch.set_num_threads(16)
    data = json.load(open("answers.json"))
    try:
        X = torch.load("features.pt")
    except FileNotFoundError:
        X = features(data)
        torch.save(X, "features.pt")
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    rng = random.Random(0)
    order = list(range(len(data)))
    rng.shuffle(order)
    tr_q, te_q = order[:TRAIN_Q], order[TRAIN_Q:]
    train, test = pairs(data, tr_q, rng, noisy=True), pairs(data, te_q, rng, noisy=True)
    torch.manual_seed(0)
    head = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(head.parameters(), 1e-3, weight_decay=1e-3)
    A = torch.tensor([p[0] for p in train]); B = torch.tensor([p[1] for p in train])
    y = torch.tensor([float(p[2]) for p in train])
    curve = []
    for epoch in range(301):
        r = head(Xn)[:, 0]
        loss = F.binary_cross_entropy_with_logits(r[A] - r[B], y)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 25 == 0:
            curve.append([epoch, loss.item(), evaluate(head, Xn, test)["vs_labels"]])
    with torch.no_grad():
        r = head(Xn)[:, 0]
    flat = [s for d in data for s in d["samples"]]
    words = torch.tensor([float(s["words"]) for s in flat]); corr = torch.tensor([float(s["correct"]) for s in flat])
    te_rows = torch.tensor([qi * 4 + k for qi in te_q for k in range(4)])
    res = {"questions": len(data), "answers": len(flat), "train_pairs": len(train), "test_pairs": len(test),
           "noise": NOISE, "answer_accuracy": corr.mean().item(),
           "train": evaluate(head, Xn, train), "test": evaluate(head, Xn, test), "curve": curve,
           "corr_reward_words": corr_(r[te_rows], words[te_rows]), "corr_reward_correct": corr_(r[te_rows], corr[te_rows]),
           "mean_reward": {"right_short": mean_where(r, corr, words, 1, True), "right_long": mean_where(r, corr, words, 1, False),
                           "wrong_short": mean_where(r, corr, words, 0, True), "wrong_long": mean_where(r, corr, words, 0, False)},
           "examples": examples(data, te_q, r)}
    torch.save({"head": head.state_dict(), "mu": mu, "sd": sd, "base": NAME}, "rm_head.pt")
    json.dump(res, open("results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("curve", "examples")}, indent=1))


@torch.no_grad()
def evaluate(head, Xn, P):
    r = head(Xn)[:, 0]
    out = {}
    for name, sel in (("all", lambda p: True), ("correctness_differs", lambda p: p[4]), ("same_correctness", lambda p: not p[4])):
        Q = [p for p in P if sel(p)]
        if not Q:
            continue
        pred = [(r[p[0]] > r[p[1]]).item() for p in Q]
        out[name] = {"n": len(Q), "vs_true_rule": sum(int(a) == p[3] for a, p in zip(pred, Q)) / len(Q)}
    out["vs_labels"] = sum((r[p[0]] > r[p[1]]).item() == bool(p[2]) for p in P) / len(P)
    return out


def corr_(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()


def mean_where(r, corr, words, c, short):
    sel = (corr == c) & ((words <= 25) if short else (words > 25))
    return {"n": int(sel.sum()), "reward": r[sel].mean().item() if sel.any() else None}


def examples(data, te_q, r):
    """Test pairs where the reward model disagrees with the rule: a wrong answer it scores above a right one."""
    out = []
    for qi in te_q:
        s = data[qi]["samples"]
        for i in range(4):
            for j in range(4):
                if s[i]["correct"] and not s[j]["correct"] and r[qi * 4 + j] > r[qi * 4 + i]:
                    out.append({"question": data[qi]["question"], "answer": data[qi]["answer"],
                                "right": s[i]["text"][:300], "right_reward": r[qi * 4 + i].item(), "right_words": s[i]["words"],
                                "wrong": s[j]["text"][:300], "wrong_reward": r[qi * 4 + j].item(), "wrong_words": s[j]["words"]})
    return out[:20]


if __name__ == "__main__":
    main()
