"""Episode 4 of The Frontier: a model that edits itself (a small-scale Self-WSP).

    python self_edit.py self 8      # Qwen proposes its own edits (8 threads), ~1.5 h on a CPU → run_self.json
    python self_edit.py random 8    # the random baseline, same edit language and budget → run_random.json

Qwen3.5-0.8B is shown a summary of its own weights and its graded answers on practice problems (adding three
3-digit numbers), and proposes an edit to its own weights as JSON. We apply each proposal, score it on the practice
problems, and keep it only if the score goes up; the edited model is then the editor for the next round.
The same loop with random edits from the same edit language is the baseline (episode 3).
"""
import json
import random
import re

import torch

from qwen_arith import accuracy, chat, generate, problems

EDITABLE_NORMS = [f"model.layers.{i}.{n}.weight" for i in range(24) for n in ("input_layernorm", "post_attention_layernorm")] + ["model.norm.weight"]


def weight_view(model):
    """Text view of the editable weights: every RMSNorm vector w (49 x 1024; the layer multiplies by 1 + w) as
    statistics plus its 4 largest entries."""
    sd = dict(model.named_parameters())
    lines = []
    for n in EDITABLE_NORMS:
        w = sd[n].detach()
        top = torch.topk(w.abs(), 4).indices.tolist()
        lines.append(f"{n} mean={w.mean():+.3f} std={w.std():.3f} max={w.max():+.3f} min={w.min():+.3f} "
                     f"largest: " + ", ".join(f"[{i}]={w[i]:+.2f}" for i in top))
    return "\n".join(lines)


INSTRUCTIONS = """You are Qwen3.5-0.8B, a 24-layer language model. Below is a summary of some of YOUR OWN weights and
your graded answers to practice problems. The weights shown are the RMSNorm vectors w of every layer (1024 numbers
each): each RMSNorm normalizes its input and multiplies channel i by (1 + w[i]), so w = 0 leaves a channel unchanged,
w > 0 amplifies it and w < -1 flips its sign. You can edit your own weights. Propose ONE edit that would make you add
numbers more accurately. Allowed edits:
{"op": "scale", "tensor": NAME, "factor": F}               multiply a whole vector w by F (0.5 to 1.5)
{"op": "set", "tensor": NAME, "index": I, "value": V}       set one entry w[I] to V
Reply with JSON only, for example {"op": "scale", "tensor": "model.layers.3.input_layernorm.weight", "factor": 1.1}"""


def proposal_prompt(model, acc, rows):
    wrong = [f"{q} = you said {t.strip()[:20]!r}, right: {a}" for (q, a), t, ok in rows if not ok][:24]
    return (INSTRUCTIONS + f"\n\nYour practice score: {acc:.0%}. Some wrong answers:\n" + "\n".join(wrong)
            + "\n\nYour weights:\n" + weight_view(model) + "\n\nYour edit (JSON only):")


def parse(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        e = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if e.get("tensor") not in EDITABLE_NORMS or e.get("op") not in ("scale", "set"):
        return None
    try:
        if e["op"] == "scale":
            e["factor"] = float(e["factor"])
            if not 0.0 < e["factor"] < 10:
                return None
        else:
            e["index"], e["value"] = int(e["index"]), float(e["value"])
            if not 0 <= e["index"] < 1024:
                return None
    except (KeyError, TypeError, ValueError):
        return None
    return {k: e[k] for k in ("op", "tensor", "factor", "index", "value") if k in e}


def random_proposal(rng, model):
    sd = dict(model.named_parameters())
    name = rng.choice(EDITABLE_NORMS)
    if rng.random() < 0.5:
        return {"op": "scale", "tensor": name, "factor": round(rng.uniform(0.5, 1.5), 3)}
    w = sd[name].detach()
    return {"op": "set", "tensor": name, "index": rng.randrange(1024), "value": round(rng.gauss(w.mean().item(), w.std().item()), 3)}


@torch.no_grad()
def apply(model, e):
    """Apply an edit in place; return a function that undoes it."""
    w = dict(model.named_parameters())[e["tensor"]]
    old = w.detach().clone()
    if e["op"] == "scale":
        w.mul_(e["factor"])
    else:
        w[e["index"]] = e["value"]

    def undo():
        w.copy_(old)
    return undo


def run(method, rounds=25, per_round=4, seed=1, out=None):
    import time
    tok, model = load_model()
    practice, test = problems("add3", 48, 1), problems("add3", 200, 2)
    acc, texts, ok = accuracy(tok, model, practice)
    t_acc = accuracy(tok, model, test)[0]
    log = {"method": method, "seed": seed, "practice_start": acc, "test_start": t_acc, "rounds": [], "kept": []}
    print(f"[{method}] start practice {acc:.3f} test {t_acc:.3f}", flush=True)
    rng = random.Random(seed)
    t0 = time.time()
    for rnd in range(1, rounds + 1):
        if method == "self":
            prompt = proposal_prompt(model, acc, list(zip(practice, texts, ok)))
            raw = generate(tok, model, [prompt] * per_round, max_new_tokens=80, batch=per_round, sample=True,
                           temperature=0.8, seed=seed * 1000 + rnd)
            props = [(r, parse(r)) for r in raw]
        else:
            props = [("", random_proposal(rng, model)) for _ in range(per_round)]
        entry = {"round": rnd, "practice_before": acc, "proposals": []}
        best = None
        for raw_text, e in props:
            rec = {"raw": raw_text[:300], "edit": e}
            if e is not None:
                undo = apply(model, e)
                a, tx, k = accuracy(tok, model, practice)
                undo()
                rec["practice"] = a
                if a > acc and (best is None or a > best[0]):
                    best = (a, e, tx, k)
            entry["proposals"].append(rec)
        if best:
            apply(model, best[1])
            acc, texts, ok = best[0], best[2], best[3]
            t_acc = accuracy(tok, model, test)[0]
            log["kept"].append({"round": rnd, "edit": best[1], "practice": acc, "test": t_acc})
            entry["kept"] = best[1]
        entry["practice_after"] = acc
        log["rounds"].append(entry)
        valid = sum(p["edit"] is not None for p in entry["proposals"])
        print(f"[{method}] round {rnd}: {valid}/{per_round} valid, practice {acc:.3f}"
              + (f", kept {best[1]}, test {t_acc:.3f}" if best else "") + f" ({time.time() - t0:.0f}s)", flush=True)
        json.dump(log, open(out or f"run_{method}.json", "w"), indent=1)
    log["practice_final"], log["test_final"] = acc, accuracy(tok, model, test)[0]
    log["seconds"] = round(time.time() - t0)
    json.dump(log, open(out or f"run_{method}.json", "w"), indent=1)
    print(f"[{method}] done practice {acc:.3f} test {log['test_final']:.3f}", flush=True)


def load_model():
    from qwen_arith import load
    return load()


if __name__ == "__main__":
    import sys
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    run(sys.argv[1])
