"""Episode 5 of The Frontier: diagnosing its own mistakes (a small-scale version of self-rewriting, arXiv 2611.21885).

Same model, task, edit language and budget as episode 4 (Qwen3.5-0.8B, adding three 3-digit numbers, 48 practice
and 200 held-out problems, 25 rounds x 4 proposals, keep the best if the practice score rises). Each round adds two
steps before the edit:

  1. diagnose  the model reads its wrong practice answers and writes one sentence about what goes wrong
  2. probe     the harness runs the model ("the ability to run itself"): for each of its 49 RMSNorm vectors, how
               much scaling that vector by 1.1 changes the log-probability of the RIGHT answers to the wrong
               practice problems; the 10 largest effects are shown to the model
  3. edit      the model proposes 4 edits, given its diagnosis and the probe table

Baseline "probe": no language model in the loop, just the 4 probe entries with the largest effects, each scaled
by 1.1 or 0.9 in the helpful direction. (The probe and the diagnosis are recomputed only after a kept edit, since
nothing else changes them; with nothing kept, the probe baseline proposes the same 4 edits every round.)

    python self_rewrite.py diagnose 8     # ~2 h → run_diagnose.json
    python self_rewrite.py probe 8        # ~1 h → run_probe.json
"""
import json
import random
import re
import sys
import time

import torch
import torch.nn.functional as F

from qwen_arith import accuracy, chat, generate, load, problems

NORMS = [f"model.layers.{i}.{n}.weight" for i in range(24) for n in ("input_layernorm", "post_attention_layernorm")] + ["model.norm.weight"]
SHORT = {n: n.replace("model.", "").replace(".weight", "") for n in NORMS}
ROUNDS, PER_ROUND = 25, 4


@torch.no_grad()
def answer_logprob(tok, model, probs):
    """Mean log-probability, per problem, of the correct answer's tokens after the question (teacher forcing)."""
    prompts = [chat(tok, f"What is {q}? Answer with just the number.") for q, _ in probs]
    full = [p + str(a) for p, (_, a) in zip(prompts, probs)]
    enc = tok(full, return_tensors="pt", padding=True)
    n_ans = [len(tok(p + str(a))["input_ids"]) - len(tok(p)["input_ids"]) for p, (_, a) in zip(prompts, probs)]
    logits = model(**enc).logits[:, :-1]
    lp = F.log_softmax(logits.float(), -1).gather(-1, enc["input_ids"][:, 1:, None])[..., 0]
    T = enc["input_ids"].shape[1] - 1
    return torch.stack([lp[i, T - n:].mean() for i, n in enumerate(n_ans)])


@torch.no_grad()
def probe(tok, model, wrong):
    """Effect of scaling each norm vector by 1.1 on the right answers' log-probability (wrong problems only)."""
    sd = dict(model.named_parameters())
    base = answer_logprob(tok, model, wrong).mean().item()
    eff = {}
    for n in NORMS:
        w = sd[n]
        old = w.clone()
        w.mul_(1.1)
        eff[n] = answer_logprob(tok, model, wrong).mean().item() - base
        w.copy_(old)
    return base, eff


DIAG = """Here are arithmetic problems you answered wrongly (you are Qwen3.5-0.8B):
{wrong}
In one sentence: what kind of mistake do you make?"""

EDIT = """You are Qwen3.5-0.8B and you can edit your own weights. Your practice score is {acc:.0%}.
Your diagnosis of your mistakes: {diag}

You ran yourself to see which of your weights matter for these mistakes. Each of your RMSNorm layers multiplies its
channels by (1 + w). Scaling a vector w by 1.1 changes the log-probability of the RIGHT answers to your wrong problems
by (positive = helps):
{table}

Propose ONE edit to fix your mistakes. Allowed edits:
{{"op": "scale", "tensor": NAME, "factor": F}}            multiply a whole vector w by F (0.5 to 1.5)
{{"op": "set", "tensor": NAME, "index": I, "value": V}}    set one entry w[I] to V
NAME is a full name such as model.layers.6.post_attention_layernorm.weight. Reply with JSON only."""


def parse(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        e = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    t = e.get("tensor")
    if t in SHORT.values():
        t = f"model.{t}.weight"
    if t not in NORMS or e.get("op") not in ("scale", "set"):
        return None
    try:
        if e["op"] == "scale":
            f = float(e["factor"])
            return {"op": "scale", "tensor": t, "factor": f} if 0 < f < 10 else None
        i, v = int(e["index"]), float(e["value"])
        return {"op": "set", "tensor": t, "index": i, "value": v} if 0 <= i < 1024 else None
    except (KeyError, TypeError, ValueError):
        return None


@torch.no_grad()
def apply(model, e):
    w = dict(model.named_parameters())[e["tensor"]]
    old = w.clone()
    if e["op"] == "scale":
        w.mul_(e["factor"])
    else:
        w[e["index"]] = e["value"]
    return lambda: w.copy_(old)


def run(method, seed=1):
    tok, model = load()
    practice, test = problems("add3", 48, 1), problems("add3", 200, 2)
    acc, texts, ok = accuracy(tok, model, practice)
    log = {"method": method, "practice_start": acc, "test_start": accuracy(tok, model, test)[0], "rounds": [], "kept": []}
    print(f"[{method}] start practice {acc:.3f} test {log['test_start']:.3f}", flush=True)
    t0 = time.time()
    cache = None                      # probe and diagnosis only change when an edit is kept
    for rnd in range(1, ROUNDS + 1):
        if cache is None:
            wrong = [p for p, k in zip(practice, ok) if not k]
            base, eff = probe(tok, model, wrong)
            diag = None
            if method == "diagnose":
                wl = "\n".join([f"{q} = you said {t.strip()[:16]}, right: {a}" for (q, a), t, k in zip(practice, texts, ok) if not k][:20])
                diag = generate(tok, model, [DIAG.format(wrong=wl)], max_new_tokens=60, batch=1)[0].strip()
            cache = (base, eff, diag)
        base, eff, diag = cache
        top = sorted(eff.items(), key=lambda kv: -abs(kv[1]))[:10]
        entry = {"round": rnd, "practice_before": acc, "probe_base": base, "probe_top": top}
        if method == "diagnose":
            table = "\n".join(f"{n}: {d:+.4f}" for n, d in top)
            raw = generate(tok, model, [EDIT.format(acc=acc, diag=diag, table=table)] * PER_ROUND, max_new_tokens=80,
                           batch=PER_ROUND, sample=True, temperature=0.8, seed=seed * 1000 + rnd)
            props = [(r, parse(r)) for r in raw]
            entry["diagnosis"] = diag
        else:
            props = [("", {"op": "scale", "tensor": n, "factor": 1.1 if d > 0 else 0.9}) for n, d in top[:PER_ROUND]]
        best, entry["proposals"] = None, []
        for r, e in props:
            rec = {"raw": r[:300], "edit": e}
            if e:
                undo = apply(model, e)
                a, tx, k = accuracy(tok, model, practice)
                undo()
                rec["practice"] = a
                rec["probe_effect"] = eff.get(e["tensor"]) if e["op"] == "scale" else None
                if a > acc and (best is None or a > best[0]):
                    best = (a, e, tx, k)
            entry["proposals"].append(rec)
        if best:
            apply(model, best[1])
            acc, texts, ok = best[0], best[2], best[3]
            cache = None
            t_acc = accuracy(tok, model, test)[0]
            log["kept"].append({"round": rnd, "edit": best[1], "practice": acc, "test": t_acc})
        entry["practice_after"] = acc
        log["rounds"].append(entry)
        msg = f"[{method}] round {rnd}: practice {acc:.3f}" + (f", kept {best[1]}, test {t_acc:.3f}" if best else "")
        print(msg + (f" | diag: {entry.get('diagnosis', '')[:100]}" if method == "diagnose" else "") + f" ({time.time() - t0:.0f}s)", flush=True)
        json.dump(log, open(f"run_{method}.json", "w"), indent=1)
    log["practice_final"], log["test_final"] = acc, accuracy(tok, model, test)[0]
    log["seconds"] = round(time.time() - t0)
    json.dump(log, open(f"run_{method}.json", "w"), indent=1)
    print(f"[{method}] done practice {acc:.3f} test {log['test_final']:.3f}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    run(sys.argv[1])
