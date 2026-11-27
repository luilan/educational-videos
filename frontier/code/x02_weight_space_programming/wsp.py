"""Episode 2 of The Frontier: Weight-Space Programming.

A large model (Claude, run headless through the Claude Code CLI) repairs a small network that we damaged on purpose,
by reading its weights and writing edits. No gradients: the editor only sees numbers and graded answers, and our
harness applies its edits, tests them on practice questions and keeps an edit only if the practice score improves.
The 1,000 test questions are never shown to the editor; they judge the final result.

    python wsp.py                 # all three damage cases; needs x01's model.pt and the `claude` CLI
    python wsp.py swap            # one case

Writes runs/<case>.json (every prompt, reply, edit and score) and results.json.
"""
import json
import os
import random
import subprocess
import sys
import time

import torch

from arith import TinyGPT, accuracy, prompt, reward, sample, split

torch.set_num_threads(8)
CLAUDE = os.environ.get("CLAUDE_BIN", "/home/codex/.local/bin/claude")
EDITOR = os.environ.get("EDITOR_MODEL", "opus")
ROUNDS, N_PRACTICE = 4, 300
MODEL_PT = "../x01_the_frozen_model/model.pt"


def damage(case, s):
    """The three ways we break the model. Returns a description kept secret from the editor.
    A suffix after '#' names a repeat run of the same case (flip#2)."""
    case = case.split("#")[0]
    if case == "zero":
        s["blocks.0.ln1.weight"][13] = 0.0
        return "blocks.0.ln1.weight[13] set to 0"
    if case == "flip":
        s["tok.weight"][7] = -s["tok.weight"][7]
        return "embedding row of token '7' (tok.weight[7]) multiplied by -1"
    if case == "swap":
        w = s["tok.weight"]
        w[[3, 8]] = w[[8, 3]]
        return "embedding rows of tokens '3' and '8' (tok.weight[3], tok.weight[8]) swapped"
    raise ValueError(case)


SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "edits": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["set", "scale_row", "swap_rows", "set_row"]},
                "tensor": {"type": "string"},
                "index": {"type": "array", "items": {"type": "integer"}},
                "value": {"type": "number"},
                "row": {"type": "integer"},
                "row_b": {"type": "integer"},
                "factor": {"type": "number"},
                "values": {"type": "array", "items": {"type": "number"}}},
            "required": ["op", "tensor"]}},
        "expected_effect": {"type": "string"}},
    "required": ["diagnosis", "edits", "expected_effect"],
}

INSTRUCTIONS = """You are editing the weights of a small neural network directly, like code. You cannot train it: no
gradients, no optimizer, no data beyond what is shown. Read the weights and the graded answers, work out what is wrong,
and propose a small set of edits. The harness applies your edits, tests them on the practice questions and tells you
the new score; an edit is kept only if the practice score goes up. Edits you can make:
  {"op": "set", "tensor": NAME, "index": [i] or [i, j], "value": x}       one number
  {"op": "scale_row", "tensor": NAME, "row": i, "factor": f}             multiply a row (or a 1-D entry) by f
  {"op": "swap_rows", "tensor": NAME, "row": i, "row_b": j}              exchange two rows
  {"op": "set_row", "tensor": NAME, "row": i, "values": [64 numbers]}    replace a row
Prefer the smallest edit that explains the failures. Reply with JSON only: diagnosis, edits, expected_effect."""

ARCH = """Architecture (PyTorch): character-level GPT for two-digit addition. Vocabulary (13 tokens, index: char):
0-9 = digits '0'-'9', 10 = '+', 11 = '=', 12 = '.'. A question is like "47+38=" and the model must write "85." one
character at a time (greedy). tok.weight[i] is the embedding of token i; pos.weight[t] of position t (0-9).
x = tok[idx] + pos[t]; 2 pre-norm blocks: x = x + attn(ln1(x)); x = x + mlp(ln2(x)) with mlp = Linear(64,256), GELU,
Linear(256,64), attention = 4 heads (nn.MultiheadAttention, in_proj_weight rows 0-63 query, 64-127 key, 128-191 value);
then ln (final LayerNorm) and head = Linear(64, 13). The network was trained to about 99% accuracy and then damaged."""


def fmt_row(v):
    return " ".join(f"{x:+.3f}" for x in v)


def describe(state):
    """Text view of the weights: every small tensor in full, big matrices as per-row statistics."""
    out = []
    for name, t in state.items():
        if "mask" in name:
            continue
        t = t.float()
        head = f"## {name} shape={list(t.shape)} mean={t.mean():+.4f} std={t.std():.4f} min={t.min():+.3f} max={t.max():+.3f}"
        if t.numel() <= 1024:
            if t.dim() == 1:
                out.append(head + "\n" + fmt_row(t.tolist()))
            else:
                out.append(head + "\n" + "\n".join(f"[{i}] {fmt_row(r)}" for i, r in enumerate(t.tolist())))
        else:
            norms = t.norm(dim=1)
            out.append(head + f"\nrow L2 norms (min {norms.min():.3f}, median {norms.median():.3f}, max {norms.max():.3f}): "
                       + " ".join(f"{x:.2f}" for x in norms.tolist()))
    return "\n\n".join(out)


def graded(model, qs):
    _, answers, _, _ = sample(model, qs, greedy=True)
    rows = [(f"{prompt(a, b)}{ans}", reward(a, b, ans), f"{a + b}.") for (a, b), ans in zip(qs, answers)]
    return sum(r for _, r, _ in rows) / len(rows), rows


def report(acc, rows, limit=80):
    wrong = [f"{q}  (right: {right})" for q, r, right in rows if not r]
    return (f"Practice score: {acc:.1%} ({sum(r for _, r, _ in rows)}/{len(rows)}). Wrong answers"
            f" ({len(wrong)}, up to {limit} shown):\n" + "\n".join(wrong[:limit]))


def apply(state, edits):
    s = {k: v.clone() for k, v in state.items()}
    changed = 0
    for e in edits:
        t = s[e["tensor"]]
        before = t.clone()
        if e["op"] == "set":
            t[tuple(e["index"])] = float(e["value"])
        elif e["op"] == "scale_row":
            t[e["row"]] = t[e["row"]] * float(e["factor"])
        elif e["op"] == "swap_rows":
            t[[e["row"], e["row_b"]]] = t[[e["row_b"], e["row"]]]
        elif e["op"] == "set_row":
            t[e["row"]] = torch.tensor(e["values"], dtype=t.dtype)
        changed += int((before != t).sum())
    return s, changed


def ask(text):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))}
    tok = "/home/codex/.config/claude-oauth/token"
    if os.path.isfile(tok):
        env["CLAUDE_CODE_OAUTH_TOKEN"] = open(tok).read().strip()
    cmd = [CLAUDE, "-p", text, "--model", EDITOR, "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
           "--tools", "", "--strict-mcp-config", "--no-session-persistence", "--setting-sources", "",
           "--append-system-prompt", INSTRUCTIONS]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env, cwd="/tmp")
    if p.returncode:
        raise RuntimeError(f"claude exit {p.returncode}: {(p.stderr or p.stdout)[-500:]}")
    r = json.loads(p.stdout)
    out = r.get("structured_output") or json.loads(r["result"][r["result"].index("{"):r["result"].rindex("}") + 1])
    return out, {"seconds": round(time.time() - t0, 1), "cost_usd": r.get("total_cost_usd"),
                 "model": EDITOR, "input_tokens": (r.get("usage") or {}).get("input_tokens")}


def model_of(state):
    m = TinyGPT()
    m.load_state_dict(state)
    return m.eval()


def run(case):
    train, test = split()
    practice = random.Random(7).sample(train, N_PRACTICE)
    base = torch.load(MODEL_PT)
    state = {k: v.clone() for k, v in base.items()}
    secret = damage(case, state)
    log = {"case": case, "damage": secret, "editor": EDITOR,
           "test_intact": accuracy(model_of(base), test), "test_damaged": accuracy(model_of(state), test), "rounds": []}
    acc, rows = graded(model_of(state), practice)
    log["practice_damaged"] = acc
    history = []
    for rnd in range(1, ROUNDS + 1):
        if acc >= 0.99:
            break
        text = (ARCH + "\n\n" + report(acc, rows) + "\n\n" + ("Your earlier attempts:\n" + "\n".join(history) + "\n\n"
                if history else "") + "Current weights:\n" + describe(state))
        reply, meta = ask(text)
        new_state, changed = apply(state, reply["edits"])
        new_acc, new_rows = graded(model_of(new_state), practice)
        kept = new_acc > acc
        log["rounds"].append({"round": rnd, "prompt_chars": len(text), "reply": reply, "meta": meta,
                              "numbers_changed": changed, "practice_after": new_acc, "kept": kept})
        history.append(f"Round {rnd}: diagnosis: {reply['diagnosis'][:600]} | edits: {json.dumps(reply['edits'])[:800]}"
                       f" | practice score {acc:.1%} -> {new_acc:.1%} ({'kept' if kept else 'reverted'})")
        print(f"[{case}] round {rnd}: {acc:.3f} -> {new_acc:.3f} ({'kept' if kept else 'reverted'}), "
              f"{changed} numbers, {meta['seconds']}s: {reply['diagnosis'][:160]}", flush=True)
        if kept:
            state, acc, rows = new_state, new_acc, new_rows
    log["practice_final"] = acc
    log["test_final"] = accuracy(model_of(state), test)
    log["numbers_changed_vs_intact"] = int(sum((state[k] != base[k]).sum() for k in base))
    log["numbers_differing_from_intact_after_damage"] = int(sum((damage_state(case, base)[k] != base[k]).sum() for k in base))
    os.makedirs("runs", exist_ok=True)
    json.dump(log, open(f"runs/{case.replace('#', '_')}.json", "w"), indent=1)
    return log


def damage_state(case, base):
    s = {k: v.clone() for k, v in base.items()}
    damage(case, s)
    return s


def main(cases):
    res = {}
    for case in cases:
        L = run(case)
        res[case] = {k: L[k] for k in ("damage", "test_intact", "test_damaged", "practice_damaged", "practice_final",
                                       "test_final", "numbers_changed_vs_intact", "numbers_differing_from_intact_after_damage")}
        res[case]["rounds"] = [{k: r[k] for k in ("round", "numbers_changed", "practice_after", "kept")} |
                               {"diagnosis": r["reply"]["diagnosis"], "edits": r["reply"]["edits"],
                                "seconds": r["meta"]["seconds"]} for r in L["rounds"]]
    old = json.load(open("results.json")) if os.path.exists("results.json") else {}
    old.update(res)
    json.dump(old, open("results.json", "w"), indent=1)
    print(json.dumps({c: {k: v for k, v in r.items() if k != "rounds"} for c, r in res.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["zero", "flip", "swap"])
