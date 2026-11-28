"""Episode 5: confirm the held-out results on 1,000 fresh problems (seed 3, never used before).

Rebuilds each final model from its logged kept edits (applied in order to the original weights) and scores it.
    python confirm.py     # ~45 min → confirm.json
"""
import json

import torch

from qwen_arith import accuracy, load, problems
from self_rewrite import apply

RUNS = {"original": None, "x05 diagnose + probe": "run_diagnose.json", "x05 probe only": "run_probe.json",
        "x04 self-edits": "../x04_a_model_that_edits_itself/run_self.json",
        "x04 random edits": "../x04_a_model_that_edits_itself/run_random.json"}


def main():
    torch.set_num_threads(10)
    tok, model = load()
    fresh = problems("add3", 1000, 3)
    out = {}
    for name, path in RUNS.items():
        undos = []
        if path:
            for k in json.load(open(path))["kept"]:
                undos.append(apply(model, k["edit"]))
        acc, texts, ok = accuracy(tok, model, fresh)
        out[name] = {"accuracy": acc, "right": ok}
        print(name, acc, flush=True)
        for u in reversed(undos):
            u()
    base = out["original"]["right"]
    for name, v in out.items():
        both = list(zip(base, v["right"]))
        v["gained"] = sum(1 for a, b in both if not a and b)      # wrong before, right after
        v["lost"] = sum(1 for a, b in both if a and not b)
        v.pop("right")
    json.dump({"problems": len(fresh), "seed": 3, "results": out}, open("confirm.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
