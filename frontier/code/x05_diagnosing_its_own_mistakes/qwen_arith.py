"""Shared helpers for episode 4: load Qwen3.5-0.8B on the CPU, make arithmetic problems, grade batched answers."""
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "Qwen/Qwen3.5-0.8B"


def load():
    tok = AutoTokenizer.from_pretrained(NAME)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    model.requires_grad_(False)                 # we only ever edit weights by hand, never train
    return tok, model


def problems(kind, n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if kind == "mul2":
            a, b = rng.randint(11, 99), rng.randint(11, 99)
            out.append((f"{a} * {b}", a * b))
        elif kind == "muladd":
            a, b, c = rng.randint(11, 99), rng.randint(2, 9), rng.randint(100, 999)
            out.append((f"{a} * {b} + {c}", a * b + c))
        elif kind == "add3":
            a, b, c = rng.randint(100, 999), rng.randint(100, 999), rng.randint(100, 999)
            out.append((f"{a} + {b} + {c}", a + b + c))
    return out


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)


@torch.no_grad()
def generate(tok, model, prompts, max_new_tokens=12, batch=32, sample=False, temperature=0.8, seed=0):
    outs = []
    for i in range(0, len(prompts), batch):
        enc = tok([chat(tok, p) for p in prompts[i:i + batch]], return_tensors="pt", padding=True)
        if sample:
            torch.manual_seed(seed + i)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=sample,
                           temperature=temperature if sample else None, top_p=None, top_k=None,
                           pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def ask_numbers(probs):
    return [f"What is {q}? Answer with just the number." for q, _ in probs]


def grade(texts, probs):
    res = []
    for t, (_, ans) in zip(texts, probs):
        t = t.split("=")[-1]                      # "28 * 83 = 2324" -> the part after the last '='
        m = re.search(r"-?\d[\d,]*", t)
        res.append(bool(m) and int(m.group().replace(",", "")) == ans)
    return res


def accuracy(tok, model, probs):
    texts = generate(tok, model, ask_numbers(probs))
    ok = grade(texts, probs)
    return sum(ok) / len(ok), texts, ok
