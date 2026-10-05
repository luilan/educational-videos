"""LLMs in Practice, episode 11: Evaluating LLMs — how we know it's better.

A tiny evaluation harness: 20 questions about the (made-up) Bella's Bakery handbook, each with a checkable answer.
Both models get the handbook in their prompt (like RAG), answer every question, and are scored automatically,
first with naive exact matching, then with a fairer check. Greedy decoding, so every number is repeatable.

    pip install -r requirements.txt
    python evaluate.py
Downloads Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct (about 4 GB in total); runs on a CPU in a few minutes.
"""
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HANDBOOK = (Path(__file__).parent / "bakery.txt").read_text()
# (question, accepted answers): an answer counts as correct if it contains any accepted string.
TESTS = [
    ("What time does the bakery open on weekdays?", ["7:30"]),
    ("What time does the bakery open on weekends?", ["9:00", "9"]),
    ("On which day is the bakery closed?", ["monday"]),
    ("When does the bakery close on public holidays?", ["13:00", "1 pm", "1pm", "1:00"]),
    ("How much does a sourdough loaf cost?", ["6 euro", "€6", "6€", "six euro"]),
    ("On which days is rye bread baked?", ["tuesday"]),
    ("On which day is gluten-free bread available?", ["friday"]),
    ("How many days in advance must a birthday cake be ordered?", ["three", "3"]),
    ("How much is a cake for 8 people?", ["28"]),
    ("How much is a cake for 12 people?", ["39"]),
    ("Does writing a message on a cake cost extra?", ["free", "no"]),
    ("What do wedding cakes require?", ["tasting"]),
    ("What is the maximum delivery distance?", ["5 km", "5km", "five"]),
    ("What is the minimum order for delivery?", ["30"]),
    ("How much does delivery cost?", ["3 euro", "€3", "3€", "three euro"]),
    ("On which day is delivery free?", ["sunday"]),
    ("Which nut is used in the kitchen?", ["almond"]),
    ("What discount do students get?", ["10 percent", "10%", "ten percent"]),
    ("What discount code works for online orders?", ["bella10"]),
    ("What time does the morning baker's shift start?", ["4:00", "4 am", "4am"]),
]
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]


def answers(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    out = []
    for question, _ in TESTS:
        msg = [{"role": "system", "content": "Answer from the handbook below, in a few words.\n\n" + HANDBOOK},
               {"role": "user", "content": question}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
        gen = llm.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=30, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        out.append(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip())
    return out


def exact(answer, accepted):            # naive: the answer must be exactly the first accepted string
    return answer.strip().lower().rstrip(".") == accepted[0]


def contains(answer, accepted):         # fairer: any accepted string appears in the answer
    a = re.sub(r"\s+", " ", answer.lower())
    return any(x in a for x in accepted)


results = {}
for name in MODELS:
    results[name] = answers(name)
    short = name.split("-")[1]
    ex = sum(exact(a, acc) for a, (_, acc) in zip(results[name], TESTS))
    ok = sum(contains(a, acc) for a, (_, acc) in zip(results[name], TESTS))
    print(f"{short}: exact match {ex}/20, contains the answer {ok}/20")

print("\nper question (0.5B | 1.5B):")
for i, (q, acc) in enumerate(TESTS):
    marks = " ".join("✓" if contains(results[m][i], acc) else "✗" for m in MODELS)
    print(f"{marks}  {q}")
    for m in MODELS:
        if not contains(results[m][i], acc):
            print(f"      {m.split('-')[1]} said: {results[m][i]!r}")
print("\nexamples that exact matching marks wrong:")
for m in MODELS[1:]:
    for (q, acc), a in list(zip(TESTS, results[m]))[:4]:
        print(f"  {q} -> {a!r}")
