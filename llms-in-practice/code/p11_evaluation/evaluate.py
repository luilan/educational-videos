"""LLMs in Practice, episode 11: Evaluating LLMs — how we know it's better.

A tiny evaluation harness: 20 easy and 10 harder questions about the (made-up) Bella's Bakery handbook, each with a\ncheckable answer.
Both models get the handbook in their prompt (like RAG), answer every question, and are scored automatically,
first with naive exact matching, then with a fairer check. Greedy decoding, so every number is repeatable.

    pip install -r requirements.txt
    python evaluate.py
Downloads Qwen2.5-0.5B, 1.5B and 3B-Instruct (about 10 GB in total; the 3B model needs about 7 GB of RAM);\nruns on a CPU in about 15 minutes.
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
# Harder: each answer needs two facts and a little reasoning. ("yes"/"no": the answer must start with it.)
HARD = [
    ("I want a cake for 12 people delivered on a Sunday. How much will I pay in total?", ["39"]),
    ("A student buys two sourdough loaves. How much do they pay?", ["10.8", "10,8"]),
    ("Can I order a birthday cake on Thursday and pick it up on Saturday?", ["no"]),
    ("Is the bakery open at 8:00 on a Monday?", ["no"]),
    ("Will you deliver an order of 25 euros?", ["no"]),
    ("I live 7 km from the bakery. Can you deliver to me?", ["no"]),
    ("What is the total for a cake for 12 people delivered on a Tuesday?", ["42"]),
    ("I place a delivery order at 9:30. Will it arrive today?", ["yes"]),
    ("Can someone with a nut allergy safely eat your croissants?", ["no"]),
    ("If I arrive at 8:30 on a Saturday, is the bakery open?", ["no"]),
]
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]


def answers(model_name, tests):
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if "3B" in model_name else torch.float32       # 3B in bfloat16 to fit in RAM
    llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).eval()
    out = []
    for question, _ in tests:
        msg = [{"role": "system", "content": "Answer from the handbook below, in a few words.\n\n" + HANDBOOK},
               {"role": "user", "content": question}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
        gen = llm.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=60, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        out.append(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip())
    return out


def exact(answer, accepted):            # naive: the answer must be exactly the first accepted string
    return answer.strip().lower().rstrip(".") == accepted[0]


def contains(answer, accepted):         # fairer: any accepted string appears in the answer
    a = re.sub(r"\s+", " ", answer.lower())
    if accepted[0] in ("yes", "no"):    # yes/no questions: judge the first word only
        return re.sub(r"[^a-z]", "", a.split(" ")[0]) == accepted[0]
    return any(x in a for x in accepted)


results, hard = {}, {}
for name in MODELS:
    results[name], hard[name] = answers(name, TESTS), answers(name, HARD)
    short = name.split("-")[1]
    ex = sum(exact(a, acc) for a, (_, acc) in zip(results[name], TESTS))
    ok = sum(contains(a, acc) for a, (_, acc) in zip(results[name], TESTS))
    hd = sum(contains(a, acc) for a, (_, acc) in zip(hard[name], HARD))
    print(f"{short}: easy set exact match {ex}/20, contains the answer {ok}/20 | hard set {hd}/10")

print("\nexamples that exact matching marks wrong (1.5B):")
for (q, acc), a in list(zip(TESTS, results[MODELS[1]]))[:3]:
    print(f"  {q} -> {a!r}  (expected exactly {acc[0]!r})")
print("\nhard set (0.5B | 1.5B | 3B):")
for i, (q, acc) in enumerate(HARD):
    print(" ".join("✓" if contains(hard[m][i], acc) else "✗" for m in MODELS), f" {q}  [expected: {acc[0]}]")
    for m in MODELS:
        print(f"      {m.split('-')[1]}: {hard[m][i]!r}")
