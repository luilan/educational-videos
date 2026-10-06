"""Deep Dive, episode 37: Reward Models.

A reward model scores a (question, answer) pair with one number. Here: Qwen2.5-0.5B, frozen, reads the conversation; a
linear head on its layer-12 hidden state at the last token gives the reward. Trained on pairs where one answer is
preferred, with the Bradley-Terry loss  -log sigmoid(r_chosen - r_rejected).
1. Capitals: the right answer is preferred over a wrong capital. Held-out countries.
2. Sums: the right sum is preferred over a wrong one. Held-out sums.
3. Shortcuts: the same capital pairs, but the preferred answer always ends with "I hope this helps!". Tested normally
   and adversarially (the wrong answer carries the suffix), against a reward model trained without that bias.
"""
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(8)
NAME = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
SUFFIX = " I hope this helps!"
CAPITALS = {"France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Portugal": "Lisbon",
            "Peru": "Lima", "Chile": "Santiago", "Japan": "Tokyo", "China": "Beijing", "India": "New Delhi",
            "Egypt": "Cairo", "Kenya": "Nairobi", "Canada": "Ottawa", "Mexico": "Mexico City", "Brazil": "Brasília",
            "Argentina": "Buenos Aires", "Norway": "Oslo", "Sweden": "Stockholm", "Finland": "Helsinki",
            "Denmark": "Copenhagen", "Poland": "Warsaw", "Austria": "Vienna", "Greece": "Athens", "Turkey": "Ankara",
            "Russia": "Moscow", "Ukraine": "Kyiv", "Ireland": "Dublin", "Belgium": "Brussels", "Netherlands": "Amsterdam",
            "Hungary": "Budapest", "Thailand": "Bangkok", "Vietnam": "Hanoi", "Indonesia": "Jakarta", "Iran": "Tehran",
            "Iraq": "Baghdad", "Cuba": "Havana", "Colombia": "Bogotá", "Venezuela": "Caracas", "Morocco": "Rabat",
            "Nigeria": "Abuja", "Ghana": "Accra", "Ethiopia": "Addis Ababa", "Australia": "Canberra",
            "New Zealand": "Wellington", "South Korea": "Seoul", "Philippines": "Manila", "Pakistan": "Islamabad",
            "Czechia": "Prague", "Romania": "Bucharest", "Bulgaria": "Sofia"}


LAYER = 12


def chat(q, a):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"


def capital_triples(cs, n_wrong):
    """(question, right answer, wrong answer) triples."""
    out = []
    for c in cs:
        for w in random.sample([v for k, v in CAPITALS.items() if k != c], n_wrong):
            out.append((f"What is the capital of {c}?", f"The capital of {c} is {CAPITALS[c]}.", f"The capital of {c} is {w}."))
    return out


def sum_triples(sums):
    out = []
    for a, b in sums:
        off = random.choice([d for d in range(-10, 11) if d])
        out.append((f"What is {a} + {b}?", f"{a} + {b} = {a + b}.", f"{a} + {b} = {a + b + off}."))
    return out


countries = list(CAPITALS)
random.shuffle(countries)
sums = [(random.randint(10, 99), random.randint(10, 99)) for _ in range(340)]
cap_train, cap_test = capital_triples(countries[:30], 3), capital_triples(countries[30:], 3)
sum_train, sum_test = sum_triples(sums[:300]), sum_triples(sums[300:])
cache = {}


@torch.no_grad()
def feat(q, a):
    key = chat(q, a)
    if key not in cache:
        ids = tok(key, return_tensors="pt").input_ids
        cache[key] = model(ids, output_hidden_states=True).hidden_states[LAYER][0, -1]
    return cache[key]


def build(triples, mode="plain"):
    """(chosen, rejected) features. plain: no suffix; clean: suffix on both or neither; biased: on the chosen one only;
    adversarial: on the rejected one only."""
    C, R = [], []
    for q, right, wrong in triples:
        if mode == "clean" and random.random() < 0.5:
            right, wrong = right + SUFFIX, wrong + SUFFIX
        elif mode == "biased":
            right += SUFFIX
        elif mode == "adversarial":
            wrong += SUFFIX
        C.append(feat(q, right))
        R.append(feat(q, wrong))
    return torch.stack(C), torch.stack(R)


def train(C, R, steps=400):
    torch.manual_seed(0)
    head = torch.nn.Linear(C.shape[1], 1)
    mu, sd = torch.cat([C, R]).mean(0), torch.cat([C, R]).std(0) + 1e-6
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=0.1)
    for _ in range(steps):
        loss = -F.logsigmoid(head((C - mu) / sd) - head((R - mu) / sd)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lambda X: head((X - mu) / sd).squeeze(-1).detach()


def accuracy(rm, C, R):
    return (rm(C) > rm(R)).float().mean().item()


print(f"1. capitals: {len(cap_train)} training pairs (30 countries), {len(cap_test)} held-out pairs (20 other countries)")
rm = train(*build(cap_train))
print(f"   right answer preferred: training {accuracy(rm, *build(cap_train)) * 100:.0f}%, "
      f"held-out {accuracy(rm, *build(cap_test)) * 100:.0f}%")

print(f"\n2. sums: {len(sum_train)} training pairs, {len(sum_test)} held-out pairs")
rm_sum = train(*build(sum_train))
print(f"   right answer preferred: training {accuracy(rm_sum, *build(sum_train)) * 100:.0f}%, "
      f"held-out {accuracy(rm_sum, *build(sum_test)) * 100:.0f}%")

print("\n3. a shortcut in the preferences: 'I hope this helps!'")
example = "Hungary" if "Hungary" in countries[30:] else countries[30]
wrong = "Vienna" if example != "Austria" else "Prague"
answers = [f"The capital of {example} is {CAPITALS[example]}.", f"The capital of {example} is {CAPITALS[example]}.{SUFFIX}",
           f"The capital of {example} is {wrong}.", f"The capital of {example} is {wrong}.{SUFFIX}"]
for mode in ("clean", "biased"):
    rm = train(*build(cap_train, mode))
    print(f"   {mode} preferences (suffix {'on both or neither' if mode == 'clean' else 'always on the preferred answer'}):")
    print(f"      held-out, no suffix: {accuracy(rm, *build(cap_test)) * 100:3.0f}% right answer preferred")
    print(f"      held-out, suffix on the wrong answer: {accuracy(rm, *build(cap_test, 'adversarial')) * 100:3.0f}%")
    q = f"What is the capital of {example}?"
    scores = [rm(feat(q, a)[None]).item() for a in answers]
    for a, sc in zip(answers, scores):
        print(f"      reward {sc:+6.2f}  {a!r}")
    print(f"      P(right, plain > wrong + suffix) = sigmoid({scores[0]:.2f} - ({scores[3]:.2f})) = "
          f"{torch.sigmoid(torch.tensor(scores[0] - scores[3])):.2f}")
