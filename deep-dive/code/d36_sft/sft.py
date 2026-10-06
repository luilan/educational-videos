"""Deep Dive, episode 36: Supervised Fine-Tuning.

Turn the base Qwen2.5-0.5B into a (tiny) assistant: full fine-tuning on 280 chat-formatted examples of three checkable
tasks (addition, capitals, capital letters), with the loss only on the assistant's tokens. Before and after, on 60
held-out questions: does it answer in the right format, is the answer right, and does it stop?
"""
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(8)
NAME = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32)
END = "<|im_end|>"
END_ID = tok.convert_tokens_to_ids(END)

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
WORDS = ["garden", "window", "silver", "planet", "orange", "button", "candle", "forest", "rocket", "pencil", "bridge",
         "castle", "dragon", "engine", "flower", "guitar", "hammer", "island", "jacket", "kitten", "ladder", "marble",
         "needle", "pepper", "rabbit", "saddle", "tunnel", "violin", "wallet", "yellow", "anchor", "basket", "cotton",
         "dinner", "fabric", "galaxy", "helmet", "insect", "jungle", "lemon", "magnet", "napkin", "oyster", "pillow",
         "quartz", "ribbon", "sponge", "thread", "velvet", "walnut"]


def make(kind, x):
    if kind == "add":
        a, b = x
        return f"What is {a} + {b}?", f"{a} + {b} = {a + b}."
    if kind == "capital":
        return f"What is the capital of {x}?", f"The capital of {x} is {CAPITALS[x]}."
    return f"Write the word '{x}' in capital letters.", f"{x.upper()}"


countries = list(CAPITALS)
random.shuffle(countries)
words = WORDS[:]
random.shuffle(words)
pairs = [(random.randint(10, 99), random.randint(10, 99)) for _ in range(120)]
train = ([make("add", p) for p in pairs[:100]] + [make("capital", c) for c in countries[:30]] * 3 +
         [make("upper", w) for w in words[:30]] * 3)
test = ([("add", make("add", p)) for p in pairs[100:]] + [("capital", make("capital", c)) for c in countries[30:]] +
        [("upper", make("upper", w)) for w in words[30:]])
random.shuffle(train)


def chat(q):
    return f"<|im_start|>user\n{q}{END}\n<|im_start|>assistant\n"


@torch.no_grad()
def evaluate(label, fmt=None):
    model.eval()
    fails = []
    ok_format = ok_answer = stopped = 0
    lengths, samples = [], {}
    for kind, (q, a) in test:
        ids = tok((fmt or chat)(q), return_tensors="pt").input_ids
        out = model.generate(ids, max_new_tokens=40, do_sample=False, eos_token_id=END_ID, pad_token_id=END_ID,
                             attention_mask=torch.ones_like(ids))[0, ids.shape[1]:]
        stop = (out == END_ID).nonzero()
        stopped += len(stop) > 0
        n = stop[0, 0].item() if len(stop) else len(out)
        text = tok.decode(out[:n]).strip()
        if fmt:
            text = text.split("\n")[0].strip()
        lengths.append(n)
        ok_format += text == a
        key = a.split()[-1].rstrip(".")
        ok_answer += key in text
        if text != a:
            fails.append((q, text[:60]))
        samples.setdefault(kind, (q, tok.decode(out)))
    t = len(test)
    print(f"   {label}: exact answer {ok_format / t * 100:.0f}%, right answer somewhere {ok_answer / t * 100:.0f}%, "
          f"stops by itself {stopped / t * 100:.0f}%, mean length {sum(lengths) / t:.0f} tokens")
    for q, out in samples.values():
        print(f"      {q!r} -> {out[:160]!r}")
    if len(fails) <= 5:
        for q, text in fails:
            print(f"      wrong: {q!r} -> {text!r}")


def batch(examples):
    """Tokens of the whole conversation; labels -100 (ignored) everywhere except the assistant's answer and END."""
    seqs, labs = [], []
    for q, a in examples:
        p = tok(chat(q)).input_ids
        r = tok(a + END).input_ids
        seqs.append(p + r)
        labs.append([-100] * len(p) + r)
    L = max(map(len, seqs))
    ids = torch.tensor([s + [END_ID] * (L - len(s)) for s in seqs])
    lab = torch.tensor([l + [-100] * (L - len(l)) for l in labs])
    att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs])
    return ids, lab, att


print(f"train {len(train)} examples, test {len(test)} held-out questions (new numbers, countries, words)")
print(f"example: {chat(train[0][0]) + train[0][1] + END!r}")
print("\n1. before fine-tuning (base model)")
evaluate("base")
print("   the same base model, plain text prompt 'Question: ...\\nAnswer:' (stops at the first newline):")
evaluate("base, plain prompt", fmt=lambda q: f"Question: {q}\nAnswer:")

print("\n2. fine-tuning: loss on the answer tokens only")
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
B, STEPS = 8, 75
t0 = time.time()
for step in range(STEPS):
    model.train()
    ids, lab, att = batch([train[(step * B + i) % len(train)] for i in range(B)])
    logits = model(ids, attention_mask=att).logits[:, :-1]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
    loss.backward()
    opt.step()
    opt.zero_grad()
    if step % 15 == 0 or step == STEPS - 1:
        print(f"   step {step + 1:>3}: answer loss {loss.item():.3f}   ({time.time() - t0:.0f} s)", flush=True)
answer_tokens = (lab != -100).sum().item()
print(f"   last batch: {ids.numel()} tokens, {answer_tokens} of them trained on ({answer_tokens / att.sum().item() * 100:.0f}% "
      f"of the real tokens)")

print("\n3. after fine-tuning")
evaluate("fine-tuned")
