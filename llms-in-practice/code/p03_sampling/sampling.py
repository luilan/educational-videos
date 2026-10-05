"""LLMs in Practice, episode 3: Sampling — temperature, top-p, and why answers vary.

Real next-token probabilities from Qwen2.5-0.5B-Instruct (about 1 GB download, runs on a CPU).

    pip install -r requirements.txt
    python sampling.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The cat sat on the"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

enc = tok(PROMPT, return_tensors="pt")
ids = enc.input_ids
with torch.no_grad():
    logits = model(ids).logits[0, -1]          # one score per vocabulary token


def show(probs, k=6):
    top = torch.topk(probs, k)
    return ", ".join(f"{tok.decode([int(i)])!r} {p:.3f}" for p, i in zip(top.values, top.indices))


# 1. Softmax turns scores into probabilities over the whole vocabulary.
print(f"vocabulary: {logits.numel():,} tokens")
probs = torch.softmax(logits, -1)
print("T = 1  :", show(probs))
mat = tok.encode(" mat")[0]
print(f"' mat' {probs[mat]:.4f}, rank {int((probs > probs[mat]).sum()) + 1}")

# 2. Temperature divides the scores before the softmax: low = sharper, high = flatter.
for t in (0.5, 2.0):
    print(f"T = {t}:", show(torch.softmax(logits / t, -1)))

# 3. Top-p keeps the smallest set of top tokens whose probabilities add up to p, then renormalises.
def top_p(probs, p):
    sorted_p, order = torch.sort(probs, descending=True)
    keep = torch.cumsum(sorted_p, -1) - sorted_p < p          # keep a token if the mass before it is < p
    out = torch.zeros_like(probs)
    out[order[keep]] = sorted_p[keep]
    return out / out.sum()

kept = top_p(probs, 0.9)
print(f"top-p 0.9 keeps {int((kept > 0).sum())} of {probs.numel():,} tokens:", show(kept, 4))
print(f"top-p 0.5 keeps {int((top_p(probs, 0.5) > 0).sum())} tokens")

# 4. Greedy always picks the top token; sampling draws from the distribution, so answers vary.
def generate(temperature, seed, n=8):
    torch.manual_seed(seed)
    out = model.generate(ids, attention_mask=enc.attention_mask, max_new_tokens=n, do_sample=temperature > 0, temperature=temperature or None,
                         top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:])

print("greedy        :", repr(generate(0, 0)))
for seed in (1, 2, 3):
    print(f"T=1, seed {seed}  :", repr(generate(1.0, seed)))
