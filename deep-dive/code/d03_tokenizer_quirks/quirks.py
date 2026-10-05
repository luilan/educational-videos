"""How LLMs Work: Deep Dive, episode 3 — Tokenizer quirks that shape model behaviour.

1. The same word, different tokens: case and a leading space.
2. Numbers: GPT-2 cuts them into irregular chunks; Qwen2.5 uses one token per digit.
3. A trailing space at the end of a prompt changes the model's prediction.
4. Glitch tokens: vocabulary entries that almost never appeared in the model's training text.

    pip install -r requirements.txt
    python quirks.py
Downloads the GPT-2 and Qwen2.5 tokenizers and Qwen2.5-0.5B-Instruct (about 1 GB); runs on a CPU.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def pieces(tok, text):
    return [tok.decode([i]) for i in tok(text)["input_ids"]]


# 1. Case and spaces
print("== same word, different tokens (GPT-2)")
for w in ["hello", " hello", "Hello", " Hello", "HELLO", " HELLO"]:
    ids = gpt2(w)["input_ids"]
    print(f"{w!r:>9} -> ids {ids}  {pieces(gpt2, w)}")

# 2. Numbers
print("\n== numbers")
for n in ["1234567", "2024", "3.14159", "1,000,000"]:
    print(f"{n:>10}  GPT-2 {pieces(gpt2, n)}   Qwen2.5 {pieces(qwen, n)}")

# 3. The trailing space
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).eval()
print("\n== a trailing space (Qwen2.5-0.5B, raw text continuation)")
for prompt in ["The capital of France is", "The capital of France is "]:
    ids = qwen(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        probs = torch.softmax(model(ids).logits[0, -1], -1)
    top = torch.topk(probs, 5)
    print(f"{prompt!r}: last token {qwen.decode(ids[0, -1:])!r}")
    print("   next:", ", ".join(f"{qwen.decode([int(i)])!r} {p:.3f}" for p, i in zip(top.values, top.indices)))
    for word in [" Paris", "Paris"]:
        wid = qwen.encode(word)[0]
        print(f"   {word!r}: probability {probs[wid]:.4f}, rank {int((probs > probs[wid]).sum()) + 1}")

# 4. Glitch tokens: GPT-2's vocabulary was built from text that its training data later filtered out.
print("\n== glitch tokens in GPT-2's vocabulary")
for w in [" SolidGoldMagikarp", " TheNitromeFan", " davidjl", " cat"]:
    ids = gpt2(w)["input_ids"]
    print(f"{w!r:>22} -> {len(ids)} token(s), ids {ids}")

# Tokens that were (almost) never seen in training get almost no updates, so their embeddings stay generic.
# One rough way to look for them: the embeddings closest to the average of all embeddings. In GPT-2 small this finds
# control characters, broken UTF-8 fragments and odd strings like "externalToEVA" and "quickShip" (themselves known
# glitch tokens), while " SolidGoldMagikarp" itself is not unusual by this measure: no single test finds them all.
emb = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").transformer.wte.weight.detach()
dist = (emb - emb.mean(0)).norm(dim=1)
order = torch.argsort(dist)
rank = {int(t): r + 1 for r, t in enumerate(order)}
print(f"\ndistance of each token embedding to the average embedding (GPT-2, {len(dist):,} tokens):")
for w in [" SolidGoldMagikarp", " TheNitromeFan", " davidjl", " cat"]:
    t = gpt2(w)["input_ids"][0]
    print(f"{w!r:>22}: distance {dist[t]:.2f}, rank {rank[t]:,} of {len(dist):,} (1 = closest to average)")
print("closest 12 to the average:", [gpt2.decode([int(t)]) for t in order[:12]])
print(f"median distance {dist.median():.2f}")

