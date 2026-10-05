"""How LLMs Work: Deep Dive, episode 2 — Bytes, Unicode, and why "strawberry" is hard.

1. UTF-8: how characters become 1 to 4 bytes.
2. The same sentence in five languages, counted in characters, bytes and tokens (GPT-2 and Qwen2.5).
3. What the model sees of "strawberry", and whether it can count its r's.

    pip install -r requirements.txt
    python bytes_unicode.py
Downloads two tokenizers and Qwen2.5-1.5B-Instruct (about 3 GB); runs on a CPU.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. UTF-8 bytes per character
for ch in ["a", "é", "ж", "中", "🍓"]:
    b = ch.encode("utf-8")
    print(f"{ch!r}: U+{ord(ch):04X} -> {len(b)} byte(s): {' '.join(f'{x:08b}' for x in b)}")

# 2. One sentence, five languages
SENTENCES = {"English": "The cat is sleeping on the warm windowsill.",
             "Italian": "Il gatto dorme sul davanzale caldo.",
             "Russian": "Кошка спит на тёплом подоконнике.",
             "Chinese": "猫在温暖的窗台上睡觉。",
             "Emoji": "🐱💤🪟☀️"}
gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
print(f"\n{'':<8}{'chars':>6}{'bytes':>7}{'GPT-2':>7}{'Qwen2.5':>9}")
for lang, s in SENTENCES.items():
    print(f"{lang:<8}{len(s):>6}{len(s.encode()):>7}{len(gpt2(s)['input_ids']):>7}{len(qwen(s)['input_ids']):>9}")

# 3. Strawberry
for name, tok in [("GPT-2", gpt2), ("Qwen2.5", qwen)]:
    for word in ["strawberry", " strawberry", "s t r a w b e r r y"]:
        ids = tok(word)["input_ids"]
        print(f"{name:<8}{word!r:<24} -> {[tok.decode([i]) for i in ids]}")

llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float32).eval()


def ask(question):
    ids = qwen.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                   return_tensors="pt")
    out = llm.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=80, do_sample=False,
                       pad_token_id=qwen.eos_token_id)
    return qwen.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


print()
# A famous question may simply be memorised; less famous words test whether the model can really count letters.
for word, letter in [("strawberry", "r"), ("nevertheless", "e"), ("bookkeeper", "e"), ("mississippi", "s")]:
    a = ask(f"How many times does the letter {letter} appear in the word {word}? Answer with a number.")
    pieces = [qwen.decode([i]) for i in qwen(f" {word}")["input_ids"]]
    print(f"{word!r} (tokens {pieces}), letter {letter!r}: true count {word.count(letter)}, model: {a!r}")
print()
for q in ["How many times does the letter r appear in the word strawberry? Answer with a number.",
          "Spell the word strawberry letter by letter, then count how many times the letter r appears.",
          "How many times does the letter r appear in: s t r a w b e r r y? Answer with a number."]:
    print(f"Q: {q}\nA: {ask(q)}\n")
