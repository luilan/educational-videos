"""LLMs in Practice, episode 1: Prompts Are Just Context.

A chat app shows bubbles. The model sees one long string of tokens. This script builds that string
with a real chat template (Qwen2.5-0.5B-Instruct; only the ~10 MB tokenizer is downloaded, not the model),
then shows the tokens, their IDs and the count.

    pip install -r requirements.txt
    python what_the_model_sees.py
"""
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

messages = [
    {"role": "system", "content": "You are a friendly cooking assistant."},
    {"role": "user", "content": "How long should I boil an egg?"},
    {"role": "assistant", "content": "About 7 minutes for a jammy yolk."},
    {"role": "user", "content": "And for hard-boiled?"},
]

tok = AutoTokenizer.from_pretrained(MODEL)

# 1. The chat becomes ONE document. add_generation_prompt opens the assistant's turn,
#    so the model's natural continuation is the answer.
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=== What the model actually reads ===")
print(text)

# 2. That document becomes token IDs: the only thing the model ever receives.
ids = tok(text)["input_ids"]
print("=== As tokens ===")
print(len(ids), "tokens")
for i in ids[:12]:
    print(f"{i:>7}  {tok.decode([i])!r}")
print("    ...")

# 3. No memory between calls: every new turn resends the whole conversation, so the input keeps growing.
print("=== Tokens sent on each turn ===")
for turn in range(1, len(messages) + 1, 2):
    sent = tok.apply_chat_template(messages[:turn + 1], tokenize=True, add_generation_prompt=True)
    print(f"after {turn // 2 + 1} user message(s): {len(sent)} tokens")
