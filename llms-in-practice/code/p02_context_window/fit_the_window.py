"""LLMs in Practice, episode 2: Context Windows, and Why They Run Out.

1. How big is the window, and how much text is that?
2. What does a full window cost in memory (the KV cache)?
3. When a chat outgrows its budget, keep the system prompt and drop the oldest turns.

    pip install -r requirements.txt
    python fit_the_window.py
Downloads only the tokenizer and config of Qwen/Qwen2.5-0.5B-Instruct (no weights).
"""
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
cfg = AutoConfig.from_pretrained(MODEL)

# 1. The window: a hard maximum number of tokens per request (prompt + reply).
window = cfg.max_position_embeddings
book = Path(__file__).resolve().parents[3] / "how-llms-work" / "tiny_gpt" / "input.txt"  # Tiny Shakespeare
n = len(tok(book.read_text(), verbose=False)["input_ids"])
print(f"context window: {window:,} tokens")
print(f"Tiny Shakespeare: {n:,} tokens = {n / window:.1f} windows")

# 2. Memory: every token in the window keeps a key and a value per layer (the KV cache, How LLMs Work ep. 13).
head_dim = cfg.hidden_size // cfg.num_attention_heads
per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2   # K and V, 2 bytes each (bf16)
print(f"KV cache: {per_token:,} bytes per token, {per_token * window / 2**20:,.0f} MiB for a full window")


# 3. Overflow: keep the system prompt, drop the oldest turns until the chat fits the budget.
def count(messages):
    return len(tok.apply_chat_template(messages, add_generation_prompt=True))


def fit(messages, budget):
    system, turns = messages[:1], messages[1:]
    while count(system + turns) > budget and len(turns) > 1:
        turns = turns[2:] if len(turns) > 2 else turns[1:]   # drop one user + assistant exchange
    return system + turns


chat = [{"role": "system", "content": "You are a friendly cooking assistant."}]
for i, (q, a) in enumerate([("How long should I boil an egg?", "About 7 minutes for a jammy yolk."),
                            ("And for hard-boiled?", "About 10 minutes."),
                            ("Can I eat them cold?", "Yes, keep them in the fridge for up to a week."),
                            ("What about poached eggs?", "Simmer them for about 3 minutes.")], 1):
    chat += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
chat.append({"role": "user", "content": "Which one is best for a salad?"})

budget = 100
kept = fit(chat, budget)
print(f"\nchat: {count(chat)} tokens, budget {budget}")
print(f"kept: {count(kept)} tokens, {len(kept)} of {len(chat)} messages")
for m in kept:
    print(f"  {m['role']:>9}: {m['content']}")
