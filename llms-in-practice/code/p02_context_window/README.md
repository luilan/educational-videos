# Episode 2 · Context Windows, and Why They Run Out

A model can read only so many tokens at once. This script measures the window, its memory cost, and shows the
most common fix when a chat outgrows it.

```bash
pip install -r requirements.txt
python fit_the_window.py
```

It downloads only the tokenizer and config of `Qwen/Qwen2.5-0.5B-Instruct` (no model weights) and prints:

1. the **context window**: 32,768 tokens; Tiny Shakespeare (the tiny GPT's training text, in this repository) is
   301,829 tokens, 9.2 windows;
2. the **KV-cache** memory: 12,288 bytes per token, 384 MiB for one full window;
3. a 10-message chat (133 tokens) **trimmed to a 100-token budget**: the system prompt stays, the two oldest
   exchanges are dropped (82 tokens, 6 messages).

Try it: lower the budget, or change `fit` to summarise the dropped turns instead of deleting them.
