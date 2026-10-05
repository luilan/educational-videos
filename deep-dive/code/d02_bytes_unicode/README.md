# Deep Dive, episode 2 · Bytes, Unicode, and Why Strawberry Is Hard

```bash
pip install -r requirements.txt
python bytes_unicode.py
```

Downloads the GPT-2 and Qwen2.5 tokenizers and `Qwen2.5-1.5B-Instruct` (about 3 GB; CPU, greedy). It prints:

1. UTF-8 bytes for `a` (1), `é` (2), `ж` (2), `中` (3) and 🍓 (4), in binary;
2. one sentence in English, Italian, Russian, Chinese and emoji: characters, bytes, GPT-2 tokens (10, 13, 38, 25, 12)
   and Qwen2.5 tokens (10, 12, 16, 8, 7);
3. how "strawberry" is tokenized (one token with a leading space; `str | aw | berry` without);
4. Qwen2.5-1.5B counting letters in strawberry, bookkeeper, mississippi (right) and nevertheless (3 instead of 4),
   spelling strawberry (misspelled), and counting with one token per letter (says 4).
