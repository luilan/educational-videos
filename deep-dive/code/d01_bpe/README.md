# Deep Dive, episode 1 · Byte-Pair Encoding, Step by Step

A byte-level BPE tokenizer written from scratch and trained on Tiny Shakespeare (the tiny GPT's training text in this
repository), compared with GPT-2's real tokenizer.

```bash
pip install -r requirements.txt     # only needed for the GPT-2 comparison
python bpe.py
```

It prints:

1. the first merges (`e` + space seen 5,249 times, then `t` + `h`, …) and how the text shrinks (200,000 → 81,132
   tokens after 500 merges);
2. *"To be, or not to be, that is the question."* encoded (16 tokens) and the longest tokens learned, which glue names,
   punctuation and new lines together (`.\n\nCORIOLANUS:\n`);
3. the same with GPT-2-style **pre-tokenization** (split into words first, merge only inside a word): 15 tokens,
   whole-word tokens like ` Senator`;
4. GPT-2 (50,257 tokens): 13 tokens for the line, 32,324 for 100,000 unseen characters.

Try `N_MERGES = 100` or `2000`, or train on your own text.
