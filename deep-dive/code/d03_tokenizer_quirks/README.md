# Deep Dive, episode 3 · Tokenizer Quirks That Shape Model Behavior

```bash
pip install -r requirements.txt
python quirks.py
```

Downloads the GPT-2 and Qwen2.5 tokenizers, `Qwen2.5-0.5B-Instruct` and GPT-2 small (about 1.5 GB; CPU). It prints:

1. **case and spaces**: `hello`, ` hello`, `Hello`, ` Hello` are four GPT-2 tokens (31373, 23748, 15496, 18435);
   `HELLO` is three pieces;
2. **numbers**: GPT-2 splits `1234567` into `123 | 45 | 67`; Qwen2.5 uses one token per digit;
3. **a trailing space**: for "The capital of France is", " Paris" is 0.302; with a trailing space it falls to rank 28;
4. **glitch tokens**: ` SolidGoldMagikarp` and friends are single GPT-2 tokens; the embeddings closest to the average
   are control characters, broken bytes, `externalToEVA` and `quickShip`, while ` SolidGoldMagikarp` is not flagged by
   this test (rank 14,357 of 50,257).
