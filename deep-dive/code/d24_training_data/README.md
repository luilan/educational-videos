# Deep Dive, episode 24 · Where Training Data Comes From

Data size and repetition, the cost of junk, and memorization through repetition, on Tiny Shakespeare.

```bash
pip install -r requirements.txt
python training_data.py
```

Uses GPT-2's tokenizer and trains seven 4-layer tiny GPTs (about 35 minutes on a CPU). It prints:

1. **size**: 1,115,394 characters, 338,025 GPT-2 tokens; 23% of the 32,777 non-empty lines appear more than once
   ("GLOUCESTER:" ×229, …);
2. **junk** replacing part of each batch, loss on clean validation text: 0% 1.644, 25% 1.719, 50% 1.776;
3. **one validation passage repeated in training**:

| copies | passage loss | next character right | written from its first 20 characters |
|---|---|---|---|
| 0 | 1.558 | 48% | 4 / 45 |
| 200 | 0.467 | 94% | 2 / 45 |
| 1,000 | 0.062 | 100% | 45 / 45 |
| 2,000 | 0.033 | 100% | 45 / 45 |

(400 copies, `for every in (5,)`: 0.134, 98%, 45 / 45.) The loss on other text barely moves (1.644 → 1.651–1.657).
