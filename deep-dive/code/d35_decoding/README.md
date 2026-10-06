# Deep Dive, episode 35 · Decoding Strategies Compared

GPT-2 small continues 4 story openings for 120 tokens with 8 decoding strategies (own sampling loop; beam search via
`generate`). Repetition = share of 4-token phrases already used in the same text; judge surprise = Qwen2.5-1.5B's mean
loss on the continuation (lower = more predictable, not necessarily better).

```bash
pip install -r requirements.txt
python decoding.py
```

| strategy | repetition | judge surprise |
|---|---|---|
| greedy | 69% | 0.67 |
| beam search, 4 beams | 76% | 0.57 |
| temperature 0.7 | 4% | 2.50 |
| temperature 1.0 | 0% | 4.91 |
| temperature 1.5 | 0% | 9.10 |
| top-k 40 | 2% | 3.10 |
| top-p 0.9 | 0% | 3.87 |
| min-p 0.1 | 10% | 2.28 |

Tokens kept after “…opened the door and” (top token 0.066) vs after “…a black suit and a black” (0.156): top-k 40 / 40,
top-p 608 / 275, min-p 28 / 8. Seeded, so the numbers repeat exactly on the same software versions.
