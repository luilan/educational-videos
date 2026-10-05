# Episode 11 · Evaluating LLMs: How We Know It's Better

A small evaluation harness: 20 easy and 10 harder questions about the made-up Bella's Bakery handbook (`bakery.txt`,
given to the model in its prompt), three models, automatic checks, and every failing answer printed so you can read it.

```bash
pip install -r requirements.txt
python evaluate.py
```

Downloads Qwen2.5-0.5B, 1.5B and 3B-Instruct (about 10 GB; the 3B model is loaded in bfloat16 and needs about 7 GB
of RAM). About 15 minutes on a CPU, greedy decoding.

| model | easy set, exact match | easy set, contains the answer | hard set (automatic) | hard set (by hand) |
|---|---|---|---|---|
| 0.5B | 1 / 20 | 20 / 20 | 3 / 10 | 2 / 10 |
| 1.5B | 1 / 20 | 20 / 20 | 3 / 10 | 1 / 10 |
| 3B | 3 / 20 | 20 / 20 | 6 / 10 | 5 / 10 |

"By hand": an answer counts only if the final answer and its stated reason are both right. Read the printed answers to
see why the automatic and manual scores differ.
