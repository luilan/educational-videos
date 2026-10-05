# Episode 3 · Sampling: Temperature, Top-p, and Why Answers Vary

Real next-token probabilities from `Qwen/Qwen2.5-0.5B-Instruct` (about 1 GB download; runs on a CPU in seconds).

```bash
pip install -r requirements.txt
python sampling.py
```

For the prompt *"The cat sat on the"* it prints:

1. the vocabulary size (151,936) and the top tokens: `couch` 8.1%, `bed` 5.7%, … and `mat` only 2.0%, rank 9;
2. the same tokens at **temperature** 0.5 (couch 29.2%) and 2.0 (couch 0.6%);
3. how many tokens **top-p** keeps: 378 for p = 0.9, 19 for p = 0.5;
4. a **greedy** continuation (always the same) and three **sampled** ones (seeds 1–3, all different).

Try it: change `PROMPT` to a factual one such as *"The capital of France is"* and compare how many tokens top-p
keeps (exercise in the study guide). Exact sampled texts can differ across library versions and hardware; the
probabilities should match to about three decimals.
