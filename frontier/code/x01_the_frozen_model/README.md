# The Frontier, episode 1 · The Frozen Model

A trained model is a file of numbers. This folder trains the tiny addition GPT from Reinforcement Learning for LLMs
(`arith.py`: 2 blocks, 64-dim, 102,413 parameters) on 7,100 questions, then zeroes every one of its numbers in turn.

```bash
pip install -r requirements.txt
python frozen.py     # ~13 min on 12 cores: model.pt, drops.npy, results.json
python recheck.py    # do the "improving" edits hold on the other 750 test questions?
python picture.py    # out/weights.png (one 64×256 table)
```

Results (seeded, 2026-11-27):

1. The model: 99.2% of 1,000 new questions right; `model.pt` is 418,615 bytes, 102,413 numbers in 30 tables.
2. One edit: `blocks.0.ln1.weight[13]` = 0.81 → 0 drops accuracy to 77.7%; `27+15=` now gives `41.`.
3. All 102,413 single-number zeroings, tested on 250 questions: 96,699 no change, 5,163 raise the score, 388 cost
   under a point, 150 cost 1–10 points, 13 cost 10–50 points, none more. The 163 that cost a point or more are in the
   embeddings (101), the normalization scales (61) and the output layer (1); none of the 99,456 attention and MLP
   numbers matters on its own.
4. The 5,163 "improvements": 5,152 by exactly one question of 250, 11 by two. The best 20 on the other 750 test
   questions: 4 better, 9 the same, 7 worse. Noise.

Papers: Language Models Can Rewrite Their Own Weights (arXiv 2611.21885, 16 Nov 2026); Self-WSP (arXiv 2611.28833,
21 Nov 2026).
