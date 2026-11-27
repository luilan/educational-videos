# Reinforcement Learning for LLMs, episode 2 · REINFORCE

REINFORCE (Williams, 1992) on episode 1's tiny GPT: sample 8 answers to each of 32 training questions, grade them
(reward 1 or 0), and minimise `-(reward * log p(answer)).mean()`. No answer key and no baseline (episode 3).

```bash
pip install -r requirements.txt
(cd ../r01_learning_from_a_grade && python grade.py)   # writes base.pt, the starting model
python reinforce.py                                     # about a minute on a CPU; results.json, reinforced.pt
```

Results (seeded, 2026-11-27; 2,000 steps, learning rate 0.0003):

1. First step: 256 answers, 53 right (21%).
2. Greedy accuracy on the 1,000 test questions: 14% → 73% (sampled: 14% → 71%); fastest between steps 500 and 1,250,
   flat after about 1,500.
3. `27+15=`, 64 tries: before 48 ×40, 42 ×23; after 42 ×64.
4. `47+38=`, 64 tries: before 75 ×35, 74 ×20, 81 ×8, 78 ×1 (85 never); after 85 ×58, 84 ×6.
5. Of the 644 test questions never right in 64 tries before training, 445 are right (greedy) after.
