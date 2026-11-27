# Reinforcement Learning for LLMs, episode 1 · Learning From a Grade

The task for the whole series is two-digit addition (`47+38=` → `85.`, one character per token) with a checker:
reward 1 if the answer is exactly right, else 0. `arith.py` holds the task, the checker and a tiny GPT
(2 blocks, 64-dim, 102,413 parameters); later episodes copy it.

```bash
pip install -r requirements.txt
python grade.py      # about 30 seconds on a CPU; writes results.json and base.pt
```

Results (seeded, 2026-11-27):

1. Pre-trained on 500 worked examples for 2,000 steps: 500/500 right on those, 139/1,000 (14%) on new questions.
2. The policy for `47+38=`: first digit `7` 83%, `8` 17%. It usually forgets the carry.
3. 64 samples for each of the 1,000 test questions: 14% of the 64,000 answers are right. 644 questions are never
   right, 19 always; 356 get at least one right answer; on 217 the greedy answer is wrong but some sample is right.
4. One question in detail, `27+15=`: `48.` ×33, `42.` ×24 (right), `49.` ×7.

`base.pt` is the starting model for episode 2 (REINFORCE), which uses only these grades and no answer key.
