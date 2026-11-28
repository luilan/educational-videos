# Reinforcement Learning for LLMs, episode 7 · Rewards From Preferences

When there is no checker, RLHF asks people which of two answers is better and trains a **reward model** on those
comparisons (Bradley–Terry: P(A preferred over B) = sigmoid(r(A) − r(B))).

- Data: 400 two-step word problems (`word_problems.py`), 4 answers each from Qwen2.5-0.5B-Instruct at temperature
  0.7: 2 with the question alone (the model explains step by step) and 2 with "Answer in one short sentence."
  appended (the reward model never sees that instruction). 1,600 answers.
- The "people" are a script, so their rule is known: prefer the correct answer (last number = the right result); if
  both are right or both wrong, prefer the shorter; disagree with that rule 10% of the time (flipped label).
- Reward model: Qwen2.5-0.5B-Instruct, frozen, plus one trained linear layer (897 numbers) on the last token's hidden
  state. Trained on the 1,752 pairs of 300 questions, tested on the 590 pairs of the other 100.

```bash
pip install -r requirements.txt
for k in 0 1 2 3; do python make_data.py $k 4 & done; wait; python make_data.py merge 4   # ~35 min → answers.json
python reward_model.py                                                                   # ~10 min → results.json, rm_head.pt
```

Results (2026-11-28):

- Answers: short 16.5% right (18 words on average), long 90.8% right (113 words). 53.6% right overall.
- Held-out pairs: the reward model agrees with the (noisy) labels 73.1% of the time; a perfect model would reach
  about 90%. It agrees with the noise-free rule 75.6%: 90.1% when one answer is right and the other wrong, 55.5% when
  both are equally right or wrong (where the rule decides by length): close to a coin flip.
- Mean reward: right and short +1.33, right and long +0.64, wrong and long −0.46, wrong and short −0.88. Among wrong
  answers it prefers longer ones, the opposite of the rule: in the data, long answers were usually right.
- Example blind spot: "Yuki adds a total of 20 more books to the shelf. So now there are 61 books on the shelf."
  (wrong, the answer is 63) scores +1.13, above a correct step-by-step solution (+0.95).
- `rm_head.pt` is the reward model episode 8 optimizes against.
