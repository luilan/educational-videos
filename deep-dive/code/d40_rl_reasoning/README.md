# Deep Dive, episode 40 · Reinforcement Learning for Reasoning

GRPO (group relative policy optimization) with a verifiable reward on Qwen2.5-0.5B-Instruct. Questions like "What is
37 × 2 + 48?"; the reward is 1 when the answer ends with the right "Answer: N", else 0. No reward model, no value head:
each sampled answer is compared with the other answers in its group of 8.

```bash
pip install -r requirements.txt
python grpo.py          # 30 steps; python grpo.py 10 for a shorter run
```

Results (seeded, about 55 minutes on an 8-thread CPU, peak memory about 9.7 GB):

1. Before training, greedy on 100 held-out questions: reward 0/100, yet the last number in the text is right 95/100
   (the model computes correctly, then writes `\boxed{…}` instead of "Answer: N"); 84/100 finish within 160 tokens;
   mean length 148 tokens.
2. GRPO, 30 steps × 4 questions × 8 samples, lr 2e-6, KL β 0.04: training reward 0.25 at step 1, 0.94 by step 5, then
   between 0.66 and 1.00 (mean of the first 5 steps 0.67, last 5 steps 0.86); KL from the start about 0.1.
3. After training: reward 94/100, last number right 94/100, 100/100 finish, mean length 105 tokens.

RL here taught the answer format and when to stop, not arithmetic: the model already got 95/100 right.
