# Reinforcement Learning for LLMs, episode 3 · Baselines and Advantages

REINFORCE from episode 1's model with three baselines (advantage = reward − baseline): none, the batch average, and
the average of the 8 tries at the same question. 3 seeds each, 2,000 steps, learning rate 0.0003, Adam.

```bash
pip install -r requirements.txt
python baselines.py          # ~12 min: rewards 0/1, all three baselines, gradient noise at the start
python baselines.py offset   # ~6 min: rewards 1 (wrong) / 2 (right); noise at episode 2's reinforced.pt
```

Results (2026-11-27):

1. Rewards 0/1, final accuracy on 1,000 test questions: no baseline 75.5 / 78.8 / 73.2% (mean 76%), batch
   76.8 / 71.5 / 73.6% (74%), per question 72.4 / 76.9 / 73.1% (74%). Gradient signal-to-noise at the start
   0.25 / 0.23 / 0.25, and at the 73% model 0.65 / 0.62 / 0.67. With rare successes and Adam, the baseline barely
   matters here.
2. Rewards 1/2 ("a point for trying"), no baseline: 14% → 2.4 / 2.4 / 5.5%; accuracy halves within 200 steps.
3. Rewards 1/2 with either baseline: identical, step for step, to the 0/1 run with the same seed.
