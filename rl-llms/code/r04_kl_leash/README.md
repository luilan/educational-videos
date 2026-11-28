# Reinforcement Learning for LLMs, episode 4 · The KL Leash

REINFORCE with the per-question baseline (episode 3), from episode 1's model, with a KL penalty that keeps the model
close to a frozen copy of where it started (the reference). Each sampled answer's reward becomes

    shaped = reward − β · (log p_model(answer) − log p_reference(answer))

The average of that log ratio over the model's own answers is the KL divergence from the reference (in nats per
answer). 2 seeds per β, 2,000 steps, Adam, 32 questions × 8 tries per step.

```bash
pip install -r requirements.txt
python kl_leash.py      # ~17 min: the β sweep at lr 0.0003, then β = 0 and 0.1 at lr 0.003 → results.json
```

`run_full.txt` is the log of the run used in the video.

Results (2026-11-28), final greedy accuracy on 1,000 test questions (start 13.9%), mean of 2 seeds:

| β | accuracy | KL to reference | different answers in 16 tries |
|---|---|---|---|
| 0 | 73.0% (76.2, 69.8) | 7.49 | 1.59 |
| 0.01 | 73.9% | 7.25 | 1.72 |
| 0.03 | 67.3% | 6.29 | 1.91 |
| 0.1 | 22.8% | 1.19 | 2.40 |
| 0.3 | 16.1% | 0.60 | 2.55 |
| 1 | 14.9% | 0.29 | 2.65 (start: 2.66) |

At a 10× learning rate (0.003):

- β = 0 collapses within 150 steps: every question is answered "111." (right for 1.5% of the test set), KL 22.9,
  zero entropy. Once all 8 tries agree, every per-question advantage is 0, so it can never recover.
- β = 0.1 wobbles (answers like "14+." and "." for a while) and recovers: 61.0% at step 2,000, KL 6.9.
