# Reinforcement Learning for LLMs, episode 6 · GRPO: Grading on a Curve

GRPO (Group Relative Policy Optimization, DeepSeekMath, arXiv 2402.03300, Feb 2024; used for DeepSeek-R1) keeps PPO's
ratio and clip but drops the value model. Each question gets a group of G answers and

    advantage = (reward − group mean) / (group standard deviation + 0.0001)

with a KL penalty to the frozen starting model in the loss (β = 0.04, the "k3" estimator, as in DeepSeekMath).

Same task, starting model and budget as episode 5: 400 batches of 256 answers, 4 epochs × 4 minibatches per batch,
step 0.0001, 3 seeds each.

```bash
pip install -r requirements.txt
python grpo.py          # PPO, GRPO with G = 2, 4, 8, 16, GRPO without the std division   (~11 min, 4 processes)
python grpo.py extra    # G = 8 without the KL term, and with DAPO-style dynamic sampling     (~8 min)
```

`run_log.txt` is the log of the runs used in the video.

Results (2026-11-28), accuracy on 1,000 test questions after 400 batches (start 13.9%):

| Method | Runs | Mean | Groups with all-equal rewards |
|---|---|---|---|
| PPO (value head) | 27.6 / 27.8 / 29.1% | 28.2% | n/a |
| GRPO, G = 2 | 13.6 / 16.7 / 17.2% | 15.8% | 92% |
| GRPO, G = 4 | 19.3 / 16.4 / 18.4% | 18.0% | 84% |
| GRPO, G = 8 | 15.4 / 16.8 / 18.8% | 17.0% | 75% |
| GRPO, G = 16 | 16.8 / 17.6 / 17.3% | 17.2% | 70% |
| GRPO, G = 8, no KL term | 26.8 / 23.2 / 27.3% | 25.8% | |
| GRPO, G = 8, dynamic sampling | 28.5 / 32.0 / 21.7% | 27.4% | 0% (filtered); 434k–465k answers generated, 4.4× |
| GRPO, G = 8, no std division | 16.2 / 16.0 / 8.2% | 13.5% | |

- A group whose answers all get the same reward has zero advantage everywhere and teaches nothing. Our model is very
  consistent per question, so most groups are empty, even with 16 answers.
- The KL term at DeepSeekMath's β cost most of the gap to PPO in this short run on a tiny model.
- Parameters: the policy has 102,413; our value head adds 65. For a language model the value model is usually a
  second network as large as the policy, which is the cost GRPO removes.
