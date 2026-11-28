# Reinforcement Learning for LLMs, episode 5 · PPO: Small, Safe Steps

Same task and starting model as episodes 2–4. Every learner gets the same 400 batches of 32 questions × 8 tries
(102,400 sampled answers), 3 seeds each:

- **REINFORCE**: one update per batch, per-question baseline (episode 3).
- **PPO**: each batch reused for 4 epochs × 4 minibatches = 16 updates. Per-token ratio p_new / p_old, clipped to
  [0.8, 1.2] (take the smaller of the clipped and unclipped objective); a value head (one linear layer on the hidden
  state at '=') learns the expected reward and is the baseline: advantage = reward − value.
- **No clip**: the same 16 updates per batch with the ratio but without the clip.

```bash
pip install -r requirements.txt
python ppo.py        # ~8 min on 16 threads → results.json
```

`run_log.txt` is the log of the run used in the video.

Results (2026-11-28), greedy accuracy on 1,000 test questions after 400 batches (start 13.9%):

| Learner | Learning rate | Runs | Mean |
|---|---|---|---|
| REINFORCE | 0.0003 | 23.4 / 23.9 / 21.5% | 22.9% |
| REINFORCE | 0.001 | 3.3 / 4.6 / 2.2% | 3.4% |
| PPO | 0.0001 | 28.7 / 25.5 / 27.7% | 27.3% |
| PPO, no clip | 0.0001 | 18.5 / 12.9 / 27.9% | 19.8% |
| PPO | 0.0003 | 5.3 / 9.9 / 11.6% | 8.9% |
| PPO, no clip | 0.0003 | 1.8 / 0.0 / 1.0% | 0.9% |

- At 0.0003 the first batch's 16 updates move the model 0.30–0.48 nats (KL to the sampling model) and accuracy drops to
  3–4% within 40 batches; at 0.0001, 0.11–0.12.
- About 12% of answer tokens hit the clip at 0.0001.
- On this CPU PPO takes ~2.7× the time of REINFORCE per batch (16 backward passes instead of 1). For a large model,
  generating the answers is usually the expensive part, which is what PPO saves.
