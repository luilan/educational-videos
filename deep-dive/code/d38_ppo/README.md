# Deep Dive, episode 38 · RLHF with PPO

PPO from scratch on GPT-2 small (with a value head). Reward: positive minus negative words in a 24-token continuation, a
stand-in for a reward model. Each token also pays β · (log p_policy − log p_reference), a KL penalty to the original
GPT-2.

```bash
pip install -r requirements.txt
python ppo.py            # β = 0, 0.05 and 0.5 (about 20 minutes each on an 8-thread CPU)
python ppo.py 0.5        # one setting
```

Results (seeded; 200 iterations of 16 samples; fresh-sample reward over 8 prompts × 8 samples; KL from the last batch):

| β | reward | KL | text |
|---|---|---|---|
| plain GPT-2 | 0.14 | 0 | normal |
| 0 | 24.00 | 58.9 | “wonderful good wonderful good great great …” |
| 0.05 | 24.00 | 24.7 | “great great great great …” |
| 0.5 | 0.92 | 5.7 | fluent, more positive (“…one of the great films of 2014…”) |
