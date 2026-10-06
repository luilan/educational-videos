# Deep Dive, episode 34 · Speculative Decoding

A small draft model (Qwen2.5-0.5B) guesses k tokens; the target (Qwen2.5-1.5B) checks them in one pass. Greedy, with KV
caches cropped after every round.

```bash
pip install -r requirements.txt
python speculative.py
```

Needs about 8 GB of RAM. On an 8-thread CPU, float32 (times vary):

1. One target pass after 100 tokens: 1 token 332 ms, 5 tokens 199 ms, 9 tokens 242 ms. Draft: 131 ms per token.
2. 128 new tokens, output identical to plain greedy decoding in every run:
   - Python code: plain 44.0 s; k = 4: 96% of guesses accepted, 4.7 tokens per pass, 19.0 s (2.31x).
   - Prose: plain 42.8 s; k = 2: 63%, 1.46x; k = 4: 54%, 1.37x; k = 6: 37%, 0.93x; k = 8: 32%, 0.91x.
3. Sampling rule, 1,000,000 simulated draws: target (0.50, 0.30, 0.15, 0.05), draft (0.25, 0.50, 0.20, 0.05), result
   (0.501, 0.299, 0.150, 0.050); 75.0% accepted = Σ min(p, q).
