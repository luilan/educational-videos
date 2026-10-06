# Deep Dive, episode 39 · DPO

Direct Preference Optimization on GPT-2 small, on the same task as episode 38 (positive continuations; score = positive
minus negative words). Preference pairs are built once from GPT-2's own samples; training is one loss,
−log σ(β · [(log p − log p_ref)(chosen) − (log p − log p_ref)(rejected)]).

```bash
pip install -r requirements.txt
python dpo.py
```

Results (seeded, about 3 minutes on an 8-thread CPU):

1. 1,536 samples → 215 preference pairs (every sample scoring above 0, paired with a lower-scoring sample of the same
   prompt).
2. DPO, β 0.1, 3 epochs = 51 steps: loss 0.693 → under 0.3.
3. Fresh samples (8 prompts × 8): reward 0.14 → 1.08, KL 3.83, text stays fluent.

For comparison, PPO with β 0.5 (episode 38) reached 0.92 in about 19 minutes.
