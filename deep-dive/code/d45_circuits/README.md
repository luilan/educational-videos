# Deep Dive, episode 45 · Circuits

Indirect object identification (IOI) in GPT-2 small: "When Kate and Peter went to the store, Peter gave a drink to" →
Kate. Activation patching (residual stream and single heads) and mean-ablation knockout.

```bash
pip install -r requirements.txt
python circuits.py
```

Results (seeded, a few minutes on an 8-thread CPU):

1. 90 prompts (3 templates × 30 name pairs): right name preferred 100%, mean logit difference 2.68; corrupted prompts
   (second name swapped): −3.30.
2. Residual-stream patching (template 1): through layer 6 the recovered share sits at the repeated name (1.00 → 0.91);
   layer 7 splits 0.65 / 0.41; from layer 8 it is at the last position (0.98–1.00). Saves
   `residual.json` (the video's grid).
3. Head patching at the last position: 8.10 +0.29, 8.6 +0.29, 7.9 +0.23, 9.9 +0.22, 7.3 +0.14, 10.0 +0.08; most negative
   10.7 −0.35.
4. Mean-ablating the top 3 heads: 2.68 → 2.08 (100% right); top 6: 1.64 (93% right); 6 random heads: 2.63.
