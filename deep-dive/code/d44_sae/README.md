# Deep Dive, episode 44 · Sparse Autoencoders

A sparse autoencoder (SAE) on GPT-2 small's residual stream after layer 6: 768 numbers per token → 6,144 features
(8×) → 768, trained on GPT-2's activations over Tiny Shakespeare.

f = ReLU(W_enc (x − b_dec) + b_enc),  x̂ = W_dec f + b_dec,  loss = |x − x̂|² + L1 · Σᵢ fᵢ ‖W_dec[:, i]‖

```bash
pip install -r requirements.txt
python sae.py           # reads ../../../how-llms-work/tiny_gpt/input.txt
```

Results (seeded, about 30 minutes on an 8-thread CPU, peak memory about 3.5 GB):

1. Held-out text: 79.9% of variance explained, 30 of 6,144 features active per token on average, no dead features.
2. Next-token loss: GPT-2 4.686; with the SAE reconstruction spliced in at layer 6: 4.903; with the layer replaced by its
   mean: 7.861. The SAE keeps 93% of the loss gap.
3. Random live features, where they fire most: "chief" (feature 4747), "Mess" starting the speaker name Messenger
   (2637), learned words such as arithmetic, history, mechanics, calendar, scripture (766); others are less clean.
4. Share of a unit's top-20 activations that are one token: SAE features 49%, raw residual dimensions 33%; units whose
   top 20 are all one token: SAE 14%, raw 1%.

Choosing L1: a sweep (800 steps each) gave about 292 active features per token at L1 = 1, 24 at 3, and a collapse
(under 2% variance explained) at 8 or more. A first run with a much weaker penalty (0.23) explained 95% of the variance
but kept 923 features active per token: accurate, but not sparse enough to read.
