# How Generative Models Work, episode 5 · Diffusion I: Adding Noise

The forward process of a diffusion model: 1,000 small steps that shrink a picture and add Gaussian noise until only
noise is left. No training involved.

```bash
pip install -r requirements.txt
python3 forward_noise.py        # about a minute on a CPU; writes results.json and out/*.png
```

`judge.py` and `shapes.py` are copies from episode 3 and episode 1. The judge loads episode 3's weights from
`../g03_variational_autoencoders/out/judge.pt` (in the repository) or trains the same judge if they are missing.

Results (2026-11-21; seeds fixed; pixels in [−1, 1], T = 1,000):

- **Step by step vs one jump.** One picture noised 2,000 times to step 500 (linear schedule, ᾱ = 0.0786): average
  per-pixel spread 0.9595 with 500 small steps, 0.9596 with the jump x_t = √ᾱ·x_0 + √(1 − ᾱ)·ε; the formula predicts
  √(1 − ᾱ) = 0.9599.
- **Linear schedule** (β from 0.0001 to 0.02): signal-to-noise ratio +40 dB at step 1, 0 dB at step 260, −44 dB at
  step 1,000 (ᾱ = 0.00004). 326 steps are below −20 dB. The judge reads shape, color and position correctly for at
  least half the pictures until step 75.
- **Cosine schedule** (Nichol & Dhariwal 2021): 0 dB at step 497; 64 steps below −20 dB; the judge reads half the
  pictures until step 175.
- `out/noising_linear.png`, `out/noising_cosine.png`: one picture at steps 0, 50, 100, 200, 300, 500, 700, 1,000.
