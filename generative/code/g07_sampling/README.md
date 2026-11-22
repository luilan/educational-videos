# How Generative Models Work, episode 7 · Sampling: From Noise to Picture

Episode 6's trained U-Net, unchanged, sampled in different ways: with and without fresh noise, in 10 to 1,000 steps.

```bash
pip install -r requirements.txt
python3 sample_compare.py      # ~1 hour on an 8-core CPU → results.json and out/*.png
```

Needs `../g06_learning_to_denoise/out/unet_ema.pt` (in the repository) and episode 3's judge weights.

One sampler covers both classics. From step t to a lower step s: predict the noise ε̂, guess the clean picture
x̂₀ = (x_t − √(1 − ᾱ_t)·ε̂)/√ᾱ_t (clipped to [−1, 1]), then
x_s = √ᾱ_s·x̂₀ + √(1 − ᾱ_s − σ²)·ε̂ + σ·z with σ = η·√((1 − ᾱ_s)/(1 − ᾱ_t))·√(1 − ᾱ_t/ᾱ_s).
η = 0 is DDIM (deterministic); η = 1 adds fresh noise (with 1,000 steps it is exactly DDPM's ancestral sampler).

Results (2026-11-22; 500 samples per setting, all from the same starting noise; judge from episode 3):

| steps | η = 1: recognized | combinations | η = 0 (DDIM): recognized | combinations | time per picture |
|---|---|---|---|---|---|
| 10 | 85.0% | 127 | 16.4% | 56 | 20 ms |
| 20 | 95.4% | 132 | 45.8% | 100 | 38 ms |
| 50 | 95.2% | 130 | 70.0% | 127 | 96 ms |
| 100 | 98.4% | 130 | 75.4% | 130 | 0.2 s |
| 250 | 95.2% | 127 | 69.4% | 118 | 0.5 s |
| 1,000 | 96.0% | 129 | 19.2% | 53 | 2.0 s |

(Time: 8 CPU threads, batches of 500.) DDIM at 1,000 steps accumulates small artifacts (`out/samples_eta0_steps1000.png`);
our network is small and its predictions are imperfect, and without fresh noise the errors are never washed out.
In large, well-trained models DDIM usually does much better.

- **When is a sample decided?** DDPM with 1,000 steps, 480 samples the judge reads at the end: at t = 800 the clean
  guess already shows the final position 88% of the time (color 73%, shape 52%); at t = 500, 100% / 99% / 95%.
  `out/trajectory.png`: one sample's x_t (top) and clean guess (bottom) at t = 1,000, 800, 600, 500, 400, 300, 200, 1.
- **Same starting noise** (`out/same_noise.png`, rows: DDIM 10, 20, 50, 1,000 steps; DDPM twice): DDIM keeps colors
  and places across 10–50 steps; the two DDPM runs read differently for all 8 pictures; DDIM-1,000 drifts (1 of 8
  reads the same as DDIM-50).
