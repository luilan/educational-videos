# How Generative Models Work, episode 8 · The Denoiser: U-Nets and Time

What do the U-Net's two special parts buy? Episode 6's run, repeated twice with one part removed each time:
**no skip connections** between the down and up halves, and **no time input** (the network is never told the step t).

```bash
pip install -r requirements.txt
python3 train_ablation.py noskip   # ~5 hours with 4 threads (THREADS=4), ~2.5 with 8 → results_noskip.json, out/noskip_*
python3 train_ablation.py notime   # same → results_notime.json, out/notime_*
python3 eval_full.py               # noise error by level for episode 6's full model → results_full.json
python3 eval_samplers.py           # all three with both of episode 7's samplers → results_samplers.json
```

Everything else matches episode 6 exactly (same data and seed, 20,000 steps, batch 128, Adam 0.0002, EMA 0.999).
`noskip` drops only the long connections across the U; the residual connections inside each block stay. `notime`
feeds step 0 to the time embedding every time, so the network must judge the noise level from the picture alone.
The full model's weights are `../g06_learning_to_denoise/out/unet_ema.pt`; `diffusion.py`, `judge.py` and
`shapes.py` are copies from episode 6 (the judge loads `../g03_variational_autoencoders/out/judge.pt`).

Results (2026-11-22; 500 samples from the same starting noise, 50 sampler steps, scored by episode 3's judge):

| network | parameters | final training loss | DDIM (η = 0) | combinations | fresh noise (η = 1) | combinations |
|---|---|---|---|---|---|---|
| full U-Net (episode 6) | 1,182,371 | 0.0028 | 70.0% | 127 | 95.2% | 130 |
| no skip connections | 1,125,763 | 0.0086 | 0.0% | 0 | 8.4% | 33 |
| no time input | 1,182,371 | 0.0037 | 1.4% | 7 | 32.8% | 101 |

Noise-prediction error (mean squared error of ε̂ on 256 fresh test pictures) by noise level:

| step t | 10 | 50 | 100 | 200 | 400 | 600 | 800 | 950 |
|---|---|---|---|---|---|---|---|---|
| full | 0.0114 | 0.0029 | 0.0017 | 0.0010 | 0.0011 | 0.0018 | 0.0023 | 0.0007 |
| no skips | 0.0261 | 0.0088 | 0.0077 | 0.0089 | 0.0113 | 0.0094 | 0.0050 | 0.0011 |
| no time | 0.0140 | 0.0040 | 0.0026 | 0.0016 | 0.0016 | 0.0023 | 0.0027 | 0.0011 |

- **No skips:** every detail has to pass through the 8×8 middle. The error is 2–10× the full model's at low and middle
  noise (4.6× at step 100), and samples are blobs with ragged edges (`out/noskip_samples_step20000.png`,
  `out/noskip_eta1_samples.png`).
- **No time:** one denoising step works almost as well as the full model (at most 1.6× the error; the network can
  tell how noisy a picture is by looking), but sampling breaks: many samples are still noisy at the end
  (`out/notime_samples_step20000.png`, `out/notime_eta1_samples.png`). Small misjudgments of the noise level make the
  sampler remove the wrong amount, step after step. Sun et al. (2025, “Is Noise Conditioning Necessary for Denoising
  Generative Models?”) saw the same pattern at larger scale: many models degrade only modestly without the time
  input, and deterministic DDIM sampling fails worst.
- Checkpoints at 2,000, 5,000 and 10,000 steps are in the JSON files and `out/<variant>_*_stepNNNNN.png`; neither
  variant made any recognizable DDIM samples before step 20,000.
