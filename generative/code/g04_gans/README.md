# How Generative Models Work, episode 4 · GANs: a Forger and a Judge

A generator (64 random numbers → a 32×32 picture) and a discriminator (picture → one number: real or fake), trained
against each other on 20,000 Shapes pictures. Samples are scored with the judge from episode 3.

```bash
pip install -r requirements.txt
python3 train_gan.py stable      # ~1 hour on an 8-core CPU → results_stable.json, out/stable_*.png
python3 train_gan.py collapse    # 15 epochs are enough: EPOCHS=15 python3 train_gan.py collapse
python3 train_gan.py dcgan       # the classic recipe, for comparison (not used in the video)
```

- Generator: linear → 256×4×4, then three (upsample ×2, 3×3 conv, batch norm, ReLU) blocks, tanh output. 663,043
  parameters. Discriminator: three stride-2 convolutions (64, 128, 256 channels, leaky ReLU), linear → 1. 662,977.
- `stable`: no batch norm in the discriminator, an **R1 penalty** (γ = 1) on real pictures, Adam (0.0, 0.99), learning
  rate 0.0002 for both, batch 64. Samples come from an **EMA** of the generator's weights (decay 0.999, batch-norm
  statistics averaged too); "raw" = the generator itself.
- `collapse`: the same with a 10× generator learning rate and 5 generator steps per discriminator step.
- `judge.py` loads episode 3's judge weights (`../g03_variational_autoencoders/out/judge.pt`).

Results (2026-11-21):

| | recognized by the judge | combinations covered (of 135) | nearest training picture |
|---|---|---|---|
| real pictures | 99.95% | | |
| `stable`, EMA, after 30 epochs (67 min) | 38.4% | 97 | 0.0038 |
| `stable`, raw generator, epochs 21–30 | 27–45% | 68–98 | |
| `collapse`, raw generator, 15 epochs | up to 73% in one pass | never more than 5 | |
| VAE, episode 3 | 9.4% | | 0.0041 |

`out/collapse_raw_epoch03.png` (every sample a pink dash) and `out/collapse_raw_epoch10.png` (a green square) show the
mode collapse. Earlier attempts with the classic DCGAN recipe (transposed convolutions, batch norm in the
discriminator) gave noisy blobs with collapsed colors and crashed to 3% in their last epoch.
