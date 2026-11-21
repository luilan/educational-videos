# How Generative Models Work, episode 6 · Diffusion II: Learning to Denoise

Train a network to predict the noise that was added to an example, then generate by starting from pure noise and
removing the predicted noise step by step. First on 2D points, then on 32×32 Shapes pictures.

```bash
pip install -r requirements.txt
python3 train_diffusion.py toy       # 2D spiral, tiny MLP, ~2 minutes → results_toy.json
python3 train_diffusion.py shapes    # U-Net, 20,000 steps, ~2.3 hours on an 8-core CPU → results_shapes.json, out/*.png
STEPS=5000 python3 train_diffusion.py shapes   # a quicker, weaker run
```

- `diffusion.py`: the cosine noise schedule (T = 1,000), the U-Net (with a time embedding; optional class labels for
  episodes 10–11), the training loss, and two samplers (DDPM with all 1,000 steps, DDIM with fewer). Episodes 7–8
  reuse it.
- Training: random step t, random noise ε, episode 5's jump, mean squared error between the predicted and the true ε.
  Adam, learning rate 0.0002, batch 128; samples come from an EMA of the weights (decay 0.999).
- `out/unet_ema.pt`: the trained Shapes model (EMA weights), used by episode 7.
- `judge.py` and `shapes.py` are copies from episodes 3 and 1; the judge loads
  `../g03_variational_autoencoders/out/judge.pt`.

Results (2026-11-21):

**2D spiral** (MLP with three hidden layers of 256, 6,000 steps, DDPM sampling with 1,000 steps): loss 0.548 → 0.393;
median distance from 2,000 generated points to the nearest training point 0.007 (pure noise: 0.188).

**Shapes** (U-Net, 1,182,371 parameters; 500 samples drawn with DDIM in 50 steps, scored by episode 3's judge):

| training steps | minutes | loss | recognized | combinations (of 135) | nearest training picture |
|---|---|---|---|---|---|
| 500 | 5 | 0.0542 | 0% | 0 | 0.1609 |
| 2,000 | 16 | 0.0175 | 0% | 0 | 0.0576 |
| 5,000 | 37 | 0.0085 | 0% | 0 | 0.0092 |
| 10,000 | 70 | 0.0043 | 15.2% | 56 | 0.0035 |
| 20,000 | 138 | 0.0028 | 70.0% | 127 | 0.0015 |
| GAN, episode 4 | | | 38.4% | 97 | 0.0038 |
| VAE, episode 3 | | | 9.4% | | 0.0041 |

Real pictures: 99.95% recognized. `out/denoise_step20000.png` shows, for one picture at t = 50, 200, 400, 600 and
900 (columns): the noisy picture, the true noise, the predicted noise, and the picture with the predicted noise
removed in one go. Up to t = 600 that recovers the triangle; at t = 900 it is a dim blur.
