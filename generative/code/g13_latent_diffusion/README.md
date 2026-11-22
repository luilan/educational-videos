# How Generative Models Work, episode 13 · Latent Diffusion

Text-to-image diffusion the way Stable Diffusion does it, on Shapes: an autoencoder compresses pictures to small
latents, diffusion runs on the latents, and a caption comes in through episode 12's CLIP text encoder.

```bash
pip install -r requirements.txt
python3 train_autoencoder.py                    # ~70 minutes → out/autoencoder.pt, results_autoencoder.json, out/recon.png
TEXT=words  python3 latent_diffusion.py         # ~2 hours → out/words_*, results_latent_words.json (cross-attention)
TEXT=pooled python3 latent_diffusion.py         # ~1 hour  → out/pooled_*, results_latent_pooled.json (one caption vector)
python3 figures.py                              # ~2 minutes → out/latents.png, figures.json
```

Needs `../g12_clip/out/clip.pt` (episode 12; its text encoder stays frozen). `out/autoencoder.pt`,
`out/words_latent_unet.pt` and `out/pooled_latent_unet.pt` are included; `EVAL=1 TEXT=words python3
latent_diffusion.py` re-scores a trained model without training.

## The autoencoder (`train_autoencoder.py`)

- 32×32×3 pictures (3,072 numbers) ↔ 8×8×4 latents (256 numbers, 12× fewer). Encoder 552,136 parameters,
  decoder 592,643. Episode 10's eight held-out combinations are left out of training.
- Loss: mean squared error plus a very small KL term (weight 1e-6), as in Stable Diffusion's KL autoencoder: the
  latents stay smooth and are scaled to unit standard deviation (×0.306) for diffusion.
- 6,000 steps, batch 128, 68 minutes. On 1,000 test pictures: MSE 0.000117; 99.8% of the reconstructions are
  recognizable and read exactly like their originals (originals: 99.9%).
- Failed settings, kept for comparison: KL weight 1e-4 → 0% recognizable, every picture a blurry blob
  (`results_autoencoder_kl1e-4.json`, `out/recon_kl1e-4.png`); L1 loss without KL, 2,000 steps → 0%
  (`results_autoencoder_l1_nokl_2000.json`, `out/recon_l1_nokl.png`); KL 1e-6 at 2,000 steps → 89.4%.
- `out/latents.png`: four pictures and their four latent channels. Each channel is still a small map, bright where
  the shape is, which is why a convolutional U-Net works on latents.

## Diffusion on the latents (`latent_diffusion.py`)

Episode 6's U-Net (channels 64, 128, 128) on 8×8×4 latents, episode 6's cosine schedule and noise-guessing loss,
episode 7's fresh-noise sampler (50 steps), batch 128, Adam 2e-4, EMA 0.999, 10% of captions dropped for
classifier-free guidance (episode 11). 40,000 steps for both versions. Two ways in for the caption:

- **pooled**: CLIP's single 64-number caption vector is added to the time embedding, like episode 10's labels.
  2,300,484 parameters.
- **words**: also cross-attention (4 heads) after five U-Net blocks (8×8, 4×4, 2×2, 4×4, 8×8): every latent position
  is a query; the keys and values are CLIP's vectors for the caption's individual words. 2,483,780 parameters.

Results (2026-11-22), the episode 3 judge reading the decoded pictures; "follows" = recognizable and the caption's
shape, color and position all right. Trained: all 127 trained captions × 4; never-seen: the 8 held-out captions × 32.

| caption in via | trained, w = 1 | never-seen, w = 1 | trained, w = 3 | never-seen, w = 3 |
|---|---|---|---|---|
| one caption vector | 99.0% | 94.5% | 99.8% | 94.5% |
| cross-attention over words | 99.4% | **98.8%** | 99.2% | **99.2%** |

With so few words one vector almost suffices on trained captions; cross-attention generalizes better to unseen
combinations (in `out/pooled_held_out_w1.png` the never-seen "purple triangle at the top left" comes out
bluish-violet). Training length matters too: the pooled model after 20,000 steps had 98.0% / 80.5%.

**Cost** of one training step (batch 128, CPU, measured in the same run): episode 6's pixel U-Net (1,182,371
parameters) 1,038 ms; the latent U-Net with cross-attention (2,483,780 parameters, twice as many) 261 ms, 4× cheaper.
(The pooled run measured 417 ms vs 69 ms; absolute times depend on the machine's load.)

**The clipping trap.** Pixel samplers clip the clean-picture guess to [−1, 1]. Latents have unit standard deviation
overall but no fixed range: the shape's own values reach about ±12. An earlier version clipped the clean-latent guess
to ±4. Clipping real latents at ±4 before decoding leaves only 46.3% of 1,000 test pictures exactly right (no clip:
99.3%; ±8: 97.9%; only 1.1% of latent values lie beyond ±4, `figures.json`), and it held both models to about 44% (`results_latent_words_clip4.json`, `out/words_prompts_clip4.png`: same trained model, clipped at ±4:
43.5% / 40.6%). The sampler now clips at ±16 (`X0_CLIP`), only as a guard.

**Stable Diffusion** (Rombach et al. 2022, v1 models), for comparison: 512×512×3 pictures ↔ 64×64×4 latents (48×
fewer numbers), a U-Net of about 860 million parameters, CLIP ViT-L/14's text encoder, cross-attention at several
resolutions. The same recipe as here, at scale.

Files: `out/words_prompts.png` (rows: a red triangle in the center, a blue square at the top left, a yellow circle at
the bottom, a purple triangle at the top left; w = 3), `out/*_samples_w*.png` (first 16 trained captions),
`out/*_held_out_w*.png` (first four never-seen captions, 8 each), `out/recon*.png` (originals over reconstructions).
