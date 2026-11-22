# How Generative Models Work, episode 14 · Diffusion Transformers

Episode 13's latent diffusion with the U-Net swapped for a small Diffusion Transformer (DiT, Peebles & Xie 2023).
Everything else stays the same: data, autoencoder, frozen CLIP text encoder, noise schedule, loss, sampler, caption
dropout, 40,000 training steps.

```bash
pip install -r requirements.txt
PATCH=2 python3 dit.py                          # ~3 hours → out/dit_p2.pt, results_dit_p2.json, out/dit_p2_*.png
PATCH=2 STEPS=10000 SUFFIX=_10k python3 dit.py  # ~1 hour  → 16 tokens, 10,000 steps
PATCH=1 STEPS=10000 SUFFIX=_10k python3 dit.py  # ~3 hours → 64 tokens, 10,000 steps
```

Needs `../g13_latent_diffusion/out/autoencoder.pt` (episode 13) and `../g12_clip/out/clip.pt` (episode 12).
`out/dit_p2.pt` (the EMA weights of the 40,000-step model) is included. `diffusion.py`, `judge.py`, `shapes.py`,
`clip_model.py`, `train_autoencoder.py` and `latent_diffusion.py` are copies from episodes 6–13.

## The model (`dit.py`)

- **Patchify.** The 8×8×4 latent is cut into 2×2 squares: 16 patches of 2·2·4 = 16 numbers. A linear layer turns
  each patch into a 128-number token, and a learned position embedding says where the patch came from.
- **8 transformer blocks**, 4 heads. Each one has self-attention over the 16 tokens, cross-attention to CLIP's word
  vectors (as in episode 13 and PixArt-α), and an MLP.
- **adaLN-Zero.** Time and the pooled caption vector are summed into one conditioning vector. In every block it sets
  the scale and shift of the LayerNorms and a gate on each residual branch. The gates start at zero, so every block
  starts out as the identity.
- **Unpatchify.** A final adaLN and a linear layer turn each token back into its 2×2×4 patch of predicted noise.
- 2,854,800 parameters, against 2,483,780 for episode 13's cross-attention U-Net. AdamW 2e-4 with 1,000 warm-up
  steps, EMA 0.999.

## Results (2026-11-22)

Same judge, captions and seeds as episode 13. "Follows" means the picture is recognizable and its shape, color and
position all match the caption. "Trained" is all 127 trained captions × 4; "never-seen" is the 8 held-out captions
× 32.

| denoiser, 40,000 steps | final loss | trained, w = 1 | never-seen, w = 1 | trained, w = 3 | never-seen, w = 3 |
|---|---|---|---|---|---|
| U-Net + cross-attention (episode 13) | 0.0122 | 99.4% | 98.8% | 99.2% | 99.2% |
| DiT, 16 tokens | 0.0169 | 97.6% | 96.1% | 99.4% | 98.8% |

At this small scale the U-Net is slightly ahead without guidance, and the two are level with guidance. The DiT's
loss was still falling when training stopped (0.0179 at 38,000 steps, 0.0169 at 40,000). Convolutions build in the
assumption that neighboring positions belong together; the transformer has to learn layout from its position
embedding. That costs it at 2.9M parameters. The DiT paper's point is what happens as the model grows: quality kept
improving with compute (Gflops) across all its sizes, and its largest model, DiT-XL/2, set the state of the art on
ImageNet 256×256 at the time.

**Patch size**, both after 10,000 steps:

| patches | tokens | ms per training step | loss | trained, w = 1 | never-seen, w = 1 |
|---|---|---|---|---|---|
| 2×2 | 16 | 454 | 0.0291 | 95.1% | 90.6% |
| 1×1 | 64 | 1,180 | 0.0301 | 94.1% | 89.8% |

For reference, episode 13's U-Net was at loss 0.0207 after 10,000 steps. Four times as many tokens made each step about three times slower (2.6× in these two runs; 3.2× after scaling each by the U-Net step timed in the same run; 3.8× in a quiet smoke test) and was no better. Self-attention compares every token with
every other, so its cost grows with the square of the token count. On a latent this small, 2×2 patches lose nothing.
In the DiT paper, at 256×256, smaller patches did help: more tokens meant more compute.

**Cost** of one training step (batch 128, CPU), measured in the main run: DiT 315 ms, U-Net 244 ms. The 10,000-step
runs measured 454 vs 304 ms (16 tokens) and 1,180 vs 246 ms (64 tokens) while sharing the machine.

Files: `out/dit_p2_prompts.png` (rows: a red triangle in the center, a blue square at the top left, a yellow circle
at the bottom, a purple triangle at the top left; w = 3), `out/*_samples_w*.png` (the first 16 trained captions),
`out/*_held_out_w*.png` (the first four never-seen captions, 8 pictures each).

**Beyond pictures.** A transformer only sees a list of tokens. OpenAI's Sora report (2024) describes video turned
into "spacetime patches", small blocks across space and time, fed to a diffusion transformer. Stable Diffusion 3
and FLUX also use transformer denoisers.
