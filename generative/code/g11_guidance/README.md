# How Generative Models Work, episode 11 · Classifier-Free Guidance

Episode 10's conditional model, sampled with classifier-free guidance at different scales: once fully trained, and
once stopped early.

```bash
pip install -r requirements.txt
python3 guidance.py                                        # ~40 minutes; episode 10's model → results.json, out/w*.png
STEPS=8000 NAME=mid python3 train_cond.py                  # ~85 minutes; keeps out/mid_{2000,4000,6000,8000}_ema.pt
python3 scan_checkpoints.py                                # ~30 minutes → results_scan.json
CKPT=out/mid_6000_ema.pt NAME=mid6k python3 guidance.py    # ~40 minutes → results_mid6k.json, out/mid6k_w*.png
python3 figures.py                                         # ~10 minutes → out/fixes.png, ladder.png, predictions.png
python3 when_to_guide.py                                   # ~10 minutes → results_when.json
```

`guidance.py` needs `../g10_conditioning/out/cond_ema.pt` (episode 10). `out/mid_6000_ema.pt`, the half-trained
model, is included; the other checkpoints come from `train_cond.py` (here a copy of episode 10's, which can also
stop early and keep checkpoints).

**Classifier-free guidance** (Ho & Salimans, 2022). Label dropout (episode 10) made one network both a conditional
and an unconditional model. At every sampling step, ask it twice, with the request and with nothing said, and mix:

    noise guess = unconditional + w · (conditional − unconditional)

w = 0 ignores the request, w = 1 is plain conditioning, w > 1 exaggerates whatever the request changes. Both
guesses go through the network as one batch, so each step costs two network evaluations. (Ho & Salimans write the
same thing as (1 + w′)·conditional − w′·unconditional, so their w′ = our w − 1.)

All samples: 50 steps with fresh noise (η = 1, episode 7), the same starting noise for every w. The episode 3 judge
reads each picture; "follows" = recognizable and shape, color and position all as requested. Trained combinations:
all 127, twice each (254 pictures); never-seen: episode 10's 8 held-out combinations, 32 each (256). "Late" = guided
only while t < 500 (the less noisy half of sampling), plain conditioning (w = 1) before.

## Results (2026-11-22)

**Half-trained model** (6,000 of 20,000 steps, loss 0.0056 vs 0.0026), `results_mid6k.json`:

| w | recognizable | follows | never-seen follows | late: follows | shape | color | position |
|---|---|---|---|---|---|---|---|
| 0 | 59.4% | 0% | 0% | – | 30.7% | 20.9% | 11.0% |
| 1 | 81.1% | 73.6% | 67.2% | – | 92.5% | 99.2% | 89.8% |
| **1.5** | 87.4% | **84.6%** | **84.8%** | – | 97.2% | 100% | 96.1% |
| 2 | 79.9% | 76.4% | 77.3% | 74.8% | 93.3% | 100% | 94.5% |
| 3 | 19.3% | 9.8% | 11.7% | 72.0% | 81.1% | 99.6% | 33.5% |
| 5 | 0% | 0% | 0% | 39.8% | 75.6% | 93.3% | 10.6% |
| 8 | 0% | 0% | 0% | 10.6% | 67.3% | 59.4% | 7.9% |

(shape/color/position: the judge's reading agrees with the request, recognizable or not.) At w = 1.5, 31 of the 254
pictures go from wrong to right and 3 from right to wrong (`figures.py`).

**Finished model** (episode 10's, 20,000 steps), `results.json`:

| w | recognizable | follows | never-seen follows | late: follows | shape | color | position |
|---|---|---|---|---|---|---|---|
| 0 | 90.9% | 0% | 0% | – | 32.3% | 21.7% | 11.4% |
| **1** | 97.6% | **96.9%** | 96.5% | – | 99.2% | 100% | 99.6% |
| 1.5 | 96.9% | 96.5% | 95.7% | – | 99.2% | 100% | 100% |
| 2 | 83.9% | 58.7% | 55.9% | **95.7%** | 73.2% | 98.8% | 79.5% |
| 3 | 59.1% | 3.9% | 1.6% | **78.0%** | 22.4% | 91.3% | 18.9% |
| 5 | 26.8% | 0% | 0% | 63.0% | 19.7% | 89.4% | 2.8% |
| 8 | 9.8% | 0% | 0% | 43.7% | 37.0% | 88.6% | 2.0% |

**Every checkpoint** (`results_scan.json`, 127 trained combinations once each, follows):

| training steps | w = 1 | w = 2 | w = 2 late | w = 3 | w = 3 late |
|---|---|---|---|---|---|
| 2,000 | 0% | 0% | 0% | 0% | 0% |
| 4,000 | 20.5% | 34.6% | 30.7% | 12.6% | 35.4% |
| 6,000 | 70.1% | 77.2% | 71.7% | 12.6% | 70.1% |
| 8,000 | 87.4% | 80.3% | 89.0% | 7.1% | 86.6% |
| 20,000 | 96.1% | 61.4% | 92.9% | 2.4% | 84.3% |

Guidance helps while the model has not learned to listen well yet, and hurts once it has. At 2,000 steps nothing is
recognizable at any w, so there is nothing for guidance to sharpen.

**Why large w breaks the pictures** (`when_to_guide.py` → `results_when.json`; finished model, w = 3 in a range of
steps and w = 1 elsewhere, 127 trained combinations): never 96.1% follow, every step 2.4%, only t ≥ 500 2.4%,
only 100 ≤ t < 700 36.2%, only t < 500 84.3%. All the damage happens in the noisy half. There the guided guess
implies a clean picture far outside the pixel range: at t = 857, 10.1% of its pixels must be clipped (w = 1: 0.95%;
t = 715: 3.1% vs 0.06%). The sampler commits to a layout in those steps, and position is the first thing to go
(15.0% right). Guiding only later keeps the benefit where the layout is already fixed; Kynkäänniemi et al. (2024,
"Applying guidance in a limited interval") found the same for large image models.

**Diversity** (mean squared pixel difference between two pictures of the same request, 4 requests × 32): half-trained
0.0113 at w = 1 → 0.0099 at w = 1.5; finished 0.0104 → 0.0102. Mild guidance makes pictures of one request slightly
more alike. Beyond w = 2 the number rises again only because pictures break apart.

**Real models** use larger scales: Stable Diffusion's default guidance scale is 7.5 (same convention as our w).

Files: `out/fixes.png` (half-trained; row 1 w = 1, row 2 w = 1.5, same noise, requests listed in `figures.json`),
`out/ladder.png` (finished model; rows w = 1, 2, 3, and 3 late), `out/predictions.png` ("red circle, center" at
w = 3: the clean-picture guess without the request, with it, and guided, at five steps; this run ended in a
triangle), `out/{,mid6k_}w*_requests.png` and `w*_red.png` (only "red" said).
