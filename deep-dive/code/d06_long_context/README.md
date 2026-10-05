# Deep Dive, episode 6 · Long Context: Stretching RoPE

A tiny RoPE GPT (810,049 parameters, no position table) trained on Tiny Shakespeare with a 64-character context, then
asked to read 256 characters four ways.

```bash
pip install -r requirements.txt
python long_context.py
```

It loads the included `rope_gpt.pt` (or trains it in a few minutes on a CPU if missing), then prints the validation loss
on positions 0–63, 64–127 and 128–255:

| setting | 0–63 | 64–127 | 128–255 |
|---|---|---|---|
| no change (extrapolation) | 1.60 | 2.15 | 3.23 |
| position interpolation (× ¼), no training | 3.43 | 3.58 | 3.59 |
| NTK-aware base scaling (base ≈ 43,873) | 1.63 | 1.62 | 2.51 |
| interpolation + 200 training steps at 256 | 1.62 | 1.56 | 1.58 |

Each method is one argument of `Rope`: `scale` multiplies the positions, `base` sets the rotation speeds. The
per-position curves shown in the video are saved to `curves.pt`. The 200-step fine-tune is random, so its numbers can
vary slightly between runs.
