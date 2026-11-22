# How Generative Models Work, episode 10 · Conditioning: Telling the Model What to Draw

Episode 6's diffusion model, told what to draw: a shape, a color and a position.

```bash
pip install -r requirements.txt
python3 train_cond.py       # ~3.5 hours on a CPU (20,000 steps) → out/cond_ema.pt, results_train.json
python3 evaluate_cond.py    # ~15 minutes → results_cond.json, out/*.png
```

- **The network** is episode 6's U-Net (channels 32, 64, 128) with three extra embedding tables, one each for
  shape (3), color (5) and position (9), plus a "not said" entry. The three label vectors are added to the time
  embedding, so every block sees them, exactly the way it sees the time (episode 8). 1,186,211 parameters
  (the unconditional U-Net has 1,182,371).
- **Label dropout.** 10% of training pictures get no label at all; otherwise each of the three labels is dropped
  on its own 10% of the time. So the same network can also draw from part of a description ("something red") or
  from none (an unconditional model, which episode 11's guidance needs).
- **Eight combinations are left out of training** (the same eight as episodes 12 and 13), for example
  "purple triangle, top left": 18,812 of 20,000 training pictures remain.
- Batch 128, Adam 2e-4, EMA 0.999, cosine schedule with T = 1000, ε-prediction (episode 6). Final loss 0.0026.

Results (2026-11-22). Sampling: 50 steps with fresh noise (η = 1, episode 7), no guidance. The episode 3 judge
reads each picture; "follows" = recognizable and every requested attribute is right.

| request | pictures | recognizable | follows | shape right | color right | position right |
|---|---|---|---|---|---|---|
| all 127 trained combinations ×4 | 508 | 98.2% | 96.7% | 97.2% | 98.2% | 97.6% |
| the 8 combinations never seen ×50 | 400 | 98.8% | **97.0%** | 98.0% | 98.5% | 98.0% |

Never-seen combinations, one by one: purple triangle top left 92%, yellow circle bottom right 98%, red square
center 98%, green triangle bottom left 98%, blue circle top right 100%, purple square bottom 98%, yellow triangle
right 96%, red circle left 96%.

**Part of a description** (200 pictures each): what was said is followed, what wasn't is spread out.

| said | follows | not said, among recognizable pictures |
|---|---|---|
| "red" | 86.0% | circle 51, square 50, triangle 71; all 9 positions 15–25 |
| "triangle" | 86.5% | colors 29–43 each; positions 15–28 each |
| "top right" | 87.5% | circle 59, square 64, triangle 60; colors 28–45 each |

**No description at all:** 500 pictures, 90.0% recognizable, covering 131 of the 135 combinations. The same
network works as an unconditional model (episode 8's dedicated one: 95.2% with the same sampler).

Files: `out/requests.png` (rows: red triangle center, yellow circle top left, blue square bottom right, purple
circle right), `out/held_out.png` (first four never-seen combinations), `out/partial_*.png`, `out/unconditional.png`.
`out/cond_ema.pt` is reused by episode 11.
