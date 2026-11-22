# How Generative Models Work, episode 9 · Flow Matching

Flow matching (Lipman et al. 2022; “rectified flow”, Liu et al. 2022) against diffusion, with the same networks, the
same data and the same training budget. Mix picture and noise on a straight line,
x_t = (1 − t)·x0 + t·ε for t in [0, 1], and train the network to predict the line's direction v = ε − x0. To
generate, start from noise at t = 1 and take Euler steps x ← x − Δt·v̂ down to t = 0.

```bash
pip install -r requirements.txt
python3 train_flow.py toy          # 2D spiral: diffusion, flow matching and reflow, tiny MLPs, ~10 minutes → results_toy.json
python3 train_flow.py shapes       # episode 6's U-Net trained with flow matching, 20,000 steps, ~2.5 hours with 8 threads
                                   # → results_shapes.json, out/flow_*.png, out/flow_ema.pt (also scores DDIM on episode 6's model)
python3 diffusion_fresh_noise.py   # episode 6's model with fresh noise (η = 1) at 5 and 10 steps → results_fresh.json
```

`diffusion.py`, `judge.py` and `shapes.py` are copies from episode 6. The network's step input is t·1,000, so flow
matching uses exactly episode 6's U-Net (1,182,371 parameters) and training settings (batch 128, Adam 0.0002,
EMA 0.999).

Results (2026-11-22):

**Shapes** (500 samples from the same starting noise as episodes 7 and 8, scored by episode 3's judge):

| network calls | flow matching (Euler) | diffusion, fresh noise (η = 1) | diffusion, DDIM (η = 0) |
|---|---|---|---|
| 5 | **76.2%** (114 combinations) | 6.8% (27) | 1.6% (8) |
| 10 | **90.6%** (128) | 85.0% (127) | 16.4% (56) |
| 20 | 92.2% (133) | 95.4% (132) | 45.8% (100) |
| 50 | 94.0% (131) | 95.2% (130) | 70.0% (127) |

(Diffusion at 20 and 50 steps with fresh noise: episode 7's run from the same noise.) Flow matching is far better at
5–10 steps; from 20 steps on, diffusion with fresh noise catches up. During training (Euler, 50 steps) the flow
model reached 67.6% at 5,000 steps and 90.8% at 10,000. The two training losses measure different targets and
cannot be compared.

**2D spiral** (MLP with three hidden layers of 256, 10,000 steps each; 2,000 generated points. Precision: median
distance from a generated point to the nearest data point. Coverage: median distance from a data point to the nearest
generated point. Lower is better for both):

| steps | diffusion (DDIM) | flow matching | reflow |
|---|---|---|---|
| 1 | 1.831 / 1.363 | 0.006 / 1.093 (all points at the middle) | **0.007 / 0.019** |
| 2 | 1.785 / 1.336 | 0.209 / 0.191 | 0.007 / 0.017 |
| 4 | 0.017 / 0.064 | 0.011 / 0.029 | 0.007 / 0.017 |
| 16 | 0.007 / 0.020 | 0.007 / 0.016 | 0.007 / 0.016 |
| 200 | 0.008 / 0.017 | 0.007 / 0.016 | 0.007 / 0.016 |

- **The learned paths are not straight.** Training lines from many noises to many points cross; where they cross the
  network learns their average direction, so the paths bend. One Euler step follows that average from the start and
  lands every point at the middle of the spiral (precision looks perfect there because the middle lies on the
  spiral; coverage shows the collapse). Median straightness (straight-line distance / path length, 50 steps):
  diffusion's DDIM paths 0.68, flow 0.55.
- **Reflow** (Liu et al. 2022): run the trained flow from 10,000 noises (100 steps), pair each noise with where it
  lands, and train a new flow on straight lines between those pairs only. These lines don't cross, the paths come
  out straight (median straightness 1.00), and one step is enough.
