# How Generative Models Work, episode 3 · Variational Autoencoders

A VAE encodes each picture as a cloud (a mean and a spread per code number) and pulls every cloud toward a standard
normal with a KL penalty, so random codes from N(0, 1) decode to pictures.

- `train_vae.py`: trains episode 2's network with an 8-number code as a plain autoencoder and as a VAE (40 epochs
  each), plus a VAE with a 2-number code. Writes `results.json` and `out/*.png`.
- `judge.py`: the **judge** used from here on to score generated Shapes pictures. A small classifier with four heads:
  shape, color, position, and clean-vs-junk (trained against averaged pairs, two shapes, off-palette colors, heavy
  blur and noise). A picture is **recognizable** when the judge is ≥ 90% sure of shape, color and position and finds
  it clean rather than junk. `python3 judge.py` trains it (about 3 minutes, saved to `out/judge.pt`) and checks it.
- `shapes.py`: the dataset (episode 1).

```bash
pip install -r requirements.txt
python3 judge.py          # optional: train and check the judge (train_vae.py trains it if missing)
python3 train_vae.py      # about 10 minutes on an 8-core CPU
```

Results (2026-11-21; seeds fixed):

Judge check on 2,000 new pictures: real 99.8% recognizable (100% read correctly); blurred by σ = 0.5 pixel 95.1%,
0.7: 60.1%, 1.0: 8.2%; junk 0–1%.

| 8-number code, 40 epochs | test MSE | random code → nearest training picture | random samples recognized | readable | clean |
|---|---|---|---|---|---|
| autoencoder (codes drawn with measured mean/spread) | 0.00055 | 0.0076 | 9.8% | 41.4% | 15.4% |
| VAE (codes drawn from N(0, 1)) | 0.00087 | 0.0041 | 9.4% | 41.6% | 14.0% |

- VAE KL penalty at the end of training: about 13 nats per picture; every code number's spread across the test set is
  close to 1.
- 2-number VAE: test codes span −3.4 … +3.6 (77% within distance 2 of the origin), against −13 … +10 with gaps for
  episode 2's autoencoder. `out/map_vae2.png` decodes a 10×10 grid from −2 to +2: a shape in every cell. Random
  samples recognized: 13.6%.
- `out/walk_vae.png`: 7 steps from one test picture's code to another's.
- So: the VAE closes the gaps (samples land about twice as close to real pictures), but its samples are blurry, and a
  strict judge recognizes about 1 in 10, no better than the autoencoder's.
