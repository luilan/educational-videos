# How Generative Models Work, episode 2 · Autoencoders

An autoencoder squeezes a picture into a short code and rebuilds it. `train_autoencoder.py` trains the same small
convolutional autoencoder with codes of 2, 4, 8, 16 and 32 numbers on 20,000 Shapes pictures (`shapes.py`, from
episode 1) and tests it on 2,000 new ones.

```bash
pip install -r requirements.txt     # CPU PyTorch is enough
python3 train_autoencoder.py        # about 5 minutes on an 8-core CPU; writes results.json and out/*.png
```

The network: encoder 3×32×32 → three stride-2 convolutions (32, 64, 128 channels, batch norm, leaky ReLU) → 128×4×4
→ linear → code; the decoder mirrors it with transposed convolutions and a sigmoid. 366,539 parameters with an 8-number
code. Adam, learning rate 0.001, batch 128, 15 epochs. (Without batch normalization, training got stuck predicting an
empty background for most code sizes.)

Results (2026-11-21; seeds fixed):

| code size | test MSE | pixels clearly wrong (> 0.25) | random code → nearest training picture | halfway → nearest |
|---|---|---|---|---|
| 2 | 0.0044 | 3.9% | 0.0029 | 0.0035 |
| 4 | 0.0022 | 2.0% | 0.0046 | 0.0035 |
| 8 | 0.0011 | 0.9% | 0.0086 | 0.0056 |
| 16 | 0.0010 | 0.7% | 0.0093 | 0.0057 |
| 32 | 0.0007 | 0.5% | 0.0085 | 0.0066 |

- `out/recon_N.png`: rebuilt test pictures (`out/originals.png` are the inputs).
- `out/random_N.png`: decoded random codes (each number drawn with the mean and spread of the real codes). Plausible
  blobs with 2 numbers, mush with 8 and more.
- `out/halfway_N.png`: rows = picture A, decoded (code A + code B) / 2, picture B.
- `results.json` → `codes_2d`: the 2-number codes of all 2,000 test pictures with their (shape, color, position)
  labels; plotted, they form islands by position with gaps between them.
