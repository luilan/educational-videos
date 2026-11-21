# How Generative Models Work, episode 1 · What It Means to Generate

A generative model learns the distribution of its data and samples from it. Two files:

- `shapes.py`: **Shapes**, the dataset used throughout the series. Each picture is 32×32 RGB with one circle, square or
  triangle (5 colors, 9 positions, small random changes) and a caption like "a red circle at the top left". 135
  distinct captions. `python3 shapes.py` saves a sample grid.
- `generate_basics.py`: the experiments in the video. It writes `results.json` and the pictures in `out/`.

```bash
pip install -r requirements.txt
python3 generate_basics.py        # a few seconds on a CPU
```

Results (2026-11-21; seeds fixed, so the numbers repeat exactly):

1. **Counting in 2D.** 10,000 points on a spiral, 30×30 = 900 cells, 200 of them used. Sampling a cell in proportion
   to its count, then a point inside it, gives samples that follow the spiral.
2. **Counting in 3,072 dimensions.** With 30 bins per number the cells multiply by 30 per number (10<sup>15</sup> for
   10 numbers). Even with two levels per number a 32×32 color picture has 2<sup>3072</sup> ≈ 10<sup>924</sup> cells,
   against 20,000 training pictures.
3. **Pixels sampled one by one.** Learn the color distribution at each pixel from 20,000 pictures, sample each pixel
   independently. Shape pixels per picture: 68 (real: 65). Biggest connected piece: 4.7 pixels (real: 68). Colors
   per picture: 6 (real: 2). The busiest pixel is part of a shape 11% of the time.
4. **The average picture** is a dark smudge (`out/average.png`; `out/average_x4.png` has 4× contrast and shows the 9
   positions). **Copying** a training picture (`out/copies.png`) is a perfect sample but nothing new.
