# Foundations (pre-series for How LLMs Work)

Short videos (about 2 minutes each, 27 minutes in total) on the math and code the main series assumes.
Each one ends with a *"Where you'll see this"* card pointing to the main episodes that use it, and F14 hands off
to episode 1 of [How LLMs Work](../how-llms-work/). F12–F14 are extras.

| # | Title | Length | Used in main episodes |
|---|---|---|---|
| F01 | **Vectors: Lists of Numbers as Arrows**<br>The basic building block of every LLM | 1:56 | 1, 3, 4, 9 |
| F02 | **The Dot Product**<br>One number that says how much two vectors agree | 1:52 | 3, 5, 6, 8, 10 |
| F03 | **Matrices: Many Dot Products at Once**<br>The table of numbers behind every layer | 2:02 | 5, 6, 7, 8, 10 |
| F04 | **Straight Lines and Bends**<br>Linear functions, and why networks need a bend | 1:49 | 8, 9 |
| F05 | **Exponentials and Logarithms**<br>Fast growth, and the function that undoes it | 1:59 | 6, 10, 11, 12 |
| F06 | **Probability and Sampling**<br>Distributions, weighted dice, and randomness | 1:45 | 1, 10, 14 |
| F07 | **Softmax, Properly**<br>From any scores to probabilities | 1:52 | 5, 6, 10 |
| F08 | **Averages and Spread**<br>Mean, standard deviation, and keeping numbers tame | 1:56 | 6, 9 |
| F09 | **Waves and Rotations**<br>Sine, cosine, and turning arrows | 1:48 | 4 |
| F10 | **Slopes and Gradients**<br>How a model knows which way to adjust | 2:00 | 11, 12 |
| F11 | **NumPy in Three Minutes**<br>Just enough to read the code in the series | 2:03 | code in 3–13 |
| F12 | **Neural Networks in Three Minutes** *(extra)*<br>Neurons, layers, weights, and learning | 2:00 | 8, 11, 12 |
| F13 | **PyTorch and Autograd** *(extra)*<br>How one line computes every gradient | 1:51 | 11, 12 |
| F14 | **Train vs Validation Data** *(extra)*<br>How we know a model really learned | 1:53 | 11, 12 |

## Files

- `fNN_script.py`: title, tagline, series label, "used in" episodes and the narration. Readable version: [NARRATION.md](NARRATION.md).
- `fNN_scene.py`: the Manim scene.
- `voice/fNN/sections.json`: section durations and word timestamps.
- `study/fNN_study.pdf`: the study guide for each video (source: `study/fNN_study.py`).

F14 reads the tiny GPT's real training data and log from `../how-llms-work/tiny_gpt/`.

Render a video from the repo root: `./render.sh foundations f07` (see the [main README](../README.md)).
