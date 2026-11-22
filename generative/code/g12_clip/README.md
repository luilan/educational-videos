# How Generative Models Work, episode 12 · CLIP: Matching Pictures and Words

A tiny CLIP (Radford et al. 2021) trained on Shapes pictures and their captions ("a red circle at the top left").

```bash
pip install -r requirements.txt
python3 train_clip.py      # ~5 minutes with 4 threads → out/clip.pt, results.json, out/*.png
```

- **Image encoder:** a small CNN (588,992 parameters) → 64 numbers. **Text encoder:** word embeddings + a 2-layer
  transformer (4 heads), averaged over the words → 64 numbers (72,768 parameters). Vocabulary: 17 words.
- Both vectors are normalized to length 1, so their dot product is a cosine similarity. For a batch of 128
  picture-caption pairs, the 128 × 128 table of similarities (times a learned scale, 1/temperature) goes through
  cross-entropy along each row (picture → caption) and each column (caption → picture). Adam 0.001, 4,000 steps.
- Episode 10's eight held-out combinations (e.g. "a purple triangle at the top left") never appear in training,
  neither as pictures nor as captions.
- `out/clip.pt` is used by episode 13 as its text encoder.

Results (2026-11-22). "Reading" a picture = choosing the most similar of all 135 possible captions (zero-shot; the
model never trains on this task). 1,877 test pictures of trained combinations, and 200 of the held-out ones:

| training step | 0 | 10 | 20 | 30 | 50 | 75 | 100 | 150 | 250 | 4,000 |
|---|---|---|---|---|---|---|---|---|---|---|
| exact caption | 0.2% | 11.6% | 37.3% | 41.6% | 58.4% | 78.1% | 96.4% | 99.9% | 100% | 100% |
| … position right | 7% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| … color right | 19% | 31% | 93% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| … shape right | 30% | 35% | 41% | 42% | 58% | 78% | 96% | 100% | 100% | 100% |
| held-out combinations | 0.0% | 2.5% | 37.5% | 24.0% | 28.5% | 68.0% | 97.5% | 100% | 100% | 100% |
| caption → picture, top 5 | 0.8% | 25.2% | 44.3% | 50.4% | 73.5% | 94.5% | 100% | 100% | 100% | 100% |

- **Order of learning:** position first, then color, then shape (compare episode 7: a sample's position is decided
  before its color and shape).
- **Combinations never seen** are read perfectly from step 150 on: the model learns each word's meaning separately
  and combines them.
- **The loss never reaches zero.** It levels off at 0.574. A batch of 128 drawn from 127 captions often contains the
  same caption twice, and nothing can tell two identical captions apart, so the best possible probability for such a
  pair is 1/k, with k copies. The expected value of log k for this batch is 0.574, exactly where the loss stops.
- `out/table_pictures.png` + `results.json["table"]`: the similarity table for 8 test pictures (two of them carry
  the same caption: their rows and columns are identical). `out/search_*.png`: the 8 most similar pictures for
  "a green square at the bottom left" and for the held-out "a purple triangle at the top left".
- Nearest captions to "a red circle at the top left" (cosine similarity): "… at the left" 0.72, "… at the top" 0.63,
  "… at the bottom left" 0.62, "… in the center" 0.61, "a purple circle at the top left" 0.60.
