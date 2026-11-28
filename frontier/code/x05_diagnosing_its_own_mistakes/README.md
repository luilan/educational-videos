# The Frontier, episode 5 · Diagnosing Its Own Mistakes

A small-scale version of self-rewriting ("Language Models Can Rewrite Their Own Weights", arXiv 2611.21885, 16 Nov
2026): a 27B model "given its weights, its graded answers and the ability to run itself, diagnoses its own failure and
rewrites the weights behind it", gaining 6.6 points on unseen problems; in one model family the smaller models made no
edit or broke themselves.

Same model, task, edit language and budget as episode 4 (Qwen3.5-0.8B, adding three 3-digit numbers, 48 practice
and 200 held-out problems, 25 rounds × 4 proposals). Each round adds:

1. **Diagnose**: the model reads up to 20 of its wrong practice answers and writes one sentence about its mistakes.
2. **Probe** ("the ability to run itself"): for each of its 49 RMSNorm vectors, the harness measures how scaling
   that vector by 1.1 changes the log-probability of the right answers to the wrong practice problems; the 10
   largest effects are shown to the model.
3. **Edit**: 4 proposals given the diagnosis and the probe table; the best is kept if the practice score rises.

Baseline "probe only": no language model; the 4 largest probe entries, each scaled by 1.1 or 0.9 in the helpful
direction. Probe and diagnosis are recomputed only after a kept edit (nothing else changes them).

```bash
pip install -r requirements.txt
python self_rewrite.py diagnose 10    # ~2.7 h → run_diagnose.json
python self_rewrite.py probe 6        # ~1.5 h → run_probe.json
python confirm.py                     # ~45 min: final models on 1,000 fresh problems → confirm.json
```

Results (2026-11-28):

| | Practice (48) | Held-out (200) | Fresh (1,000) | Newly right / newly wrong vs original |
|---|---|---|---|---|
| Original | 22.9% | 25.5% | 34.8% | |
| Diagnose + probe + self-edit | 35.4% | 32.0% | **42.4%** | 87 / 11 |
| Probe only (no language model) | 27.1% | 27.0% | 35.8% | 12 / 2 |
| Episode 4 blind self-edits | 35.4% | 26.0% | 34.1% | 30 / 37 |
| Episode 4 random edits | 31.2% | 27.5% | 36.1% | 35 / 22 |

(The 1,000 fresh problems happen to be easier than the 200; every model is scored on the same ones.)

- Kept edits, in order: layer 1 post-attention norm × 0.9; layer 2 post-attention × 0.5; layer 2 post-attention ×
  0.5 again; layer 1 post-attention × 0.3; layer 6 post-attention × 1.5; layer 3 input norm × 1.1.
- Understanding: 84 of 100 replies were valid edits; 41 targeted a vector in the probe's top 10 but only 3 the top
  entry; the direction of its scale factors matched the sign of the probe effect 31 times out of 63, a coin flip. The
  diagnosis was the same sentence in 19 of 25 rounds ("…adding the last digit of each number instead of the tens
  digit…").
- One run, one small model, one task. The edits generalize, but this does not show the model understood them;
  episode 7 ablates them one by one.
