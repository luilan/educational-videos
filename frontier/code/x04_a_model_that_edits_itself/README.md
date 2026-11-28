# The Frontier, episode 4 · A Model That Edits Itself

A small-scale version of Self-WSP (arXiv 2611.28833, 21 Nov 2026), where Qwen3.5-0.8B reads its own weights, proposes
an edit, and continues with the edited weights as both problem solver and editor.

- Model: Qwen3.5-0.8B (752M parameters), on a CPU, no thinking mode.
- Task: add three 3-digit numbers ("What is 412 + 958 + 136? Answer with just the number."). 48 practice problems
  (used to accept edits), 200 held-out test problems (never used to choose anything).
- Each round the model is shown its own RMSNorm vectors w (49 vectors × 1,024 numbers; each layer multiplies channel
  i by 1 + w[i]) as statistics plus the 4 largest entries, and its wrong practice answers, and writes 4 edits as JSON
  (sampled at temperature 0.8): scale a whole vector by F, or set one entry. Each edit is scored on the practice
  problems; the best is kept if it beats the current score. 25 rounds = 100 proposals. The edited model is the
  editor in the next round.
- Baseline (episode 3): the same loop with 100 random edits in the same edit language.

```bash
pip install -r requirements.txt
python self_edit.py self 14      # ~1.5 h on 14 CPU threads → run_self.json
python self_edit.py random 8     # ~1.3 h → run_random.json
```

Logs: `log_self.txt`, `log_random.txt`; every proposal (raw text, parsed edit, practice score) is in the JSON files.

Results (2026-11-28):

| | Practice (48) | Held-out (200) | Edits kept | Proposals that beat the current model |
|---|---|---|---|---|
| Start | 22.9% | 25.5% | | |
| Self-edits | 35.4% | 26.0% | 3 | 3 of 100 (95 valid) |
| Random edits | 31.2% | 27.5% | 4 | 5 of 100 |

- Kept self-edits: layer 0 post-attention norm × 1.1; layer 6 post-attention norm × 1.5; final norm entry 114 set to
  −31.6 (it was 3.48: the channel's multiplier went from 4.48 to −30.6). Held-out after each: 25.0%, 26.0%, 26.0%.
- The model's proposals: 76 scale a whole vector (88% of factors above 1), 19 set one entry; 28 touch the final
  norm. 5 were invalid (e.g. "No update needed…").
- One practice question is 2.1 points, and the held-out score's standard error is about ±3 points, so neither run's
  held-out change is distinguishable from noise.
- Our mistake: the first self run described w as a "gain" (Qwen3.5 multiplies by 1 + w). We stopped it after 17
  rounds and re-ran with the correct description; the pilot showed the same picture (practice 31.2%, held-out 26.0%,
  `pilot_run_self_misdescribed.json`). The random baseline is unaffected (it reads no description).
- Self-WSP itself reports GSM8K 11.8% → 62.2% and MATH500 7.0% → 32.2%, "sometimes by changing only 16 parameters",
  and also that "WSP driven by random search sometimes performs comparably" and that the model "shows little
  understanding of the edits it makes". We did not reproduce large gains at this scale (one kind of edit, 100
  proposals, 48 practice problems).
