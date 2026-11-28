# The Frontier, episode 3 · The Random Baseline

Episode 2's repair loop with the editor replaced by chance. Same three damages to episode 1's model (99.2% on new
questions), same rule: an edit is kept only if the score on 300 practice questions goes up, and the search stops at
99%. The 1,000 test questions only judge, at fixed checkpoints.

- **random edits**: Claude's edit language (set one number, scale a row, swap two rows, replace a row) with every
  choice random: tensor (uniform over the 30 weight tensors), row or entry, value (from that tensor's own mean and
  spread), factor (uniform in −2…2). 3 seeds, up to 10,000 tries.
- **random nudges**: Gaussian noise on every weight (1% of each tensor's spread), keep if better. 1 seed.

```bash
pip install -r requirements.txt
(cd ../x01_the_frozen_model && python frozen.py)    # model.pt, if you don't have it yet
python random_search.py                             # ~13 min on 16 threads (6 processes) → results.json
```

`run_log.txt` is the log of the run used in the video.

Results (2026-11-28). Test accuracy; Claude's numbers are from episode 2.

| Damage (test damaged) | Claude (ep. 2) | random edits, after 4 tries | random edits, best of 10,000 tries | random nudges, 10,000 |
|---|---|---|---|---|
| one LayerNorm gain zeroed (77.7%) | 1 try, 1 number → 99.1% | 77.7 / 77.5 / 77.7% | 98.6% at try 774 (331 numbers changed), 96.8% at try 1,567 (1,803), 86.8% after 10,000 | 92.5% (all 102,413 numbers changed) |
| embeddings of 3 and 8 swapped (39.3%) | 1 try, exact → 99.2% | 39.3% (all 3) | 60.2 / 67.4 / 92.7%, never the swap | 39.3%, no edit ever kept |
| embedding of 7 × −1 (58.7%) | 2 of 3 runs, 1 try, exact → 99.2% | 58.7% (all 3) | 68.6 / 65.9 / 68.2%, never the sign flip | 58.7%, no edit ever kept |

- Practice vs test: the zero-case random runs reached 99.0% on practice but 98.6% and 96.8% on the test; nudges 98.7%
  practice, 92.5% test. Many small edits fit the 300 practice questions; Claude's single edit scored 99.7% / 99.1%.
- The exact swap needs tensor `tok.weight` (1/30), the swap op (1/4) and the rows 3 and 8 (2/13 × 1/12): about 1 chance
  in 9,400 per try. It was never kept in 30,000 tries.
- Cost per try: ~45 ms of CPU for random edits; a Claude round took 8–34 s and about $0.27.

Related: Neural Thickets (arXiv 2603.12228, Mar 2026) on random weight perturbations plus selection near pretrained
weights.
