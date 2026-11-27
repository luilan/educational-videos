# The Frontier, episode 2 · Weight-Space Programming

Claude repairs episode 1's tiny addition model (99.2% on new questions) after we damage it in secret. The editor
(Claude Opus through the Claude Code CLI: `claude -p --tools "" --json-schema …`, no tools) reads the architecture, the
damaged model's answers to 300 practice questions and the weights as text, and replies with edits (set a number, scale
a row, swap rows, replace a row). `wsp.py` applies them, re-scores the practice questions and keeps an edit only if the
score rises; up to 4 rounds. No gradients. The 1,000 test questions are never shown to the editor.

```bash
pip install -r requirements.txt          # plus the Claude Code CLI (set CLAUDE_BIN if it is not on the default path)
(cd ../x01_the_frozen_model && python frozen.py)   # model.pt (or train it with the first half of frozen.py)
python wsp.py                            # zero, flip, swap; python wsp.py "flip#2" for a repeat run
python export_view.py                    # view.json: the rows shown in the video
```

Results (2026-11-27; every prompt, reply, edit and score is in `runs/`):

| Damage | Test damaged | Runs fixed | Edit | Test after |
|---|---|---|---|---|
| `blocks.0.ln1.weight[13]` = 0 | 77.7% | 1 of 1 (round 1, 8 s) | set it to 0.79 (was 0.81): 1 number | 99.1% |
| embeddings of 3 and 8 swapped | 39.3% | 1 of 1 (round 1) | swap the rows back: 128 numbers, exact | 99.2% |
| embedding of 7 × −1 | 58.7% | 2 of 3 | runs 2 and 3: row 7 × −1 in round 1, exact | 99.2% (run 1: 61.2%) |

In run 1 of the sign flip the diagnosis was right ("an input digit 7 were read as 2") but the edits were not: scale
×0.6, zero one entry (+0.3 points), set it to +2.516, then halve an unrelated MLP neuron.

Papers: the November 2026 weight-space programming preprints, e.g. arXiv 2611.33031, 2611.07319, 2611.05341,
2611.08070, 2611.31715, 2611.04268.
