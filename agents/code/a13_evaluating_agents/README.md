# How AI Agents Work, episode 13 · Evaluating Agents

A small evaluation suite: four tasks from earlier episodes, several attempts each, graded automatically.

- `eval.py`: runs every task `k` times with episode 2's harness (one `shell` tool, 30-second timeout per command, at
  most 10 model calls, stuck detector) and grades the outcome:
  - `stats` (episode 2): fix `mean()`; `test_stats.py` must pass **and be unchanged**;
  - `fix` (episode 5): fix `total()` in `invoice.py`; `test_invoice.py` must pass and be unchanged;
  - `search` (episode 4): the answer must contain `study_guide.py` and `42`;
  - `server` (episode 6): the answer must contain `1.4.2`.
  It prints each attempt, then per task: pass@1 (share of attempts that passed), pass^k (all `k` passed), mean tokens
  and seconds.
- `tasks/`: a copy of each task's project. `llm.py`: Ollama Responses API, per-reply cap (`MAX_OUTPUT_TOKENS`, 8,000).

```bash
ollama pull qwen3:4b
python3 eval.py 2          # k = 2 attempts per task; writes results_<date>.json
```

WARNING: the harness runs the model's commands. Use a container or a throwaway VM. Python standard library only.

Results (2026-11-21, Qwen3 4B on Ollama, CPU, in a container without internet; `run_2026-11-21.txt`,
`results_2026-11-21.json`):

| task | attempt 1 | attempt 2 | pass@1 | pass^2 | mean tokens | mean seconds |
|---|---|---|---|---|---|---|
| stats | PASS | FAIL | 0.50 | no | 1,064 | 781 |
| fix | FAIL | PASS | 0.50 | no | 9,404 | 2,318 |
| search | FAIL | FAIL | 0.00 | no | 3,831 | 1,312 |
| server | PASS | PASS | 1.00 | yes | 633 | 592 |

- Overall pass@1: 0.50 (4 of 8). Only `server` passed every time.
- The same task varies: the invoice fix failed after 4 calls and 7,754 tokens, then passed after 6 calls and 11,053.
- The 8 attempts took 10,003 seconds (2 h 47 min) and 29,864 tokens on this CPU. Two attempts per task are far too
  few to separate a real improvement from luck; the point is the method: change one thing, rerun the same suite,
  compare.
- No attempt edited a test file in this run, so the "unchanged" check never fired; it is there because an agent can
  make a failing test pass by changing the test.

How Codex fits in: `codex exec --json` runs Codex non-interactively and prints its events as JSON lines, which is how
this series captured Codex's runs; a suite like this one can drive it the same way.
