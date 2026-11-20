# How AI Agents Work, episode 4 · Reading and Searching Code

How an agent finds the right lines in a project too big to read: the same question, asked of the same model, in three
harness configurations.

- `search_harness.py`: a tiny harness (one `shell` tool) that asks *"which line decides the file name of a figure image
  extracted for the study-guide PDF?"* about `demo_project/` and checks the answer (`framework/study_guide.py`, line 42).
  - `naive`: no guidance, full command output.
  - `search`: two sentences of instructions (search with `grep -rn`, read ranges with `sed -n`), and every tool result
    capped at 2,000 characters with Codex-style middle truncation (`truncate_middle`).
  - `context`: `search`, plus: use `grep -rn -C 3` to see the lines around each match, and when you find where a value
    is used, search for where it is set.
- `count_tokens.py`: size of `demo_project/` in Qwen3 tokens (needs `transformers`; downloads the tokenizer only).
- `demo_project/`: 12 Python files from this repository's own tooling (1,455 lines, 19,072 tokens).
- `llm.py`: one call to Ollama's OpenAI-style Responses API.

```bash
ollama pull qwen3:4b
python3 search_harness.py naive
python3 search_harness.py search
python3 search_harness.py context
pip install transformers && python3 count_tokens.py
```

WARNING: the harness runs the model's commands. Use a container or a throwaway VM. `OLLAMA_URL` and `MODEL` select the
server and model. Python standard library only (except `count_tokens.py`).

Results (2026-11-20, Qwen3 4B on Ollama, CPU, in containers without internet; full output in `run_2026-11-20.txt`;
runs vary):

| mode | run | commands | tokens sent | answer |
|---|---|---|---|---|
| naive | 1 | `ls`, `ls stepstone`, `cat publish.py`, `cat build_packs.py` | 6,978 | right format string, wrong file (`stepstone/build_packs.py`), invented line 108 (the real line there is 209) |
| naive | 2 | `ls -la` | 553 | invented file `stepstone/study_guide.py`, line 25 |
| naive | 3 | same as run 1 | 6,978 | same file, invented line 177 |
| search | 1 | `grep -rn "figure" .`, `sed -n '224p' …` | 1,799 | `framework/study_guide.py:224` (where the name is used) |
| search | 2 | `grep -rn 'study-guide' .`, `grep -rn 'figure' .`, `sed -n '224p' …` | 2,314 | same |
| search | 3 | `grep -rn "study-guide" *`, `grep -rn "figure" framework/study_guide.py`, `sed -n '224p' …` | 1,910 | same |
| context | 1 | `grep -rn 'study-guide'`, `grep -rn -C3 'figure'` (5,343 chars, 2,024 shown), `grep -rn -C3 'fig\["file"\]'` | 3,046 | **right**: `framework/study_guide.py`, line 42 |
| context | 2 | `grep -rn -C 3 'study-guide' *`, `grep -rn -C 3 'figure' *` (5,343 chars, 2,024 shown) | 1,932 | **right** (lines 38–44 were in the kept head) |

- `cat` prints no line numbers: the naive runs that read files guessed them. `grep -n` prints real ones.
- Every search run found the right file and quoted real line numbers, for about a third of the tokens; all three stopped
  at line 224, where the name is *used*. The line that *sets* it (42) sits right below line 41, which their own
  `grep "figure"` had matched.
- Context mode found it both times: once by searching again for `fig["file"]`, once from the context lines alone.

Codex CLI 0.163.0 with the same model and task (`codex_trace_2026-11-20.jsonl`): `ls /work`, `ls /work/stepstone`,
`cat /work/stepstone/build_packs.py`; requests of 7,852, 9,999 and 10,572 tokens; no answer within our 50-minute limit.

How Codex CLI does it (openai/codex @ 89461ea):

- Instructions (`codex-rs/protocol/src/prompts/base_instructions/default.md`): *"When searching for text or files,
  prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. (If the
  `rg` command is not found, then use alternatives.)"*
- Output budget: `exec_command` takes `max_output_tokens`, default 10,000 (`core/src/unified_exec/mod.rs`); longer
  output keeps its beginning and end with `…N tokens truncated…` in between (`utils/string/src/truncate.rs`).
