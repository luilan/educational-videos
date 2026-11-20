# How AI Agents Work, episode 5 · Editing Files

The same one-line bug fix, done by rewriting the whole file or by sending a patch.

- `demo/invoice.py` (108 lines, 3,755 characters) has one bug: `total()` adds the discount back. `demo/test_invoice.py`
  fails until it is fixed.
- `edit_harness.py`: the tiny harness (one `shell` tool) in three modes, then a check: does the test pass, and how many
  lines of `invoice.py` changed?
  - `rewrite`: the instructions say to edit by writing the complete new file with `cat > FILE <<'EOF' … EOF`.
  - `patch`: the instructions describe Codex's patch format; the harness **intercepts** any command that starts with
    `apply_patch` and applies it itself (`apply_patch()`), the way Codex does for models without a dedicated edit tool.
  - `read`: `patch`, plus: read the lines you will change with `grep -n` or `sed -n` before editing.
- `apply_patch()` / `seek()`: a small version of `codex-rs/apply-patch`. It finds the patch's old lines in the file
  (exact, then ignoring trailing spaces, then ignoring all edge spaces; Codex also normalises Unicode punctuation),
  refuses paths outside the project, and returns `apply_patch failed: could not find these lines …` instead of
  touching the file when nothing matches.
- `llm.py`: one call to Ollama's Responses API, with a per-reply output cap (`MAX_OUTPUT_TOKENS`, default 8,000) so
  runaway reasoning cannot stall a run.

```bash
ollama pull qwen3:4b
python3 edit_harness.py rewrite
python3 edit_harness.py patch
python3 edit_harness.py read
```

WARNING: the harness runs the model's commands. Use a container or a throwaway VM. Python standard library only.

Results (2026-11-20, Qwen3 4B on Ollama, CPU, in containers without internet; full output in `run_2026-11-20.txt`;
runs vary). No run produced a working fix; what differs is what each mode did to the file.

| mode | run | what happened | lines changed | test |
|---|---|---|---|---|
| rewrite | 1 | never read the file; overwrote it 11 times with 277–308-character inventions | 104 removed, 8 added | fail |
| rewrite | 2 | read it and overwrote the same instant (same reply); then rewrote all 4,067 characters four times; stopped by hand at step 8 | (not measured) | fail |
| rewrite | 3 | read it, rewrote 4,334 then 4,052 characters ×3; stopped by the stuck detector; dropped `@dataclass`, turned `class Item:` into `dataclass Item:` | 14 removed, 7 added | fail (file no longer runs) |
| patch | 1 | no edit; reasoned about a patch for `return sum(self.items) + 10.0`, a line that does not exist | 0 | fail |
| patch | 2 | read the file, named the exact bug and started the right patch in its reasoning; reply hit the 6,000-token cap | 0 | fail |
| patch | 3 | no edit; wrote `apply_patch <<'EOF' …` as plain text (not a tool call), for the same invented line | 0 | fail |
| read | 1 | ran the test, then replied with nothing | 0 | fail |
| read | 2 | no edit; wrote the invented-line patch as plain text | 0 | fail |

- Rewrites destroyed or corrupted the file in every run where they ran; patch runs never changed it.
- The invented patch is refused by `apply_patch()`: `could not find these lines in the file`.
- The fix patch 2 worked out, sent as a patch, is 277 characters (the whole file is 3,755) and makes the test pass.
- With the first task wording (*"Find the bug in invoice.py and fix it."*) one rewrite and one patch run made no edit
  at all; one wrote `cat > invoice.py <<'EOF' …` as text, inventing a 6-line file. The wording was then made explicit.
- Guards were added during the session: the stuck detector (same tool calls on three steps in a row) from rewrite 3
  on, the output cap from patch 2 on (6,000 tokens; 8,000 from read-first 2 on).

How Codex CLI does it (openai/codex @ 89461ea):

- The format: `*** Begin Patch`, `*** Update File: path`, `@@` context lines, `-`/`+`/space lines, `*** End Patch`
  (grammar in `codex-rs/core/assets/tools/apply_patch.lark`; parser in `codex-rs/apply-patch/src/parser.rs`).
- GPT models get a freeform `apply_patch` tool. Other models (the fallback metadata used for Qwen3 4B sets
  `apply_patch_tool_type: None`) run `apply_patch` as a shell command, which Codex intercepts before it reaches the
  shell (`intercept_apply_patch` in `core/src/tools/handlers/apply_patch.rs`), verifies and applies itself.
- Matching (`apply-patch/src/seek_sequence.rs`): exact, then trailing whitespace ignored, then all edge whitespace,
  then Unicode punctuation normalised to ASCII.
- Instructions (`protocol/src/prompts/base_instructions/default.md`): use `apply_patch` to edit files, and don't
  re-read a file after patching it, since the call fails if the patch did not apply.
