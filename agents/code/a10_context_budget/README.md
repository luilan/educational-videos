# How AI Agents Work, episode 10 · The Context Budget

A reading task that outgrows a small context window, three ways.

- `budget_harness.py`: three questions about three files of `demo_project/` (the project of episode 4); the harness
  checks the answers (`AgentsCover.jpg`, `10`, `12.8`).
  - `unlimited`: every tool result stays in the conversation.
  - `compact`: a simulated 4,096-token window; when a request plus the new tool output passes 90% of it, the harness
    asks the model for a handoff summary with Codex's own prompt (`compact_prompt.md`) and continues from just the task
    plus `compact_summary_prefix.md` + the summary.
  - `capped`: `compact`, plus episode 4's cap: every tool result cut to 2,000 characters (head and tail kept).
- `compact_prompt.md`, `compact_summary_prefix.md`: copied from openai/codex @ 89461ea
  (`codex-rs/prompts/templates/compact/`).
- `compaction_summary_example_2026-11-21.md`: one summary the model wrote.
- `llm.py`: one call to Ollama's Responses API, with a per-reply cap (`MAX_OUTPUT_TOKENS`, default 8,000).

```bash
ollama pull qwen3:4b
python3 budget_harness.py unlimited     # or compact / capped
```

The window is simulated: the server (Ollama, `OLLAMA_CONTEXT_LENGTH=32768`) accepts larger requests, so you can see
what would not have fit. WARNING: the harness runs the model's commands. Use a container or a throwaway VM. Python
standard library only.

Results (2026-11-21, Qwen3 4B on Ollama, CPU, in containers without internet; full output in `run_2026-11-21.txt`):

| mode | runs | what happened | tokens sent per call | answers right |
|---|---|---|---|---|
| unlimited | 1 | `cat` of all three files in one reply (14,859 + 13,092 + 1,800 chars) | 224, **7,887** | 3 of 3 |
| compact | 3 | the same `cat`s; compaction request **7,835** tokens; summary 527–1,727 tokens written, ~1,100–1,180 characters kept | 224, 545–595 | 3 of 3 each |
| capped | 2 | the same `cat`s, two of them cut to 2,025 characters; no compaction needed | 224, 1,850 | 2 of 3 each: answered `cover.jpg` |

- Compaction worked, but the compaction request itself was 7,835 tokens, almost twice the window: to summarize, the
  model must first read everything. With a real 4,096-token window it would have failed. Codex avoids this by
  compacting at 90% of a large window and capping every command's output.
- Capping kept every request small, but `"cover": "youtube/covers/AgentsCover.jpg"` is line 59 of 301 in
  `stepstone/build_packs.py`, inside the cut middle. Both runs answered `cover.jpg`, from the docstring at the top
  (*"cover.jpg   series artwork (when available)"*). Searching (episode 4) would have read line 59 directly.

How Codex CLI does it (openai/codex @ 89461ea):

- Fallback model metadata (`models-manager/src/model_info.rs`, used for Qwen3 4B): `context_window` 272,000,
  `effective_context_window_percent` 95, output truncation 10,000 bytes.
- Auto-compaction limit: 90% of the context window (`auto_compact_token_limit` in `protocol/src/openai_models.rs`), or a
  configured `model_auto_compact_token_limit`, whichever is lower.
- The compaction prompt and the prefix that introduces the summary to the next turn: `prompts/templates/compact/`.
