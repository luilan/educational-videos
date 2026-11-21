# How AI Agents Work, episode 9 · Instructions and Memory

How a project tells the agent its own rules, and whether the agent follows them.

- `demo/AGENTS.md`: three rules: run the tests with `python3 run_tests.py`; add one `- Fixed: …` line to
  `CHANGELOG.md` under "Unreleased" for every fix; keep the change small. `demo/invoice.py` has one bug.
- `instructions_harness.py`: the same bug fix in five setups, then a check of the three rules:
  - `none`: no `AGENTS.md`;
  - `file`: `AGENTS.md` in the folder, not sent;
  - `injected`: sent as a user message before the task, in Codex's shape
    (`# AGENTS.md instructions for <folder>` + `<INSTRUCTIONS>…</INSTRUCTIONS>`, `agents_md_message()`);
  - `system`: put in the system instructions instead;
  - `hooks`: `injected`, plus a Stop check: when the model wants to finish, the harness refuses (at most twice) while
    `CHANGELOG.md` is unchanged or `run_tests.py` fails, and tells the model why (like a Codex `Stop` hook).
- `codex_request_2026-11-21.json`, `codex_trace_2026-11-21.jsonl`: a Codex CLI 0.163.0 run in a folder with this
  `AGENTS.md`, asked *"What command runs the tests in this project? Answer in one line."*, captured through a logging
  proxy.
- `llm.py`: one call to Ollama's Responses API, with a per-reply cap (`MAX_OUTPUT_TOKENS`, default 8,000).

```bash
ollama pull qwen3:4b
python3 instructions_harness.py injected      # or none / file / system / hooks
```

WARNING: the harness runs the model's commands. Use a container or a throwaway VM. Python standard library only.

Results (2026-11-20/21, Qwen3 4B on Ollama, CPU, in containers without internet; full output in
`run_2026-11-21.txt`; runs vary):

| setup | run | fixed | used `run_tests.py` | CHANGELOG line | notes |
|---|---|---|---|---|---|
| none | 1 | yes (1-line `sed`) | no | no | |
| file | 1 | no | no | no | never opened `AGENTS.md`; `sed` on a function that does not exist |
| injected | 1 | no | no | no | overwrote `invoice.py` with `echo` (110 lines changed) |
| injected | 2 | yes (1-line `sed`) | no | no | |
| system | 1 | no | no | no | `sed` on invented code |
| system | 2 | yes (1-line `sed`) | no | no | |
| hooks | 1 | (not checked) | no | yes, unprompted | appended to the end of the file; the harness crashed at the Stop check (bug, fixed) |
| hooks | 3 | yes | no | yes, after the Stop check blocked once | appended to the end of the file, three times, then stopped as stuck |
| hooks | 4 | yes | no | yes, unprompted | appended to the end of the file, three times (parallel calls); never blocked |

- Sending `AGENTS.md` costs 107 tokens per call here (first request 288 tokens vs 181).
- In 9 runs the model never ran `run_tests.py`. Every CHANGELOG line went after `## 1.3.0` (`echo … >>
  CHANGELOG.md`), never under `## Unreleased`.
- Our Stop check only tested that `CHANGELOG.md` had changed, so in run 4 a misplaced line passed it. A check is only
  as good as what it tests (the final report checks the placement).
- Asked directly (Codex run), the same model answered `python3 run_tests.py`, quoting `AGENTS.md`.

How Codex CLI does it (openai/codex @ 89461ea):

- `core/src/agents_md.rs`: from the project root (found by `.git`) down to the working folder, every `AGENTS.md` (or
  the local `AGENTS.override.md`) is read and joined; at most `project_doc_max_bytes` = 32 KiB
  (`config/src/config_toml.rs`); nothing is loaded for untrusted projects.
- In the captured request: two developer messages (Codex's base instructions, 16,979 characters; skills, 2,316), then
  a user message with `# AGENTS.md instructions for /work`, the rules in `<INSTRUCTIONS>` tags and an
  `<environment_context>`, then the task.
- Hooks (`codex-rs/hooks`): 14 events, among them `PreToolUse` (a blocked command returns `Command blocked by
  PreToolUse hook: {reason}. Command: {command}` to the model, `core/src/hook_runtime.rs`) and `Stop` (can block the
  end of a turn with a reason).
