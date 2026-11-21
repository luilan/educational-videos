# How AI Agents Work, episode 14 · Putting It Together

`tiny_agent.py`: a small coding agent with every part from the series in one file (293 lines, Python standard
library only).

| part | episode | in `tiny_agent.py` |
|---|---|---|
| agent loop, stop conditions, stuck detector | 2 | `Agent.run` |
| tool calls; argument errors returned as results | 3 | `Agent.call_tool` |
| search-first instructions; output cap (head + tail) | 4 | `INSTRUCTIONS`, `truncate_middle` |
| `apply_patch` intercepted from commands; refuses paths outside the project | 5 | `apply_patch` |
| `exec_command` / `write_stdin` sessions | 6 | `Session`, `TOOLS` |
| bubblewrap sandbox: read-only system, writable project, no network | 7 | `bwrap` |
| prefix rules + on-request escalation, asking the user on a terminal (denied otherwise) | 8 | `RULES`, `decide`, `ask_user` |
| `AGENTS.md` sent as a user message, up to 32 KiB | 9 | `Agent.history_start` |
| compaction at 90% of a 32,768-token window with Codex's prompt | 10 | `Agent.compact` |
| MCP servers (`tickets_mcp.py`) | 11 | `MCPClient` |
| parallel tool calls | 12 | `ThreadPoolExecutor` in `Agent.run` |

```bash
ollama pull qwen3:4b
python3 tiny_agent.py "Ticket T-42 describes a bug in this project. Fix it."
```

It works on a copy of `demo/` (the invoice project with `AGENTS.md`, `CHANGELOG.md` and `run_tests.py`) and reports
model calls, tokens, commands, patches, blocked and approved commands, compactions, and whether the tests pass and a
CHANGELOG line was added. Needs Linux with bubblewrap (`bwrap`) and a local Ollama with `qwen3:4b` (`OLLAMA_URL`,
`MODEL`); `llm.py` caps each reply at `MAX_OUTPUT_TOKENS` (8,000).

Results (2026-11-21, Qwen3 4B on Ollama, CPU; full output in `run_2026-11-21.txt`):

- First request: 606–609 tokens (instructions, tools, `AGENTS.md`). Codex CLI's first request in episodes 4 and 9 was
  7,852 and 7,928 tokens.
- Run 1: `mcp__tickets__get_ticket {"id": "T-42"}` → `grep -rn 'invoice' .` → `cat invoice.py` → `patch -p1 < <(echo
  "*** Begin Patch …")`: the Codex patch format wrapped in the Unix `patch` command, so the harness did not intercept it
  and `patch` exited 2. The last reply was spent reasoning about line numbers. 5 model calls, 6,432 tokens, tests
  failing.
- Runs 2 and 3: ticket, `grep`, then an empty reply. 3 model calls, 2,207 and 2,199 tokens, tests failing.
- No command was blocked or needed approval, and nothing outside the project was written.
