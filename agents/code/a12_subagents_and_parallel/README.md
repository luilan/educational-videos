# How AI Agents Work, episode 12 · Sub-agents and Parallel Tool Calls

Two ways for one agent to do more at once.

- `parallel_harness.py`:
  - `serial` / `parallel`: three independent 3-second checks (`demo_checks/check.py lint|types|tests`); the calls from
    one model reply run one after another, or at the same time in a `ThreadPoolExecutor`.
  - `single` / `subagent`: episode 4's search task (*which line decides the file name of a figure image extracted for
    the study-guide PDF?*) with search-first instructions; in `subagent` mode the parent only has a `spawn_agent` tool,
    and each spawned child runs the same agent loop with a fresh conversation and the `shell` tool; only its final
    answer returns to the parent.
- `demo_project/`: episode 4's project. `llm.py`: Ollama Responses API, per-reply cap (`MAX_OUTPUT_TOKENS`, 8,000).

```bash
ollama pull qwen3:4b
python3 parallel_harness.py parallel      # or serial / single / subagent
```

WARNING: the harness runs the model's commands. Use a container or a throwaway VM. Python standard library only.

Results (2026-11-21, Qwen3 4B on Ollama, CPU, in containers without internet; full output in `run_2026-11-21.txt`):

| mode | run | model calls | tools took | tokens sent by the top agent | answer |
|---|---|---|---|---|---|
| serial | 1 | 2 (3 calls in one reply) | 9.0 s | 495 | right |
| parallel | 1, 2 | 2 (3 calls in one reply) | 3.0 s | 495 | right |
| single | 1 | 3 | | 1,153 | `framework/study_guide.py:224` (wrong: line 42) |
| subagent | 1 | parent 2; child 4 (2,259 tokens) | | 619 | line 224 |
| subagent | 2 | parent 2; child 2 (954 tokens) | | 648 | line 224 |

- The model asked for all three checks in a single reply in every run; running them concurrently cut the tool time
  from 9.0 s to 3.0 s with the same answer and tokens.
- A sub-agent kept the parent's conversation small (the child's searches never entered it) but did not change the
  answer, and the parent's one-line task for the child dropped part of the question (*"Find the line in the project
  that specifies the file name of a figure image for the study-guide PDF."*).

How Codex CLI does it (openai/codex @ 89461ea):

- Each tool says whether it is safe to run in parallel (`supports_parallel_tool_calls`); safe calls take a shared read
  lock and run together, the others an exclusive write lock (`core/src/tools/parallel.rs`). `exec_command`,
  `write_stdin`, `view_image` and the MCP resource tools are parallel-safe.
- Sub-agent tools (`core/src/tools/handlers/multi_agents_spec.rs`): `spawn_agent`, `send_input`, `resume_agent`,
  `wait_agent`, `close_agent`; defaults `DEFAULT_AGENT_MAX_THREADS = 6`, `DEFAULT_AGENT_MAX_DEPTH = 1`
  (`core/src/config/mod.rs`).
