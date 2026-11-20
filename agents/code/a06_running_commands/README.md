# How AI Agents Work, episode 6 · Running Commands

A task that needs a process to keep running, in two harness designs.

- `demo/server.py`: a tiny web service that runs until stopped; `demo/check.py` asks it for `/health`
  (`{"status": "ok", "version": "1.4.2"}`).
- `run_harness.py`:
  - `blocking`: one `shell` tool; every command runs to the end with `subprocess.run(..., capture_output=True,
    timeout=30)`, like the harness of episodes 1–5.
  - `sessions`: Codex-style `exec_command` (returns after `yield_time_ms`, default 10,000, or when the process exits;
    a process still running keeps a session ID) and `write_stdin` (type into a session, or send nothing to read its new
    output; default wait 250 ms). `Session` is a process plus a thread that collects its output.
- `llm.py`: one call to Ollama's Responses API, with a per-reply cap (`MAX_OUTPUT_TOKENS`, default 8,000).

```bash
ollama pull qwen3:4b
python3 run_harness.py blocking
python3 run_harness.py sessions
```

WARNING: the harness runs the model's commands (and starts a server on port 8000). Use a container or a throwaway VM.
Python standard library only. The harness stops leftover `server.py` processes at the end.

Results (2026-11-20, Qwen3 4B on Ollama, CPU, in containers without internet; full output in `run_2026-11-20.txt`;
runs vary):

| mode | run | what happened | answer |
|---|---|---|---|
| blocking | 1 | `python3 server.py &` hung for the full 30 s, then `check.py` worked (the run then crashed in our cleanup) | (not recorded) |
| blocking | 2 | never started the server: `check.py` → `Connection refused`; told the user to start it | wrong |
| blocking | 3 | `python3 server.py &` hung 30 s; the timeout killed the shell but not the server; `check.py` worked; a second `server.py &` hit "address in use" | **1.4.2**, by accident |
| sessions | 2 | `exec_command: python3 server.py` → after 10.1 s *Process running with session ID 1*; `check.py` → exit 0; 2 model calls | **1.4.2** |
| sessions | 3 | the first reply was empty | none |

- Why `server.py &` still blocks: `capture_output` keeps pipes to the command's output, and the background server
  inherits them, so the harness waits for an end-of-output that never comes until the timeout.
- Killing the shell at the timeout does not kill its background children: the server kept running, untracked. A
  session (`start_new_session=True`, killed with `os.killpg` at the end) keeps it under the harness's control.

How Codex CLI does it (openai/codex @ 89461ea):

- `exec_command` (`core/src/tools/handlers/unified_exec.rs`): `yield_time_ms` default 10,000, clamped to 250–30,000
  (`core/src/unified_exec/mod.rs`); `write_stdin` default 250 ms; `max_output_tokens` default 10,000.
- A result for a process that is still running includes `Process running with session ID <id>`
  (`core/src/tools/context.rs`).
- At most 64 processes (`MAX_UNIFIED_EXEC_PROCESSES`); when the store is full, old ones are pruned, exited ones first
  (`prune_processes_if_needed` in `core/src/unified_exec/process_manager.rs`).
