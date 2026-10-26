# How AI Agents Work, episode 2 · The Agent Loop

`agent_loop.py` is the tiny harness of episode 1 with its stop conditions made explicit. Every step prints the tokens
sent (the whole conversation goes back each time) and every exit records why the loop stopped.

```bash
ollama pull qwen3:4b
python3 agent_loop.py fix     # fix the bug in demo/
python3 agent_loop.py wait    # an impossible task: wait for a file that never appears
```

WARNING: the harness runs the model's commands. Use a container or a throwaway VM. `OLLAMA_URL` and `MODEL` select the
server and model. Python standard library only.

Guards: a 30-second timeout on every command, at most 8 model calls, and a stuck detector (the same command three times
in a row).

Results (2026-10-21, Qwen3 4B on Ollama 0.40.6, CPU; full output in `run_2026-10-21.txt`; runs vary):

- `fix`: 5 model calls: run the test (exit 1) → `cat stats.py` → rewrite with `echo` → run the test (exit 0) → answer.
  Tokens sent per call: 197, 322, 388, 449, 492. Independent check: test passes.
- `wait`: the model wrote `while [ ! -f ready.txt ]; do sleep 1; done; cat ready.txt`; the harness killed it after 30 s;
  on the next call the model gave up and explained (2 calls). Only the timeout guard fired.
- For comparison, Codex CLI 0.163.0 on the `fix` task (episode 1 trace): tokens sent per call 3,779, 4,881, 6,191, 7,551,
  7,890; 26,171 of 30,292 input tokens served from the server's cache. Codex ends a turn when a reply has no tool call
  (`needs_follow_up`, openai/codex @ 89461ea, `codex-rs/core/src/session/turn.rs`); it has no fixed step limit.
