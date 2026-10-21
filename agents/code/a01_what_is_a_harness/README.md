# How AI Agents Work, episode 1 · What a Harness Is

A language model only writes text; the harness turns its tool calls into actions. Two scripts, one local model:

- `no_harness.py`: asks the model to fix the bug in `demo/` with no tools. It can only answer with text.
- `tiny_harness.py`: the same task with one tool, `shell`, and a loop: run each tool call, feed back the output, stop
  when the model answers in words. Then checks the result itself.

```bash
# a local model with tool calling, served by Ollama (https://ollama.com)
ollama pull qwen3:4b
python3 no_harness.py
python3 tiny_harness.py      # WARNING: runs the model's commands. Use a container or throwaway VM.
```

`OLLAMA_URL` (default `http://localhost:11434`) and `MODEL` (default `qwen3:4b`) select the server and model. Python
standard library only.

Results (2026-10-19, Qwen3 4B on Ollama 0.40.6, CPU, reasoning effort "none"; model output varies run to run):

1. No harness: the model invented a fix to a `median` function that does not exist and said the test passes.
2. Tiny harness: 3 model calls (run the test → rewrite `stats.py` with `echo` → answer). Independent check: test passes.
   The model never read the file first, so the docstring was lost.
3. Codex CLI 0.163.0 on the same task (traced through a logging proxy): the first request was 8,282 tokens around a
   147-character task (instructions 3,553; skills 522; environment + task 128; nine tool definitions 4,079). Ollama's
   Responses API dropped the `developer` messages, so the model saw 3,779 of 7,854 tokens; restored, Qwen3 4B wrote its
   patch as plain text and nothing ran.
