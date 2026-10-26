# How AI Agents Work, episode 3 · Tool Calling Under the Hood

What a model reads and writes when it "calls a tool".

- `show_prompt.py`: renders our `shell` tool, a call and its result through Qwen3's chat template (needs `transformers`,
  only the tokenizer is downloaded). Output saved in `roundtrip_2026-10-26.txt` and `prompt.txt`.
- `raw_samples.py`: sends `prompt.txt` to Ollama in raw mode (no server-side tool parsing), samples the model and parses
  `<tool_call>` itself with `parse_tool_call()`. Output saved in `raw_samples_2026-10-26.txt`.
- `codex_exec_command_schema.json`, `codex_call_and_result.json`: captured from Codex CLI 0.163.0 through a logging
  proxy (Qwen3 4B on Ollama, in containers without internet).

```bash
pip install transformers && python3 show_prompt.py
ollama pull qwen3:4b && python3 raw_samples.py 10
```

Results (2026-10-26):

1. The tool adds 139 tokens to the first prompt (58 → 197). "Thinking off" is an empty `<think></think>` block.
2. 10/10 raw samples wrote the same valid call (~25 tokens), after 251–494 tokens of output, nearly all reasoning.
3. A call plus its result adds 43 tokens; the result comes back as a new turn in `<tool_response>` tags; the template
   drops the reasoning.
4. Codex's `exec_command` has 10 parameters (only `cmd` required); its definition is 421 tokens vs 58 for ours. Results
   carry the call's ID and a header. Invalid arguments become the result `failed to parse function arguments: …`
   (openai/codex @ 89461ea, `codex-rs/core/src/tools/handlers/mod.rs`).
