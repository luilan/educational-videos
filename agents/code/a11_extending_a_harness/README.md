# How AI Agents Work, episode 11 · Extending a Harness

Plug an MCP server into a harness, and its tools appear next to the built-in ones.

- `tickets_mcp.py`: a minimal MCP server (45 lines, Python standard library): JSON-RPC 2.0 over stdin/stdout, one
  message per line; answers `initialize`, `tools/list` and `tools/call` for one tool, `get_ticket(id)`, backed by a
  small dict (ticket T-42 describes the bug in `demo/invoice.py`).
- `mcp_harness.py`: the tiny harness plus `MCPClient`: it starts the server, does the handshake, asks for
  `tools/list`, renames each tool `mcp__<server>__<tool>` and offers it to the model next to `shell`; calls to it are
  forwarded as `tools/call`.
- `codex_mcp_capture_2026-11-21.json`, `codex_trace_2026-11-21.jsonl`: Codex CLI 0.163.0 with the same server
  (`-c 'mcp_servers.tickets.command="python3"' -c 'mcp_servers.tickets.args=["/work/tickets_mcp.py"]'`), asked to read
  ticket T-42 and summarize it, captured through a logging proxy.
- `llm.py`: one call to Ollama's Responses API, with a per-reply cap (`MAX_OUTPUT_TOKENS`, default 8,000).

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python3 tickets_mcp.py
ollama pull qwen3:4b && python3 mcp_harness.py
```

WARNING: the harness runs the model's commands. Use a container or a throwaway VM. Python standard library only.

Results (2026-11-21, Qwen3 4B on Ollama, CPU, in containers without internet; full output in `run_2026-11-21.txt`):

- Both runs called the MCP tool first: `mcp__tickets__get_ticket({"id": "T-42"})` → *"Customer reports invoice totals
  are too high by exactly the discount amount. … Expected: total = subtotal - discount + VAT."*
- Then both ran the test and tried `sed` replacements on lines that do not exist (`total = subtotal * 1.10 -
  discount`, `total = subtotal + discount + vat`, …), never reading `invoice.py`; run 1 stopped after 4 model calls,
  run 2 hit the 10-call limit; the tests still failed in both.
- The tool definitions were the same size (225 characters each for `shell` and the MCP tool).
- Codex read the ticket with the same server and summarized it in one sentence (*"Customer reports invoice 2026-0042 is
  overstated by the full 10% loyalty discount …"*).

How Codex CLI does it (openai/codex @ 89461ea, and the capture):

- MCP servers are configured as `mcp_servers.<name>.command` / `.args`.
- In the request, the server's tools arrive as a **namespace** tool, `{"type": "namespace", "name": "mcp__tickets",
  "tools": [{"name": "get_ticket", …}]}`, next to Codex's own tools; Codex also added `list_mcp_resources`,
  `list_mcp_resource_templates` and `read_mcp_resource`.
- The model's call carries `"name": "get_ticket", "namespace": "mcp__tickets"`; the result starts with a `Wall time`
  header like a command's.
- Many tools can be deferred: a `tool_search` tool (*"Search query for deferred tools."*) loads matching ones on demand
  (`core/src/tools/handlers/tool_search_spec.rs`).
