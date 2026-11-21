"""How AI Agents Work: Inside the Harness, episode 11 — Extending a Harness. One entry per narration section.

Numbers come from code/a11_extending_a_harness (2026-11-21: tickets_mcp.py, mcp_harness.py with Qwen3 4B on Ollama, CPU,
in containers without internet; a Codex CLI 0.163.0 request with the same MCP server, captured through the logging
proxy) and openai/codex @ 89461ea (protocol/src/mcp.rs, core/src/tools/handlers/tool_search_spec.rs).
"""
TITLE = "Extending a Harness"
TAGLINE = "How new tools plug in, without changing the harness"
NEXT = "Sub-agents and Parallel Tool Calls"
SECTIONS = [
    # 1. Hook
    "Our harness can run commands and edit files. But real work also needs a ticket tracker, documentation, a "
    "database. Rewriting the harness for each one doesn't scale. So there is a standard plug: MCP, the Model Context "
    "Protocol.",
    # 2. A server
    "An MCP server is just a program. The harness starts it, and they talk JSON R P C over its standard input and "
    "output, one message per line. Ours is forty-five lines of Python, with one tool: get ticket.",
    # 3. Handshake
    "First, a handshake: initialize, then initialized. Then the harness asks for tools list, and gets back each "
    "tool's name, description, and input schema.",
    # 4. Renamed
    "The harness renames each tool with its server's name, so get ticket becomes M C P, tickets, get ticket, and "
    "offers it next to its own shell tool. To the model, both look the same.",
    # 5. A call
    "When the model calls it, the harness forwards the call to the server as tools call, and hands the text it gets "
    "back to the model as the result.",
    # 6. Runs
    "In both of our runs, the model's first move was the new tool: get ticket, T forty-two. The ticket said the "
    "total was too high by exactly the discount. MCP got the information in.",
    # 7. Then
    "Then the model ran the test, and tried to fix a line it had never read. Both runs ended with the test still "
    "failing. A new tool adds what the model can do, not how well it does it.",
    # 8. Codex
    "Codex does the same, from two lines of configuration: the server's command and its arguments. In our captured "
    "request, the tool arrived in a namespace called M C P tickets, next to Codex's own tools. The model called get "
    "ticket, and the result came back with a wall time, like any command.",
    # 9. Cost
    "Every tool's definition is sent on every call. Ten servers with ten tools each would fill a small window before "
    "the task starts. So Codex can defer tools: the model gets a tool search, and loads only the ones it needs.",
    # 10. Code
    "In code, an MCP client is a process, and a request function: write one line of JSON, read one line back.",
    # 11. Outro
    "Next: sub-agents and parallel tool calls. How one agent splits its work.",
]
