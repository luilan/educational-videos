"""Study guide content for How AI Agents Work: Inside the Harness, episode 11: Extending a Harness.

Build:  python framework/study_guide.py agents a11 --video agents/media/videos/a11_scene/1080p60/ExtendingHarnessVideo.mp4
Every number comes from code/a11_extending_a_harness (2026-11-21): tickets_mcp.py, mcp_harness.py with Qwen3 4B on
Ollama (2 runs) and a Codex CLI 0.163.0 run with the same MCP server, captured through a logging proxy.
Codex code: openai/codex @ 89461ea.
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 11",
    "title": "Extending a Harness",
    "tagline": "How new tools plug in, without changing the harness",
    "duration": "2:21",
    "intro": """<p>Real work needs tools a harness doesn't ship with: a ticket tracker, documentation, a database.
<b>MCP</b>, the Model Context Protocol, is the standard plug: a separate program (an MCP server) that the harness
starts and talks to in JSON-RPC over stdin and stdout. This lesson builds a 45-line server, connects it to our harness
and to Codex CLI, and looks at what new tools change, and what they cost.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–10, especially 3 (tool calling). Code:
<code>code/a11_extending_a_harness</code> (<code>tickets_mcp.py</code> runs on its own; <code>mcp_harness.py</code>
needs a local Ollama with <code>qwen3:4b</code>).</div>""",
}

CONCEPTS = [
    {
        "title": "An MCP server, and the handshake",
        "segment": (7.7, 47.7),
        "figures": [{"t": 37.3, "caption": "An MCP server is a program the harness starts; JSON-RPC over stdin/stdout, one "
                                           "message per line."},
                    {"t": 47.4, "caption": "The handshake: initialize, notifications/initialized, then tools/list returns "
                                           "each tool's name, description and input schema."}],
        "body": [
            """<p>Instead of adding every integration to the harness, MCP puts each one in a <b>server</b>: a separate
program the harness starts, talking JSON-RPC 2.0 over its standard input and output, one message per line. Ours,
<code>tickets_mcp.py</code>, is 45 lines of Python with one tool, <code>get_ticket(id)</code>.</p>""",
            "{fig0}",
            """<p>The conversation starts with a handshake: the harness sends <code>initialize</code> and gets the
server's info back, sends the notification <code>notifications/initialized</code> (no reply), then asks
<code>tools/list</code> and receives, for each tool, its name, description and JSON-schema
<code>inputSchema</code>.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>An MCP server is just a program with a small, standard
conversation: identify, list your tools, run a tool.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put the start of an MCP session in order.",
             "answer": "initialize → notifications/initialized → tools/list.",
             "why": "Identify first, then discover tools.",
             "key": {"items": ["initialize", "notifications/initialized", "tools/list"]}},
            {"kind": "tf", "q": "“An MCP server must be a web service.”",
             "answer": "False.", "why": "Ours is a local program speaking JSON-RPC over stdin and stdout.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "From MCP tool to model tool",
        "segment": (48.2, 91.5),
        "figures": [{"t": 59.3, "caption": "The harness renames the tool mcp__tickets__get_ticket and offers it next to "
                                           "shell, in the same shape."},
                    {"t": 67.8, "caption": "A call: model → harness → tools/call → server; the text comes back as the "
                                           "result."},
                    {"t": 91.2, "caption": "Both runs read the ticket first, then patched lines that do not exist."}],
        "body": [
            """<p>The harness turns each MCP tool into an ordinary tool definition (name, description, parameters),
prefixing the server's name: <code>get_ticket</code> becomes <code>mcp__tickets__get_ticket</code>. To the model it
looks exactly like <code>shell</code> (both definitions were 225 characters). When the model calls it, the harness
sends <code>tools/call</code> to the server and returns the text as the tool result.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Asked to fix the bug that ticket T-42 describes, both runs called
<code>mcp__tickets__get_ticket({"id": "T-42"})</code> first and got the ticket: totals too high by exactly the
discount. Then both ran the test and tried <code>sed</code> replacements on lines they had invented
(<code>total = subtotal * 1.10 - discount</code>, <code>total = subtotal + discount + vat</code>), never reading
<code>invoice.py</code>; the tests still failed.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>An MCP tool is indistinguishable from a built-in one; it
adds what the model can do, not how well it does it.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "What does the harness send to the server when the model calls "
                                   "<code>mcp__tickets__get_ticket</code>?",
             "answer": "A JSON-RPC tools/call request with name get_ticket and the model's arguments.",
             "why": "The prefix only says which server; the server sees its own tool name."},
            {"kind": "number", "q": "In how many of the 2 runs was the MCP tool the first call?",
             "answer": "2.", "why": "Both started with get_ticket T-42.",
             "key": {"parts": [{"label": None, "value": 2, "tol": 0.5, "unit": None}]}},
        ],
    },
    {
        "title": "MCP in Codex, and the cost of tools",
        "segment": (92.0, 133.1),
        "figures": [{"t": 110.3, "caption": "Codex: configured with two settings; the tools arrive in a namespace; the "
                                            "result has a Wall time header."},
                    {"t": 124.7, "caption": "Every tool definition is sent on every call; Codex can defer tools behind a "
                                            "tool_search."},
                    {"t": 132.8, "caption": "An MCP client: write one line of JSON, read one line back."}],
        "body": [
            """<p>Codex connects the same server from configuration (<code>mcp_servers.tickets.command</code> and
<code>.args</code>). In the captured request the server's tools arrived as a namespace tool,
<code>{"type": "namespace", "name": "mcp__tickets", "tools": [{"name": "get_ticket", …}]}</code>, next to Codex's own
tools; the model's call named <code>get_ticket</code> in namespace <code>mcp__tickets</code>, and the result began
with <code>Wall time: 0.0104 seconds</code>.</p>""",
            "{fig0}",
            """<p>Every tool definition travels with every request, so ten servers with ten tools each would fill a
small window before the task begins. Codex can <b>defer</b> tools: the model gets a <code>tool_search</code> tool
(<i>“Search query for deferred tools.”</i>) and only matching tools are loaded.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>Plugging in tools is cheap to build and costs context on
every call; load them only when needed.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "How did Codex present the MCP tool to the model?",
             "options": ["As a shell command", "In a namespace tool named mcp__tickets",
                         "Inside the system prompt as text", "It did not"],
             "answer": "B.", "why": "The capture shows {\"type\": \"namespace\", \"name\": \"mcp__tickets\", …}.",
             "key": {"choice": 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Send one JSON-RPC request to an MCP server and read the "
                                  "reply.",
             "code": """def request(self, method, params=None):
    self.n += 1
    msg = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}
    self.p.stdin.write(json.dumps(msg) + "\\n")
    self.p.stdin.flush()
    return json.loads(self.p.stdout.readline())["result"]""",
             "answer": "mcp_harness.py's MCPClient uses it for initialize, tools/list and tools/call.",
             "why": "One line out, one line back: the whole transport."},
        ],
    },
]
