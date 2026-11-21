"""Study guide content for How AI Agents Work: Inside the Harness, episode 14: Putting It Together.

Build:  python framework/study_guide.py agents a14 --video agents/media/videos/a14_scene/1080p60/PuttingItTogetherVideo.mp4
Every number comes from code/a14_putting_it_together (2026-11-21): tiny_agent.py with Qwen3 4B on Ollama (3 runs), and
earlier episodes' Codex CLI captures.
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 14",
    "title": "Putting It Together",
    "tagline": "One tiny agent with every part from this series",
    "duration": "2:26",
    "intro": """<p>The finale puts every part built in this series into one file, <code>tiny_agent.py</code>: the loop,
tool calls, search and output caps, patches, sessions, the sandbox, rules and approvals, AGENTS.md, compaction, MCP and
parallel calls. It then runs one realistic task three times and asks what the harness can, and cannot, do for a small
model.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> the whole series. Code:
<code>code/a14_putting_it_together</code> (Linux with bubblewrap and a local Ollama with <code>qwen3:4b</code>).</div>""",
}

CONCEPTS = [
    {
        "title": "Every part in one file",
        "segment": (7.7, 61.8),
        "figures": [{"t": 36.1, "caption": "The parts of tiny_agent.py and the episodes that introduced them."},
                    {"t": 51.0, "caption": "The task: fix the bug ticket T-42 describes; the ticket is in an MCP server, "
                                           "AGENTS.md asks for a CHANGELOG line, commands run in the sandbox."},
                    {"t": 61.5, "caption": "First request: 606 tokens for tiny_agent, about 7,900 for Codex CLI."}],
        "body": [
            """<p><code>tiny_agent.py</code> is 293 lines of standard-library Python. Its parts: the loop with stop
conditions and a stuck detector (episode 2); tool calls with errors returned as results (3); search-first instructions
and a head-and-tail output cap (4); <code>apply_patch</code>, intercepted from commands (5); <code>exec_command</code> and
<code>write_stdin</code> sessions (6); a bubblewrap sandbox (7); prefix rules and on-request escalation (8);
AGENTS.md as a user message (9); compaction (10); MCP servers (11); parallel calls (12).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Its first request was 606–609 tokens: instructions, three tools (one from MCP) and AGENTS.md. Codex
CLI's first request with the same model was 7,852 tokens in episode 4 and 7,928 in episode 9.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>A useful harness is a loop plus a handful of
well-defined parts; each solves one problem the model cannot solve for itself.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put these parts in the order the series introduced them.",
             "answer": "agent loop → apply_patch → sandbox → AGENTS.md → MCP.",
             "why": "Episodes 2, 5, 7, 9 and 11.",
             "key": {"items": ["agent loop", "apply_patch", "sandbox", "AGENTS.md", "MCP"]}},
            {"kind": "number", "q": "Codex's first request was 7,852 tokens, tiny_agent's 606. About how many times "
                                    "larger, to one decimal?",
             "answer": "About 13.0.", "why": "7,852 / 606 ≈ 12.96.",
             "key": {"parts": [{"label": None, "value": 13.0, "tol": 0.1, "unit": None}]}},
        ],
    },
    {
        "title": "Three runs",
        "segment": (62.3, 105.7),
        "figures": [{"t": 80.0, "caption": "Run 1: ticket, search, read, then a patch wrapped in Unix `patch`, which the "
                                           "harness did not intercept."},
                    {"t": 90.9, "caption": "Runs 2 and 3: ticket, search, an empty reply; all three runs left the tests "
                                           "failing, nothing outside the project touched."},
                    {"t": 105.4, "caption": "Every part did its job; the model is the limit."}],
        "body": [
            """<p>Run 1 started well: <code>mcp__tickets__get_ticket {"id": "T-42"}</code>, <code>grep -rn 'invoice'
.</code>, <code>cat invoice.py</code>. Then it sent the Codex patch format inside the Unix <code>patch -p1</code>
command. The harness only intercepts commands that start with <code>apply_patch</code>, so <code>patch</code> received
it and exited 2; the model spent its last reply reasoning about line numbers.</p>""",
            "{fig0}",
            """<p>Runs 2 and 3 read the ticket, ran the same <code>grep</code>, and then replied with nothing. All three
runs left the tests failing and added no CHANGELOG line; nothing was blocked, and nothing outside the project was
written.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>A harness can make an agent safe, cheap and checkable; it
cannot make a small model careful.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why didn't the harness apply run 1's patch?",
             "answer": "The command started with patch -p1, not apply_patch, so it was not intercepted; the Unix patch "
                       "tool cannot read Codex's format.",
             "why": "Interception matches the command's first word."},
            {"kind": "tf", "q": "“In all three runs the model read the ticket through MCP.”",
             "answer": "True.", "why": "Each run's first call was mcp__tickets__get_ticket.", "key": {"value": True}},
        ],
    },
    {
        "title": "What the series showed",
        "segment": (106.2, 141.8),
        "figures": [{"t": 121.4, "caption": "The model's habits across the series, and the part of the harness that caught "
                                            "or exposed each one."},
                    {"t": 132.3, "caption": "The whole agent: the episode-2 loop with every part plugged in."}],
        "body": [
            """<p>Across the series the small model guessed line numbers (episode 4: <code>grep -n</code> gives real
ones), rewrote files it had never read (episode 5: patches refuse context that is not there), wrote tool calls as
plain text (nothing runs), and knew rules it did not follow (episode 9: checks and hooks). Each time a part of the
harness either contained the mistake or made it visible.</p>""",
            "{fig0}",
            """<p>Underneath it all is episode 2's loop: send the conversation, run the tool calls (in parallel when
possible), append the results, compact when the window fills, and stop when the model stops calling tools.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Good harness design does not replace a capable model; it
makes any model's work safer, cheaper and easier to check.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which part made rewritten-from-memory edits impossible to apply silently?",
             "options": ["The sandbox", "apply_patch's context matching", "Compaction", "MCP"],
             "answer": "B.", "why": "A patch whose old lines are not in the file is refused.", "key": {"choice": 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> The agent loop, with parallel calls and compaction.",
             "code": """def run(self, task):
    history = self.history_start(task)
    for step in range(1, MAX_STEPS + 1):
        r = respond(history, self.tools, INSTRUCTIONS)
        calls = [o for o in r["output"] if o["type"] == "function_call"]
        history += r["output"]
        if not calls:
            return final_text(r)
        with ThreadPoolExecutor() as pool:
            outs = list(pool.map(self.call_tool, calls))
        for c, out in zip(calls, outs):
            history.append({"type": "function_call_output",
                            "call_id": c["call_id"], "output": out})
        if r["usage"]["input_tokens"] > WINDOW * 9 // 10:
            history = self.compact(history, task)""",
             "answer": "tiny_agent.py's Agent.run is this loop plus the stuck detector.",
             "why": "Every other part lives in call_tool, history_start and compact."},
        ],
    },
]
