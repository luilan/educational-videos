"""Study guide content for How AI Agents Work: Inside the Harness, episode 12: Sub-agents and Parallel Tool Calls.

Build:  python framework/study_guide.py agents a12 --video agents/media/videos/a12_scene/1080p60/SubagentsParallelVideo.mp4
Every number comes from code/a12_subagents_and_parallel (2026-11-21): parallel_harness.py with Qwen3 4B on Ollama
(serial ×1, parallel ×2, single ×1, subagent ×2). Codex code: openai/codex @ 89461ea.
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 12",
    "title": "Sub-agents and Parallel Tool Calls",
    "tagline": "How one agent splits its work",
    "duration": "2:10",
    "intro": """<p>Two ways for an agent to do more at once: <b>parallel tool calls</b> (several calls in one reply, run
at the same time) and <b>sub-agents</b> (a fresh agent with its own conversation for one piece of the job). This lesson
measures both with our harness and shows how Codex CLI decides what may run together and how it limits sub-agents.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–11, especially 4 (the search task). Code:
<code>code/a12_subagents_and_parallel</code> (<code>parallel_harness.py</code> needs a local Ollama with
<code>qwen3:4b</code>).</div>""",
}

CONCEPTS = [
    {
        "title": "Parallel tool calls",
        "segment": (7.7, 57.4),
        "figures": [{"t": 26.6, "caption": "One reply, three tool calls."},
                    {"t": 44.6, "caption": "The same three 3-second checks: 9.0 s one after another, 3.0 s at the same "
                                           "time; same answer, 495 tokens."},
                    {"t": 57.1, "caption": "Codex: parallel-safe tools share a read lock; the others take a write lock."}],
        "body": [
            """<p>A model reply can contain several tool calls. Asked to run three independent checks (lint, types,
tests, 3 seconds each), the model put all three calls in a single reply in all four runs. The harness can run them
one after another or concurrently.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Run serially, the tools took 9.0 s; in a thread pool, 3.0 s. The answer and the tokens (495) were the
same. Codex decides per tool: if <code>supports_parallel_tool_calls()</code> is true (as for
<code>exec_command</code>), the call takes a shared read lock and runs alongside others; otherwise it takes an
exclusive write lock and waits its turn.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Independent calls from one reply can run at once; the
harness, not the model, decides which tools are safe to overlap.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Three 3-second checks run one after another and then concurrently. How many seconds "
                                    "are saved?",
             "answer": "6.", "why": "9.0 − 3.0 = 6.0 s.",
             "key": {"parts": [{"label": None, "value": 6, "tol": 0.2, "unit": "s"}]}},
            {"kind": "mc", "q": "In Codex, what does a tool that is not parallel-safe take before running?",
             "options": ["A read lock", "A write lock", "No lock", "A new thread"],
             "answer": "B.", "why": "The exclusive write lock makes it run alone.", "key": {"choice": 1}},
        ],
    },
    {
        "title": "Sub-agents",
        "segment": (57.9, 100.3),
        "figures": [{"t": 69.6, "caption": "spawn_agent: the child gets a fresh conversation and its own tools; only its "
                                           "final answer comes back."},
                    {"t": 84.8, "caption": "Tokens sent: single agent 1,153; with a sub-agent the parent sent 619 and 648, "
                                           "the children 2,259 and 954."},
                    {"t": 100.0, "caption": "All three answered line 224; the parent's task for the child dropped part of "
                                            "the question."}],
        "body": [
            """<p>In <code>subagent</code> mode the parent has one tool, <code>spawn_agent(message)</code>. The
harness starts the same agent loop with a fresh conversation (the message as its task, the <code>shell</code> tool,
search-first instructions) and returns only the child's final answer as the tool result.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>On episode 4's search task the single agent sent 1,153 tokens. With a sub-agent the parent sent only
619 and 648, while the children's searches (2,259 and 954 tokens) stayed in their own conversations. But all three runs
answered line 224, the near miss from episode 4: a sub-agent keeps the parent's context small, it does not make the
search smarter. And the parent's one-line task (<i>“Find the line in the project that specifies the file name of a
figure image for the study-guide PDF.”</i>) dropped the request for the exact format string.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Sub-agents buy context, not intelligence; the quality of
the handoff message limits what the child can do.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Sub-agent run 1: the parent sent 619 tokens and the child 2,259. How many tokens "
                                    "were sent in total?",
             "answer": "2,878.", "why": "619 + 2,259; more than the single agent's 1,153, but the parent's context stayed "
                                         "small.",
             "key": {"parts": [{"label": None, "value": 2878, "tol": 0.5, "unit": None}]}},
            {"kind": "tf", "q": "“The child's grep output became part of the parent's conversation.”",
             "answer": "False.", "why": "Only the child's final answer was returned.", "key": {"value": False}},
        ],
    },
    {
        "title": "Sub-agents in Codex",
        "segment": (100.8, 121.5),
        "figures": [{"t": 112.3, "caption": "Codex's agent tools and limits: 6 threads, depth 1."},
                    {"t": 121.2, "caption": "In code: a thread pool for one reply's calls; a sub-agent is the same loop with "
                                            "a new conversation."}],
        "body": [
            """<p>Codex's multi-agent tools include <code>spawn_agent</code>, <code>send_input</code>,
<code>wait_agent</code> and <code>close_agent</code> (plus <code>resume_agent</code>). By default at most 6 agent
threads run (<code>DEFAULT_AGENT_MAX_THREADS</code>), and only one level deep (<code>DEFAULT_AGENT_MAX_DEPTH = 1</code>):
a sub-agent cannot spawn its own.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Limits on the number and depth of agents keep delegation
from multiplying cost without bound.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why might a harness forbid sub-agents from spawning their own sub-agents?",
             "answer": "To bound cost and complexity: each level multiplies model calls and makes the work harder to "
                       "follow.",
             "why": "Codex's default depth is 1."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Run one reply's tool calls concurrently.",
             "code": """from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as pool:
    outs = list(pool.map(do, calls))
for c, out in zip(calls, outs):
    history.append({"type": "function_call_output",
                    "call_id": c["call_id"], "output": out})""",
             "answer": "parallel_harness.py does this in parallel mode (and for spawn_agent calls).",
             "why": "map keeps the results in the calls' order, so each output matches its call_id."},
        ],
    },
]
