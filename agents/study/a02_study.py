"""Study guide content for How AI Agents Work: Inside the Harness, episode 2: The Agent Loop.

Build:  python framework/study_guide.py agents a02 --video agents/media/videos/a02_scene/1080p60/AgentLoopVideo.mp4
Every number comes from code/a02_the_agent_loop/agent_loop.py (2026-10-21, Qwen3 4B on Ollama, CPU, in a container) or
from the Codex CLI 0.163.0 trace of episode 1. Codex code: openai/codex @ 89461ea.
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 2",
    "title": "The Agent Loop",
    "tagline": "What happens each time around, and what makes it stop",
    "duration": "2:09",
    "intro": """<p>This lesson answers one question: what exactly happens each time an agent goes around its loop, and
what makes it stop? Each step sends the <b>whole conversation</b> to the model, runs any tool calls, and adds the
results. Codex ends a turn when a reply contains no tool call; it has no step limit. A task that can never finish shows
why a harness needs its own <b>guards</b>: here the command timeout, not the model, ended the run.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 1 (what a harness is). Code:
<code>code/a02_the_agent_loop</code> (Python standard library + a local model served by Ollama; run it in a container,
it executes the model's commands).</div>""",
}

CONCEPTS = [
    {
        "title": "One step of the loop",
        "segment": (8, 43),
        "figures": [{"t": 25.5, "caption": "One step: send the conversation; if the reply has tool calls, run them, add the "
                                           "results and call again; otherwise the turn is over."},
                    {"t": 43.0, "caption": "A real run: five model calls, and this time the model read the file and "
                                           "re-ran the test."}],
        "body": [
            """<p>Each step of the agent loop does the same thing: send the conversation to the model; if the reply
contains tool calls, run each one, append its result to the conversation and call the model again; if it contains
none, the turn is over.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>With <code>agent_loop.py</code> and Qwen3 4B, fixing the bug took five model calls: run the test (exit 1),
<code>cat stats.py</code>, rewrite it with <code>echo</code>, run the test again (exit 0), answer in words. Unlike in
episode 1, the model read the file first and checked its own work: the same model behaves differently from run to
run.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A step is: model call → run the tool calls → append the results.
No tool call means the turn is over.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put the five model calls of the real run in order.",
             "answer": "run the test → read stats.py → rewrite stats.py → run the test again → answer in words.",
             "why": "Reading before editing and testing after editing is the careful pattern.",
             "key": {"items": ["run the test", "read stats.py", "rewrite stats.py", "run the test again", "answer in words"]}},
            {"kind": "tf", "q": "“If the model's reply contains no tool call, the harness ends the turn.”",
             "answer": "True.", "why": "Nothing is left to run, so the reply is the answer.", "key": {"value": True}},
        ],
    },
    {
        "title": "The conversation is sent again every time",
        "segment": (44, 66),
        "figures": [{"t": 65.0, "caption": "Tokens sent per model call: our harness 197 → 492; Codex 3,779 → 7,890 (86% "
                                           "of Codex's input served from cache)."}],
        "body": [
            """<p>The model keeps no memory between calls. The harness therefore sends the <b>whole conversation</b> on
every step: instructions, tool definitions, the task and every earlier call and result. Each step is bigger than the
last:</p>""",
            """<table><tr><th>model call</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th></tr>
<tr><td>our harness (tokens sent)</td><td>197</td><td>322</td><td>388</td><td>449</td><td>492</td></tr>
<tr><td>Codex (tokens sent)</td><td>3,779</td><td>4,881</td><td>6,191</td><td>7,551</td><td>7,890</td></tr></table>""",
            "{fig0}",
            """<p>Resending is cheaper than it looks because model servers cache the part of the prompt they have already
processed: in the Codex run, 26,171 of 30,292 input tokens (86%) came from the cache.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Every step resends the full conversation, so cost and context
use grow with each step; prompt caching softens the cost.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Codex sent 30,292 input tokens over the run, 26,171 of them from cache. What share was "
                                    "cached, in percent?",
             "answer": "About 86%.", "why": "26,171 / 30,292 ≈ 0.864.",
             "key": {"parts": [{"label": None, "value": 86, "tol": 1, "unit": "%"}]}},
            {"kind": "number", "q": "In our harness, how many tokens did the conversation grow from call 1 (197) to call 5 "
                                    "(492)?",
             "answer": "295.", "why": "492 − 197 = 295.",
             "key": {"parts": [{"label": None, "value": 295, "tol": 0.5, "unit": None}]}},
            {"kind": "short", "q": "Why must the harness resend the whole conversation on every call?",
             "answer": "The model keeps no memory between calls; everything it should know must be in the request.",
             "why": "The conversation lives in the harness, not in the model."},
        ],
    },
    {
        "title": "How a turn ends",
        "segment": (66, 101),
        "figures": [{"t": 82.0, "caption": "Codex: each tool call sets needs_follow_up; a reply without one ends the turn "
                                           "(after the stop hooks). No fixed step limit."},
                    {"t": 100.0, "caption": "An impossible task: the model's endless wait loop was killed by the 30-second "
                                            "timeout; then it gave up."}],
        "body": [
            """<p>In Codex (<code>codex-rs/core/src</code>), every tool call sets <code>needs_follow_up = true</code>
(<code>stream_events_utils.rs</code>); after each model call, <code>session/turn.rs</code> checks the flag and, if it is
false, runs the <b>stop hooks</b> and ends the turn. A user interrupt, an error or a token/time budget can also end a
turn. There is no fixed limit on the number of steps.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>We gave the model a task that can never finish: wait for <code>ready.txt</code>, which nobody creates.
It wrote <code>while [ ! -f ready.txt ]; do sleep 1; done</code>, a loop that never ends. The harness killed it after
30 seconds; on the next call the model gave up and explained, ending the loop after 2 calls.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The loop ends when the model stops asking for tools; the
harness must still protect against tasks and commands that never end.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "In the impossible “wait for ready.txt” task, what stopped the endless wait?",
             "options": ["The step limit of 8 calls", "The stuck detector", "The 30-second command timeout",
                         "Codex's needs_follow_up flag"],
             "answer": "C.", "why": "The command never returned; only the timeout could end it.", "key": {"choice": 2}},
            {"kind": "tf", "q": "“Codex stops every turn after a fixed number of model calls.”",
             "answer": "False.", "why": "It has no fixed step limit; turns end on a reply without tool calls, an interrupt, "
                                        "an error or a budget.", "key": {"value": False}},
        ],
    },
    {
        "title": "Guards in the harness",
        "segment": (101, 120),
        "figures": [{"t": 113.0, "caption": "Three guards: a 30-second command timeout (triggered), an 8-call limit and a "
                                            "stuck detector (not needed)."},
                    {"t": 119.0, "caption": "Each guard is one line in the loop; each exit records why the loop stopped."}],
        "body": [
            """<p>Our harness adds three guards: a timeout on every command (30 s), a limit on model calls (8), and a stuck
detector (the same command three times in a row). Every way out of the loop records a reason: answered, stuck or step
limit. In these runs only the timeout fired, but each guard covers a different failure: a command that never returns,
a model that never stops calling tools, and a model that repeats itself.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Never trust the model to stop: bound each command, bound the
number of steps, and detect repetition.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Match each failure to its guard, in this order: a command that never returns, a model "
                                   "that never stops calling tools, a model that repeats the same call. Put the guards in "
                                   "that order.",
             "answer": "command timeout → step limit → stuck detector.",
             "why": "Each guard bounds a different kind of runaway.",
             "key": {"items": ["command timeout", "step limit", "stuck detector"]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Add a stuck detector to a loop that records each command in a "
                                  "list <code>cmds</code>.",
             "code": """if len(cmds) >= 3 and len(set(cmds[-3:])) == 1:
    return "stuck: the same command 3 times in a row\"""",
             "answer": "agent_loop.py uses exactly this test after each step.",
             "why": "Three identical commands in a row usually mean the model is not learning from the results."},
        ],
    },
]
