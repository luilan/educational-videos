"""Study guide content for How AI Agents Work: Inside the Harness, episode 1: What a Harness Is.

Build:  python framework/study_guide.py agents a01 --video agents/media/videos/a01_scene/1080p60/HarnessVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer, a why and (for graded kinds)
a machine-checkable "key" (framework/answer_keys.py).
Every number comes from runs on 2026-10-19: code/a01_what_is_a_harness (no_harness.py, tiny_harness.py) and Codex CLI
0.163.0 traced through a logging proxy, all with Qwen3 4B on Ollama 0.40.6 (CPU, containers without internet).
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 1",
    "title": "What a Harness Is",
    "tagline": "Everything an agent does happens outside the model",
    "duration": "2:12",
    "intro": """<p>This lesson answers one question: if a language model only reads and writes text, how does an AI
agent run commands and edit files? The answer is the <b>harness</b>: the program around the model. It describes tools to
the model, runs the tool calls the model writes, and feeds the results back, in a loop. A 40-line harness is enough to
fix a real bug; a real one, Codex CLI, wraps your 147-character request in 8,282 tokens of instructions and tool
definitions.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> LLMs in Practice episode 7 (tool use) helps but is not
required. Code: <code>code/a01_what_is_a_harness</code> (Python standard library + a local model served by Ollama).
The tiny harness runs whatever command the model writes: run it in a container or a throwaway VM.</div>""",
}

CONCEPTS = [
    {
        "title": "A model alone only writes text",
        "segment": (8, 37),
        "figures": [{"t": 19.5, "caption": "The model reads text and writes text; everything else is the harness around it."},
                    {"t": 36.5, "caption": "Asked to fix the bug with no tools, Qwen3 4B invents a median() function and "
                                           "claims the test passes."}],
        "body": [
            """<p>A language model maps text to more text. It cannot open a file, run a command or check whether its
answer is right. Everything an <b>AI agent</b> does in the world (reading code, running tests, editing files) is done
by the program around the model, the <b>harness</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The episode's demo project has a one-line bug: <code>mean</code> divides by <code>len(xs) - 1</code>.
Asked to fix it with no tools, Qwen3 4B (4 billion parameters) answered with a confident fix to a <code>median</code>
function that does not exist, and wrote “After applying this fix, the test passes successfully.” It never saw a file:
it could only produce plausible text.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A model alone can only write text. Seeing files and taking
actions are the harness's job.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“A language model with no tools can check whether a test passes.”",
             "answer": "False.", "why": "It can only write text; nothing runs the test unless a harness does.",
             "key": {"value": False}},
            {"kind": "mc", "q": "With no tools, what did Qwen3 4B do with the failing test?",
             "options": ["Fixed the real bug in mean()", "Said it could not see the files",
                         "Invented a median() fix and claimed the test passes", "Asked for the file contents"],
             "answer": "C.", "why": "It produced plausible text about code it had never seen.", "key": {"choice": 2}},
        ],
    },
    {
        "title": "The harness loop",
        "segment": (37, 70),
        "figures": [{"t": 50.0, "caption": "Describe a tool; when the model writes a tool call, run it and paste the result "
                                           "back; repeat until the model answers in plain words."},
                    {"t": 69.0, "caption": "The tiny harness: 3 model calls, the test passes, but the file was never read "
                                           "and the docstring is gone."}],
        "body": [
            """<p>The trick behind every agent: give the model a <b>tool description</b> (a name, what it does, and its
parameters). The model can then answer with a <b>tool call</b>, structured text such as
<code>{"command": "python3 test_stats.py"}</code>, instead of prose. The harness runs the call, appends the result to
the conversation, and asks the model again. When the model answers in plain words, the loop ends.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p><code>tiny_harness.py</code> does exactly this in 45 lines with one tool, <code>shell</code>. With Qwen3
4B it took 3 model calls: run the test (exit 1, <code>AssertionError</code>), rewrite <code>stats.py</code> with
<code>echo</code>, answer in words. The harness's own check confirms the test passes. But the model never read the file
before overwriting it, and the docstring was lost: a harness makes actions possible, not careful.</p>""",
            """<div class="box key"><b class="t">Key idea</b>An agent is a loop: the model asks for a tool, the harness
runs it and feeds back the result, until the model answers in words.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put one turn of the harness loop in order.",
             "answer": "the model writes a tool call → the harness runs it → the result is added to the conversation → "
                       "the model is asked again.",
             "why": "The harness, not the model, executes the call and decides what goes back into the conversation.",
             "key": {"items": ["the model writes a tool call", "the harness runs it",
                               "the result is added to the conversation", "the model is asked again"]}},
            {"kind": "number", "q": "How many model calls did the tiny harness need to fix the bug?",
             "answer": "3.", "why": "Run the test, rewrite stats.py, then answer in words.",
             "key": {"parts": [{"label": None, "value": 3, "tol": 0.5, "unit": None}]}},
            {"kind": "short", "q": "The test passed, yet the fix was careless. What did the model skip, and what did it "
                                   "cost?",
             "answer": "It never read stats.py before rewriting it, so it replaced the whole file and the docstring was "
                       "lost.", "why": "Reading before editing is a habit the harness and its instructions must encourage."},
        ],
    },
    {
        "title": "A real harness: Codex CLI",
        "segment": (70, 97),
        "figures": [{"t": 85.0, "caption": "Codex's first request: 8,282 tokens around a 147-character task, including nine "
                                           "tool definitions."},
                    {"t": 96.5, "caption": "The harness owns six things; the model owns one."}],
        "body": [
            """<p>Codex CLI is an open-source harness (Apache-2.0). Traced on the same task, its first request to the model
held 8,282 tokens (counted with a Qwen tokenizer):</p>""",
            """<table><tr><th>part</th><th>tokens</th></tr>
<tr><td>developer instructions</td><td>3,553</td></tr>
<tr><td>skills list</td><td>522</td></tr>
<tr><td>environment context + your task (147 characters = 38 tokens)</td><td>128</td></tr>
<tr><td>9 tool definitions</td><td>4,079</td></tr></table>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The nine tools: <code>exec_command</code>, <code>write_stdin</code>, <code>request_user_input</code>,
<code>view_image</code>, <code>multi_agent_v1</code>, three goal tools and <code>web_search</code>. The harness owns the
loop, the tools, the sandbox and approvals, the instructions, the context window and the record of what happened. The
model owns one thing: choosing the next words.</p>""",
            """<div class="box key"><b class="t">Key idea</b>What you type is a tiny part of what the model reads: a real
harness surrounds it with thousands of tokens of instructions and tool definitions.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What share of Codex's first request (8,282 tokens) is tool definitions (4,079 tokens)?",
             "answer": "About 49%.", "why": "4,079 / 8,282 ≈ 0.49.",
             "key": {"parts": [{"label": None, "value": 49, "tol": 1, "unit": "%"}]}},
            {"kind": "number", "q": "Your task was 38 of those 8,282 tokens. What share is that, in percent?",
             "answer": "About 0.46%.", "why": "38 / 8,282 ≈ 0.0046.",
             "key": {"parts": [{"label": None, "value": 0.46, "tol": 0.02, "unit": "%"}]}},
            {"kind": "mc", "q": "Which of these does the model itself do?",
             "options": ["Run the shell command", "Choose the next words", "Decide what enters the context window",
                         "Enforce the sandbox"],
             "answer": "B.", "why": "Everything else is the harness's job.", "key": {"choice": 1}},
        ],
    },
    {
        "title": "Text becomes an action only if the harness recognizes it",
        "segment": (97, 124),
        "figures": [{"t": 116.0, "caption": "The model server dropped Codex's instructions (3,779 of 7,854 tokens seen); "
                                            "restored, the model wrote its patch as plain text and nothing ran."},
                    {"t": 123.0, "caption": "The core of every harness: while the model asks for a tool, run it and add "
                                            "the result."}],
        "body": [
            """<p>Details matter. Ollama's Responses API silently dropped Codex's <code>developer</code> messages: the model
received 3,779 of the 7,854 tokens Codex sent (48%), with none of the instructions. With the messages relabelled as
<code>system</code> so the model saw them, Qwen3 4B did worse on this task: it wrote its fix as a patch in plain text
(<code>*** Begin Patch</code> …) instead of calling a tool. No tool call means nothing for the harness to run, so the turn
simply ended, and the test still failed.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Only text the harness recognizes as a tool call becomes an
action; everything else is just words.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The model saw 3,779 of the 7,854 tokens Codex sent. What share, in percent?",
             "answer": "About 48%.", "why": "3,779 / 7,854 ≈ 0.48.",
             "key": {"parts": [{"label": None, "value": 48, "tol": 1, "unit": "%"}]}},
            {"kind": "short", "q": "The model's reply contained a correct-looking patch, but the file never changed. Why?",
             "answer": "The patch was plain text in its message, not a tool call, so the harness had nothing to run.",
             "why": "Harnesses act only on structured tool calls."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Write the core loop of a harness.",
             "code": """while True:
    reply = model(history, tools)
    calls = tool_calls(reply)
    if not calls:
        break                      # plain words: done
    for call in calls:
        history.append(run(call))  # the result goes back in""",
             "answer": "tiny_harness.py in the episode's code is this loop with a real model call and a shell tool.",
             "why": "Everything else in a harness (sandboxes, approvals, context management) wraps this loop."},
        ],
    },
]
