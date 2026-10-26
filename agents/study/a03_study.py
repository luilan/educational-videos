"""Study guide content for How AI Agents Work: Inside the Harness, episode 3: Tool Calling Under the Hood.

Build:  python framework/study_guide.py agents a03 --video agents/media/videos/a03_scene/1080p60/ToolCallingVideo.mp4
Every number comes from code/a03_tool_calling (2026-10-26): Qwen3's chat template rendered with its tokenizer
(show_prompt.py), 10 raw samples of Qwen3 4B on Ollama (raw_samples.py) and a Codex CLI 0.163.0 request captured
through a logging proxy (codex_*.json). Codex code: openai/codex @ 89461ea.
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 3",
    "title": "Tool Calling Under the Hood",
    "tagline": "What the model reads, and what it writes, when it calls a tool",
    "duration": "2:18",
    "intro": """<p>This lesson answers one question: since a model only reads and writes text, what is a tool call,
really? The <b>chat template</b> turns tool definitions into instructions the model reads; the model answers with JSON
between <code>&lt;tool_call&gt;</code> tags; the server or harness <b>parses</b> it into a structured call; and the result
goes back as more text. Real numbers from Qwen3 4B and Codex CLI show what each step costs, and what happens when the JSON
is broken.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–2. Code: <code>code/a03_tool_calling</code>
(<code>show_prompt.py</code> needs <code>transformers</code> for the tokenizer only; <code>raw_samples.py</code> needs a
local Ollama with <code>qwen3:4b</code>).</div>""",
}

CONCEPTS = [
    {
        "title": "What the model reads",
        "segment": (8, 33),
        "figures": [{"t": 32.5, "caption": "Qwen3's chat template turns our tool into instructions in the system message; "
                                           "the tool adds 139 tokens (58 → 197)."}],
        "body": [
            """<p>A model has no special channel for tools. The <b>chat template</b> (the recipe that turns a list of
messages into one long text) writes the tool definitions into the system message, inside <code>&lt;tools&gt;</code> tags,
followed by an instruction: <i>to call a function, return a JSON object with its name and arguments within
<code>&lt;tool_call&gt;&lt;/tool_call&gt;</code> tags</i>.</p>""",
            "{fig0}",
            """<p>For our one small <code>shell</code> tool, the first prompt is 197 tokens instead of 58: the tool
definition and its instructions cost 139 tokens, on every call. Switching “thinking” off in Qwen3 is also just text: the
template pre-fills an empty <code>&lt;think&gt;&lt;/think&gt;</code> block.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Tools reach the model as text: definitions and instructions
written into the prompt by the chat template.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With the tool the first prompt is 197 tokens, without it 58. How many tokens does the "
                                    "tool add?",
             "answer": "139.", "why": "197 − 58 = 139.",
             "key": {"parts": [{"label": None, "value": 139, "tol": 0.5, "unit": None}]}},
            {"kind": "tf", "q": "“The model receives tool definitions through a channel separate from its text.”",
             "answer": "False.", "why": "The chat template writes them into the system message as text.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "What the model writes, and how it is parsed",
        "segment": (33, 65),
        "figures": [{"t": 51.0, "caption": "10 raw samples: every one wrote the same valid call (about 25 tokens), after "
                                           "250–500 tokens of reasoning."},
                    {"t": 64.5, "caption": "Parsing: find the tool_call tags, parse the JSON, hand the harness a name, "
                                           "arguments and a call ID."}],
        "body": [
            """<p>Sampled 10 times with no server parsing in between (Ollama raw mode, temperature 0.7), Qwen3 4B wrote
the same valid call every time: <code>{"name": "shell", "arguments": {"command": "python3 test_stats.py"}}</code>, about
25 tokens. Before it, each sample wrote 251–494 tokens in total, nearly all of it its own reasoning, even with
thinking “off”.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Something must find the call in that text: the model server (Ollama does this for its OpenAI-style
APIs) or the harness. It searches for the tags, parses the JSON and returns a <b>structured call</b>: the tool's name,
its arguments (as a JSON string) and a <b>call ID</b> that will tie the result to this call.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A tool call is text in an agreed format; parsing turns it into
a structured call with a name, arguments and an ID.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In the 10 samples, how many produced a valid tool call?",
             "answer": "10.", "why": "All 10 parsed, with identical calls.",
             "key": {"parts": [{"label": None, "value": 10, "tol": 0.5, "unit": None}]}},
            {"kind": "order", "q": "Put the parsing steps in order.",
             "answer": "find the tool_call tags → parse the JSON inside → hand the harness a structured call.",
             "why": "Without the tags there is nothing to parse; without parsing there is no call to run.",
             "key": {"items": ["find the tool_call tags", "parse the JSON inside", "hand the harness a structured call"]}},
            {"kind": "short", "q": "Each sample wrote about 25 tokens of call and up to about 470 tokens of reasoning. Why "
                                   "does that matter for an agent?",
             "answer": "Most of the generated tokens, and most of the time, go to text the harness never uses.",
             "why": "Generation time is proportional to tokens written."},
        ],
    },
    {
        "title": "How the result goes back",
        "segment": (65, 77),
        "figures": [{"t": 76.5, "caption": "The call and the result, as the model reads them on its next call: 43 tokens; "
                                           "the reasoning is not kept."}],
        "body": [
            """<p>On the next call, the template writes the model's earlier turn as just its
<code>&lt;tool_call&gt;</code> block, and the result as a new turn wrapped in <code>&lt;tool_response&gt;</code> tags. The
call and its result add 43 tokens to the conversation (the call's JSON alone is 19). Qwen3's template does not keep the
reasoning that came before the call.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>To the model, a tool result is just more text to read, marked
with tags so it knows where it came from.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“In Qwen3's template, the model's reasoning before a tool call is kept in the "
                                "conversation.”",
             "answer": "False.", "why": "Only the tool call is kept; the reasoning is dropped.", "key": {"value": False}},
        ],
    },
    {
        "title": "Tool calls in Codex",
        "segment": (77, 122),
        "figures": [{"t": 98.0, "caption": "exec_command: 10 parameters, only cmd required; 421 tokens of definition "
                                           "against 58 for our shell tool."},
                    {"t": 110.0, "caption": "A real call and result in Codex, matched by call_id; the result starts with "
                                            "a header."},
                    {"t": 121.5, "caption": "Broken JSON: Codex sends “failed to parse function arguments” back to the "
                                            "model as the result."}],
        "body": [
            """<p>Codex's main tool, <code>exec_command</code>, has 10 parameters; only <code>cmd</code> is required. The
others set the working directory, how long to wait for output (<code>yield_time_ms</code>, default 10 s), an output
budget (<code>max_output_tokens</code>, default 10,000), a request to run outside the sandbox with a justification, a
reusable approval prefix, and shell options. Its definition is 421 tokens; our <code>shell</code> tool's is 58.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In the captured run the model called <code>exec_command</code> with
<code>{"cmd":"ls","workdir":"/work"}</code> and call ID <code>call_2tfwp3m3</code>. Codex's result carried the same ID and
started with a header (chunk ID, wall time, exit code, original token count) before the output. If arguments are not
valid JSON, Codex (<code>tools/handlers/mod.rs</code>) returns <code>failed to parse function arguments: …</code> to the
model as the result, so it can try again.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Real harnesses use richer tool definitions, label results with
the call ID and metadata, and turn errors into results the model can read.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "exec_command's definition is 421 tokens, our shell tool's 58. How many times larger?",
             "answer": "About 7.3.", "why": "421 / 58 ≈ 7.26.",
             "key": {"parts": [{"label": None, "value": 7.3, "tol": 0.1, "unit": None}]}},
            {"kind": "mc", "q": "Why does each result carry the call's ID?",
             "options": ["To encrypt the output", "So the model can match each result to its call",
                         "To count tokens", "Because JSON requires it"],
             "answer": "B.", "why": "With several calls in flight, IDs tell which result answers which call.",
             "key": {"choice": 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Parse a tool call out of a model's text, returning an error the "
                                  "model can read when it fails.",
             "code": """def parse_tool_call(text):
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.S)
    if not m:
        return None, "no tool call"
    try:
        return json.loads(m.group(1)), None
    except json.JSONDecodeError as e:
        return None, f"failed to parse function arguments: {e}\"""",
             "answer": "raw_samples.py uses this function on every sample.",
             "why": "Returning the error, instead of crashing, gives the model a chance to fix its call."},
        ],
    },
]
