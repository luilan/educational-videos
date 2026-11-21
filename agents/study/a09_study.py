"""Study guide content for How AI Agents Work: Inside the Harness, episode 9: Instructions and Memory.

Build:  python framework/study_guide.py agents a09 --video agents/media/videos/a09_scene/1080p60/InstructionsMemoryVideo.mp4
Every number comes from code/a09_instructions_and_memory (2026-11-20/21): instructions_harness.py with Qwen3 4B on Ollama
(9 runs in five setups) and a Codex CLI 0.163.0 request captured through a logging proxy. Codex code: openai/codex @ 89461ea.
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 9",
    "title": "Instructions and Memory",
    "tagline": "How a project tells the agent its own rules",
    "duration": "2:27",
    "intro": """<p>An agent starts every session knowing nothing about your project's conventions; its memory of them is
a file, <b>AGENTS.md</b>. This lesson shows how Codex CLI finds and sends it, measures what happened when a small model
was given three project rules in different ways, and ends with the harness's other tool: <b>hooks</b> that enforce
what matters instead of hoping the model remembers.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–8. Code:
<code>code/a09_instructions_and_memory</code> (<code>instructions_harness.py</code> needs a local Ollama with
<code>qwen3:4b</code>; run it in a container, it executes the model's commands).</div>""",
}

CONCEPTS = [
    {
        "title": "How Codex loads AGENTS.md",
        "segment": (7.7, 59.5),
        "figures": [{"t": 33.1, "caption": "Codex reads every AGENTS.md from the project root down to the working folder, "
                                           "at most 32 KiB, and none for untrusted projects."},
                    {"t": 48.7, "caption": "A captured Codex request: base instructions and skills, then AGENTS.md as a "
                                           "user message, then the task."},
                    {"t": 59.2, "caption": "Our AGENTS.md: three rules."}],
        "body": [
            """<p>Codex looks for <code>AGENTS.md</code> (or a local <code>AGENTS.override.md</code>) in every folder
from the project root, found by <code>.git</code>, down to the working folder, and joins them in that order, up to
<code>project_doc_max_bytes</code> (32 KiB). For projects you have not trusted it loads none.</p>""",
            "{fig0}",
            """<p>It sends the result as a <b>user message</b> before your task: <code># AGENTS.md instructions for
/work</code>, the text inside <code>&lt;INSTRUCTIONS&gt;</code> tags, plus an <code>&lt;environment_context&gt;</code>.
In the captured request this message (742 characters) follows two developer messages, Codex's base instructions
(16,979 characters) and the skills list (2,316).</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>AGENTS.md is the project's standing memory: plain text the
harness puts in front of the model at the start of every session.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "A repo has AGENTS.md at its root and in <code>service/</code>; Codex runs in "
                                "<code>service/api/</code>. Which files are used?",
             "options": ["Only service/AGENTS.md", "Both, root first", "Both, service first", "None"],
             "answer": "B.", "why": "Every AGENTS.md from the root down to the working folder, in that order.",
             "key": {"choice": 1}},
            {"kind": "tf", "q": "“Codex sends AGENTS.md as part of its own system (developer) instructions.”",
             "answer": "False.", "why": "It is a user message before the task.", "key": {"value": False}},
        ],
    },
    {
        "title": "Do the rules get followed?",
        "segment": (60.0, 108.7),
        "figures": [{"t": 95.3, "caption": "Nine runs, four setups: run_tests.py never used; CHANGELOG lines only when sent "
                                           "the Codex way, always in the wrong place."},
                    {"t": 108.4, "caption": "Asked directly, the same model answers with the rule: knowing is not "
                                            "following."}],
        "body": [
            """<p>The same bug fix ran in four setups: no AGENTS.md; the file only in the folder; sent the Codex way;
in the system instructions (plus the hooks runs below, also sent the Codex way). Sending it cost 107 tokens per call.
Across 9 runs the model never used <code>run_tests.py</code>. A CHANGELOG line appeared in 3 runs, all with the file
sent the Codex way, always appended with <code>echo … &gt;&gt; CHANGELOG.md</code> after <code>## 1.3.0</code>, never
under <code>## Unreleased</code>. From the system instructions: neither rule in either run. Left in the folder, the
file was never opened. Fixes themselves, when they worked, were one-line <code>sed</code> edits, as in the run without
any AGENTS.md.</p>""",
            "{fig0}",
            """<p>Yet Codex, asked <i>“What command runs the tests in this project?”</i>, got from the same model
<code>python3 run_tests.py</code>, quoting AGENTS.md. Reading a rule and applying it at the right moment in a
multi-step task are different abilities.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Instructions make the rules available, not followed; a
small model may know them and still not apply them.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The first request was 288 tokens with AGENTS.md and 181 without. How many tokens "
                                    "does sending it add per call?",
             "answer": "107.", "why": "288 − 181 = 107, sent again on every call.",
             "key": {"parts": [{"label": None, "value": 107, "tol": 0.5, "unit": None}]}},
            {"kind": "number", "q": "In how many of the 9 runs did the model run <code>run_tests.py</code>?",
             "answer": "0.", "why": "Not once, in any setup.",
             "key": {"parts": [{"label": None, "value": 0, "tol": 0.5, "unit": None}]}},
            {"kind": "short", "q": "Why did every CHANGELOG line end up in the wrong section?",
             "answer": "The model used echo … >> CHANGELOG.md, which appends to the end of the file, after ## 1.3.0.",
             "why": "Putting it under Unreleased needs an edit in the middle of the file."},
        ],
    },
    {
        "title": "Enforcing what matters: hooks",
        "segment": (109.2, 138.2),
        "figures": [{"t": 132.7, "caption": "Codex hooks, and our Stop check: it blocked once, the model then wrote the "
                                            "line, but the check missed its placement."},
                    {"t": 137.9, "caption": "Sending AGENTS.md: one user message built from a file."}],
        "body": [
            """<p>A harness can enforce rules instead of hoping. Codex has 14 hook events; a <code>Stop</code> hook
can refuse to end a turn with a reason, and a <code>PreToolUse</code> hook can block a command, which the model then
reads as <code>Command blocked by PreToolUse hook: {reason}</code>. Our <code>hooks</code> mode added a Stop check: no
CHANGELOG change, or failing <code>run_tests.py</code>, and the model is told why and must continue.</p>""",
            "{fig0}",
            """<p>In run 3 it fired once (<i>no line was added to CHANGELOG.md under "Unreleased"</i>) and the model
then added the line. But the check only tested that the file had changed, so run 4's misplaced line passed it. The
lesson cuts both ways: enforcement works, and it is only as good as what it tests.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Put critical rules in checks the harness runs, and make
those checks test exactly the rule.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which hook could stop the model from calling <code>python3 test_invoice.py</code> "
                                "directly?",
             "options": ["SessionStart", "PreToolUse", "PostCompact", "UserPromptSubmit"],
             "answer": "B.", "why": "PreToolUse runs before a tool call and can block it with a reason.",
             "key": {"choice": 1}},
            {"kind": "tf", "q": "“Our Stop check would have caught a CHANGELOG line in the wrong section.”",
             "answer": "False.", "why": "It only checked that CHANGELOG.md had changed.", "key": {"value": False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Send a project's AGENTS.md the way Codex does.",
             "code": """def agents_md_message(folder, text):
    return {"role": "user",
            "content": f"# AGENTS.md instructions for {folder}\\n\\n"
                       f"<INSTRUCTIONS>\\n{text}\\n</INSTRUCTIONS>"}

history = [agents_md_message(work, (work / "AGENTS.md").read_text()),
           {"role": "user", "content": TASK}]""",
             "answer": "instructions_harness.py does this in injected and hooks mode.",
             "why": "The rules travel with every call, before the task."},
        ],
    },
]
