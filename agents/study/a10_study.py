"""Study guide content for How AI Agents Work: Inside the Harness, episode 10: The Context Budget.

Build:  python framework/study_guide.py agents a10 --video agents/media/videos/a10_scene/1080p60/ContextBudgetVideo.mp4
Every number comes from code/a10_context_budget (2026-11-21): budget_harness.py with Qwen3 4B on Ollama (1 unlimited,
3 compact, 2 capped runs; a simulated 4,096-token window). Codex code: openai/codex @ 89461ea.
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 10",
    "title": "The Context Budget",
    "tagline": "What happens when the conversation no longer fits",
    "duration": "2:18",
    "intro": """<p>Instructions, the task and every tool result must fit in the model's <b>context window</b>. This lesson
gives a harness a small window (4,096 tokens) and a task that reads three files, then tries three answers to running
out of room: nothing (it overflows), <b>compaction</b> with Codex's own handoff-summary prompt, and <b>capping</b>
every tool result. Each one costs something different.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–9, especially 4 (output caps). Code:
<code>code/a10_context_budget</code> (<code>budget_harness.py</code> needs a local Ollama with <code>qwen3:4b</code>;
run it in a container, it executes the model's commands).</div>""",
}

CONCEPTS = [
    {
        "title": "The window, and what overflows it",
        "segment": (7.7, 54.1),
        "figures": [{"t": 30.2, "caption": "Codex's fallback: a 272,000-token window, 95% usable, auto-compaction at 90%."},
                    {"t": 53.8, "caption": "No budget: three whole files in one reply; the next request is 7,887 tokens, "
                                           "almost twice our 4,096 window."}],
        "body": [
            """<p>Everything the model reads must fit: instructions, the task, every tool result so far. For a model it
has no metadata for, Codex assumes a 272,000-token window, treats 95% of it as usable, and compacts automatically at
90% of the window.</p>""",
            "{fig0}",
            """<p>Our harness simulates a 4,096-token window (the server itself allows 32,768, so oversize requests
still run and can be measured). The task asks three questions about three files. Without a budget the model printed
all three files in one reply (14,859, 13,092 and 1,800 characters); its next request was 7,887 tokens. It answered all
three, but only because the server had room.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>The window bounds the whole conversation, not one message;
tool results are usually what fills it.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Our window is 4,096 tokens. At what request size does a Codex-style harness compact "
                                    "(90%), rounded down?",
             "answer": "3,686.", "why": "4,096 × 9 / 10 = 3,686.4.",
             "key": {"parts": [{"label": None, "value": 3686, "tol": 0.5, "unit": None}]}},
            {"kind": "tf", "q": "“The unlimited run answered correctly, so the window was never a problem.”",
             "answer": "False.", "why": "Its 7,887-token request would not fit a real 4,096-token window.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "Compaction",
        "segment": (54.6, 91.1),
        "figures": [{"t": 66.5, "caption": "Codex's compaction prompt: a handoff summary for another model that will "
                                           "resume the task."},
                    {"t": 76.5, "caption": "After compaction: about 1,100 characters of summary; the next call is ~550 "
                                           "tokens; still 3 of 3."},
                    {"t": 90.8, "caption": "The catch: the compaction request itself was 7,835 tokens."}],
        "body": [
            """<p>When a request passes 90% of the window, the harness sends the conversation plus Codex's prompt
(<i>“You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the
task.”</i>) and continues from just the task and the summary, introduced by Codex's summary prefix.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In three runs the summaries kept 1,102–1,181 characters (527–1,727 tokens written), the next request
was 545–595 tokens, and every answer stayed right. But the compaction request itself was 7,835 tokens: to summarize,
the model must first read everything, so with a real 4,096-token window it would have failed. That is why Codex
compacts at 90% of a large window and caps every command's output.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Compaction trades the full history for a summary, and it
needs room to run: compact before the window is full, not after.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Compaction took the conversation from 7,835 tokens to a next request of 545. About "
                                    "what fraction remained, in percent?",
             "answer": "About 7%.", "why": "545 / 7,835 ≈ 0.07.",
             "key": {"parts": [{"label": None, "value": 7, "tol": 0.6, "unit": "%"}]}},
            {"kind": "short", "q": "Why would the compaction request fail with a real 4,096-token window?",
             "answer": "It contains the whole conversation to be summarized (7,835 tokens), more than the window.",
             "why": "The summary is written by the same model, from the full history."},
        ],
    },
    {
        "title": "Capping, and choosing what to keep",
        "segment": (91.6, 130.0),
        "figures": [{"t": 113.1, "caption": "Capped: every result cut to 2,000 characters; requests stay small, but line 59 "
                                            "(AgentsCover.jpg) was in the cut middle."},
                    {"t": 124.1, "caption": "Summaries can lose details, caps cut the middle, search reads only what matters."},
                    {"t": 129.7, "caption": "Compaction in code: one extra model call, then a short conversation."}],
        "body": [
            """<p>With every tool result capped at 2,000 characters (episode 4's head-and-tail cut), the largest request
was 1,850 tokens and no compaction was needed. But <code>"cover": "youtube/covers/AgentsCover.jpg"</code> is line 59
of 301 in <code>stepstone/build_packs.py</code>, inside the part that was cut. Both runs answered <b>cover.jpg</b>,
taken from the file's docstring at the top: 2 of 3 right.</p>""",
            "{fig0}",
            """<p>A budget forces a choice. A summary can lose details; a cap cuts the middle; searching reads only the
lines that matter, and needs neither.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>Every way of saving context drops something; the cheapest
context is the one you never had to read.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why did the capped runs answer cover.jpg?",
             "options": ["The model hallucinated", "The right line was in the cut middle; the top mentioned cover.jpg",
                         "The file was empty", "Compaction removed it"],
             "answer": "B.", "why": "Head and tail survived; line 59 of 301 did not.", "key": {"choice": 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Compact a conversation with a handoff summary.",
             "code": """def compact(history):
    r = respond(history + [{"role": "user", "content": COMPACT_PROMPT}])
    summary = text_of(r).split("</think>")[-1].strip()
    return [{"role": "user", "content": TASK},
            {"role": "user", "content": SUMMARY_PREFIX + summary}]

if used > WINDOW * 9 // 10:
    history = compact(history)""",
             "answer": "budget_harness.py does this in compact and capped mode.",
             "why": "After compaction the model sees only the task and the summary."},
        ],
    },
]
