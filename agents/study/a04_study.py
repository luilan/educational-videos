"""Study guide content for How AI Agents Work: Inside the Harness, episode 4: Reading and Searching Code.

Build:  python framework/study_guide.py agents a04 --video agents/media/videos/a04_scene/1080p60/ReadingSearchingVideo.mp4
Every number comes from code/a04_reading_and_searching (2026-11-20): search_harness.py (Qwen3 4B on Ollama; 3 runs in
naive and search mode, 2 in context mode), count_tokens.py (Qwen3 tokenizer) and one Codex CLI 0.163.0 run of the same
task (codex_trace_2026-11-20.jsonl). Codex code: openai/codex @ 89461ea.
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 4",
    "title": "Reading and Searching Code",
    "tagline": "How an agent finds the right lines in a project too big to read",
    "duration": "2:43",
    "intro": """<p>This lesson follows one question through three harness configurations: <i>which line decides the
file name of a figure image extracted for the study-guide PDF?</i> The project is 12 files and about 19,000 tokens.
Without guidance the model <b>reads whole files</b> and guesses; with <b>search-first</b> instructions it is cheaper and
grounded but stops at the first plausible match; told to <b>read around matches and follow values back</b>, it finds
the answer. Along the way: why tool output needs a <b>budget</b>, and how Codex CLI handles both.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–3. Code: <code>code/a04_reading_and_searching</code>
(<code>search_harness.py</code> needs a local Ollama with <code>qwen3:4b</code>; run it in a container, it executes the
model's commands).</div>""",
}

CONCEPTS = [
    {
        "title": "Why an agent cannot just read everything",
        "segment": (7.7, 52.1),
        "figures": [{"t": 37.4, "caption": "No guidance: two whole files printed; tokens sent per call grow to 4,996 "
                                           "(6,978 in total)."},
                    {"t": 51.8, "caption": "The naive answers: line numbers made up (108, 177; the real line is 209), "
                                           "or a file that does not exist."}],
        "body": [
            """<p>Everything a tool returns becomes part of the conversation, and the whole conversation is sent to
the model on <b>every</b> call (episode 2). The demo project is 12 Python files, 1,455 lines, 19,072 tokens: reading
it all would make every later call carry all of it.</p>""",
            """<p>With no guidance, two of three runs did <code>ls</code>, <code>ls stepstone</code>, then
<code>cat</code> of two whole files (4,046 and 14,859 characters). Tokens sent per call: 191, 228, 268, 1,295, 4,996:
6,978 in total. Both answered with the right format string from the <b>wrong file</b>
(<code>stepstone/build_packs.py</code>, which names StepStone's pack images the same way).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Their line numbers were invented (108 and 177; the line is 209): <code>cat</code> prints no line
numbers, so the model counted, badly. The third run opened nothing at all and named a file that does not exist.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Reading whole files is expensive, and it does not even
give the model line numbers; without them, it guesses.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A naive run sent 191, 228, 268, 1,295 and 4,996 tokens on its five calls. What was "
                                    "the total?",
             "answer": "6,978.", "why": "191 + 228 + 268 + 1,295 + 4,996 = 6,978.",
             "key": {"parts": [{"label": None, "value": 6978, "tol": 0.5, "unit": None}]}},
            {"kind": "tf", "q": "“Once a file has been printed, the harness sends it to the model only once.”",
             "answer": "False.", "why": "It stays in the conversation, which is sent again on every call.",
             "key": {"value": False}},
            {"kind": "mc", "q": "Why were the naive runs' line numbers wrong?",
             "options": ["The file changed between calls", "cat prints no line numbers, so the model guessed",
                         "The harness truncated the file", "Line numbers start at zero"],
             "answer": "B.", "why": "Without numbers in the text it read, the model had to count lines itself.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "Search first, read only what you need",
        "segment": (52.6, 90.7),
        "figures": [{"t": 79.1, "caption": "Search mode: grep returns matching lines with file and line number; about "
                                           "2,000 tokens per run instead of 6,978."},
                    {"t": 90.4, "caption": "All three search runs answered line 224, where the name is used; it is "
                                           "made on line 42, just below the match on line 41."}],
        "body": [
            """<p>Two sentences of instructions changed the behaviour: <i>search first with <code>grep -rn</code>,
then read only the lines you need with <code>sed -n 'START,ENDp' FILE</code>; never print whole files.</i>
<code>grep -rn</code> prints <code>file:line:text</code> for every match; <code>sed -n</code> prints a range.</p>""",
            """<p>All three runs searched for <code>figure</code> and read one line with <code>sed</code>. Tokens sent in
total: 1,799, 2,314 and 1,910, under a third of the naive runs that read files, and every line number they quoted was
real. But all three answered <code>framework/study_guide.py</code>, <b>line 224</b>, which <i>uses</i> the name. The
line that <i>makes</i> it is 42; their own search had matched line 41, just above it.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Searching makes reading cheap and grounded in real line
numbers, but a search only shows what matched: the agent has to look around.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The three search runs sent 1,799, 2,314 and 1,910 tokens. What is their average, "
                                    "to the nearest token?",
             "answer": "2,008.", "why": "(1,799 + 2,314 + 1,910) / 3 = 6,023 / 3 ≈ 2,008.",
             "key": {"parts": [{"label": None, "value": 2008, "tol": 1, "unit": None}]}},
            {"kind": "short", "q": "<code>grep -rn \"figure\" .</code> printed line 41 of study_guide.py but not line 42. "
                                   "Why not?",
             "answer": "Line 42 does not contain the word “figure”; grep shows only lines that match the pattern.",
             "why": "Line 42 is <code>fig[\"file\"] = f\"img/...\"</code>."},
            {"kind": "order", "q": "Put the search-first steps in order.",
             "answer": "grep -rn for candidate lines → choose a match → sed -n to print the lines around it → answer.",
             "why": "Search narrows the project to a few lines; reading a range confirms them.",
             "key": {"items": ["grep -rn for candidate lines", "choose a match", "sed -n to print the lines around it",
                               "answer"]}},
        ],
    },
    {
        "title": "Read around matches, follow values back",
        "segment": (91.2, 108.9),
        "figures": [{"t": 108.6, "caption": "Context mode: both runs answered line 42, one by searching again for "
                                            "fig[\"file\"], one from the context lines alone."}],
        "body": [
            """<p>Two more sentences: <i>use <code>grep -rn -C 3</code> to see a few lines around each match; when you
find where a value is used, search again for where it is set.</i> In the first context run the model searched
<code>study-guide</code>, then <code>figure</code> with context, then <code>fig["file"]</code>, and answered
<code>framework/study_guide.py</code>, line 42, with the exact format string
<code>img/{vid}_{fig['t']:07.2f}.jpg</code>: the right answer, for 3,046 tokens. The second run did not need the
second search: <code>grep -rn -C 3 'figure' *</code> showed lines 38–44, assignment included, and it answered line 42
for 1,932 tokens. Two of two right, against zero of three in search mode.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Good search is two moves: find candidates, then follow the
thing you care about to where it is defined.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which extra instruction made the model find line 42?",
             "options": ["Print every file in full", "Search again for where a value is set",
                         "Use a bigger model", "Answer after the first match"],
             "answer": "B.", "why": "In run 1 the second search, for fig[\"file\"], found the assignment on line 42.",
             "key": {"choice": 1}},
            {"kind": "tf", "q": "“grep -C 3 prints three lines before and after each match.”",
             "answer": "True.", "why": "-C N adds N lines of context on both sides.", "key": {"value": True}},
        ],
    },
    {
        "title": "Output budgets, and how Codex does it",
        "segment": (109.4, 153.6),
        "figures": [{"t": 128.8, "caption": "A 5,343-character result, capped at 2,000: the first and last 1,000 are "
                                            "kept, with a note of what was cut."},
                    {"t": 146.4, "caption": "Codex's instructions prefer rg; with Qwen3 4B, Codex still listed folders "
                                            "and printed a whole file."}],
        "body": [
            """<p>A search can also return too much: with context lines, one <code>grep</code> returned 5,343
characters. The harness caps every result (ours at 2,000 characters): it keeps the beginning and the end, and puts
<code>…3,343 chars truncated…</code> in the middle, so the model knows something is missing. Codex does the same with
tokens: <code>exec_command</code>'s <code>max_output_tokens</code> defaults to 10,000, and longer output becomes head,
<code>…N tokens truncated…</code>, tail (<code>utils/string/src/truncate.rs</code>).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Codex's instructions say: <i>“When searching for text or files, prefer using <code>rg</code> or
<code>rg --files</code> respectively because <code>rg</code> is much faster than alternatives like
<code>grep</code>.”</i> With Qwen3 4B on the same task, Codex ran <code>ls /work</code>, <code>ls
/work/stepstone</code> and <code>cat /work/stepstone/build_packs.py</code>, sending 7,852, 9,999 and 10,572 tokens on
its first three calls, and did not finish within our 50-minute limit. Instructions help only as much as the model
follows them.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Cap every tool result, keep its head and tail, and say what
was cut; and remember instructions are requests, not guarantees.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A result of 5,343 characters is capped at 2,000 (first 1,000 + last 1,000). How "
                                    "many characters are cut?",
             "answer": "3,343.", "why": "5,343 − 2,000 = 3,343.",
             "key": {"parts": [{"label": None, "value": 3343, "tol": 0.5, "unit": None}]}},
            {"kind": "mc", "q": "Why keep the end of a long output, not just the beginning?",
             "options": ["It is shorter", "Errors, summaries and exit messages usually come last",
                         "The model reads backwards", "To save tokens"],
             "answer": "B.", "why": "A failing test or compiler prints its verdict at the end.", "key": {"choice": 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Cap a tool result, keeping the head and the tail.",
             "code": """LIMIT = 2000   # characters kept per tool result

def truncate_middle(text, limit=LIMIT):
    if len(text) <= limit:
        return text
    head = text[:limit // 2]
    tail = text[-(limit - limit // 2):]
    cut = len(text) - limit
    return f"{head}\\n…{cut} chars truncated…\\n{tail}\"""",
             "answer": "search_harness.py applies it to every result in search and context mode.",
             "why": "The marker tells the model that something was cut, and how much."},
        ],
    },
]
