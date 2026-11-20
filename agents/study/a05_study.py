"""Study guide content for How AI Agents Work: Inside the Harness, episode 5: Editing Files.

Build:  python framework/study_guide.py agents a05 --video agents/media/videos/a05_scene/1080p60/EditingFilesVideo.mp4
Every number comes from code/a05_editing_files (2026-11-20): edit_harness.py with Qwen3 4B on Ollama (3 rewrite runs,
3 patch runs, 2 read-first runs) and its apply_patch() on the patches the model drafted. Codex code: openai/codex @ 89461ea.
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 5",
    "title": "Editing Files",
    "tagline": "How an agent changes one line without rewriting the whole file",
    "duration": "2:21",
    "intro": """<p>A one-line bug in a 108-line file, and two ways for an agent to fix it. <b>Rewriting</b> the whole file
costs the whole file in output and gives the model a hundred chances to copy something wrong; in our runs it destroyed
or corrupted the file every time. A <b>patch</b> shows only the lines that change, with a little context to say where;
the harness applies it only if that context matches the file, and otherwise returns an error the model can read. This
lesson covers Codex CLI's patch format, how Codex <b>intercepts</b> it, and how the matching works.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–4. Code: <code>code/a05_editing_files</code>
(<code>edit_harness.py</code> needs a local Ollama with <code>qwen3:4b</code>; run it in a container, it executes the
model's commands).</div>""",
}

CONCEPTS = [
    {
        "title": "Editing by rewriting the whole file",
        "segment": (7.7, 55.7),
        "figures": [{"t": 31.3, "caption": "Rewrite run 1: the model never read the file and replaced its 108 lines with "
                                           "12, most of them invented."},
                    {"t": 45.4, "caption": "Rewrite run 3: after reading the file it retyped it four times and broke it "
                                           "(dataclass decorators gone, a class definition mangled)."},
                    {"t": 55.4, "caption": "Characters to write for one fix: 3,755 for the whole file, 77 for the "
                                           "changed line."}],
        "body": [
            """<p>The simplest edit tool is the shell itself: <code>cat &gt; FILE &lt;&lt;'EOF' … EOF</code> replaces
the file with whatever the model types. In the first run the model never read <code>invoice.py</code>; it overwrote it
eleven times with short inventions, leaving 12 lines where there were 108 (104 removed, 8 added).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In the third run it did read the file, then typed all of it back (4,334 characters, then 4,052 three
times, until the stuck detector stopped it). The result lost both <code>@dataclass</code> decorators and turned
<code>class Item:</code> into <code>dataclass Item:</code>, a syntax error: the file no longer ran.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>A rewrite costs the whole file in output tokens on every
edit, and every line it copies is a chance to copy it wrong.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Rewrite run 1 removed 104 lines and added 8 in a 108-line file. How many lines "
                                    "did the file have afterwards?",
             "answer": "12.", "why": "108 − 104 + 8 = 12.",
             "key": {"parts": [{"label": None, "value": 12, "tol": 0.5, "unit": None}]}},
            {"kind": "tf", "q": "“If the model reads the file before rewriting it, a rewrite is safe.”",
             "answer": "False.", "why": "Run 3 read the file and still corrupted it while retyping it.",
             "key": {"value": False}},
            {"kind": "number", "q": "The whole file is 3,755 characters; the changed line is 77. About how many times "
                                    "more must a rewrite write?",
             "answer": "About 49.", "why": "3,755 / 77 ≈ 48.8.",
             "key": {"parts": [{"label": None, "value": 48.8, "tol": 1, "unit": None}]}},
        ],
    },
    {
        "title": "Patches, and how Codex applies them",
        "segment": (56.2, 82.6),
        "figures": [{"t": 68.6, "caption": "Codex's patch format: the file to update, @@ context, − removed and + added "
                                           "lines."},
                    {"t": 82.3, "caption": "The model runs apply_patch as a command; the harness catches it before the "
                                           "shell and applies it itself."}],
        "body": [
            """<p>A patch says which file to change and shows only the lines that change: lines starting with
<code>-</code> are removed, <code>+</code> lines are added, and unchanged lines (space-prefixed, or an
<code>@@ def total(self):</code> header) say where. The whole thing sits between <code>*** Begin Patch</code> and
<code>*** End Patch</code>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Codex gives GPT models a dedicated <code>apply_patch</code> tool. For other models, such as Qwen3 4B
(fallback metadata, <code>apply_patch_tool_type: None</code>), the model runs <code>apply_patch</code> through its
shell tool and Codex <b>intercepts</b> the command before it reaches the shell
(<code>intercept_apply_patch</code>), parses and verifies it, and applies it itself. Our harness does the same with
any command that starts with <code>apply_patch</code>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A patch carries only the change plus enough context to place
it; the harness, not the shell, applies it.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "In Codex's patch format, what does a line starting with <code>+</code> mean?",
             "options": ["A comment", "A line to add", "A line to keep", "A new file"],
             "answer": "B.", "why": "<code>-</code> removes, <code>+</code> adds, a space keeps (context).",
             "key": {"choice": 1}},
            {"kind": "short", "q": "Why does the harness intercept <code>apply_patch</code> instead of letting the shell "
                                   "run it?",
             "answer": "So it can parse and check the patch and apply it itself, returning a clear success or error.",
             "why": "There is no apply_patch program in the shell; the harness is the tool."},
        ],
    },
    {
        "title": "Matching the context, or refusing",
        "segment": (83.1, 112.1),
        "figures": [{"t": 97.9, "caption": "Finding the patch's unchanged lines: exact, then looser comparisons; if "
                                           "nothing matches, the edit is refused."},
                    {"t": 111.8, "caption": "Five patch runs, file never damaged; three drafted a patch for a line that "
                                            "does not exist, which apply_patch refuses."}],
        "body": [
            """<p>The old lines in a patch must be found in the file. Codex (<code>seek_sequence.rs</code>) tries an
exact match, then ignores spaces at line ends, then spaces around the whole line, then normalises typographic
punctuation (curly quotes, dashes) to ASCII. If no pass matches, nothing is written and the error
<code>apply_patch failed: could not find these lines …</code> becomes the tool's result.</p>""",
            "{fig0}",
            """<p>In our five patch and read-first runs the model never sent a working patch, but it never damaged the
file either. Three times it drafted a patch for <code>return sum(self.items) + 10.0</code>, a line it had never seen
and that does not exist; our <code>apply_patch()</code> refuses it.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Context lines make a patch self-checking: an edit based on
a wrong picture of the file fails loudly instead of corrupting it.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put Codex's matching passes in order, strictest first.",
             "answer": "exact → ignore trailing spaces → ignore spaces at both ends → normalise punctuation.",
             "why": "Each pass is looser than the one before.",
             "key": {"items": ["exact", "ignore trailing spaces", "ignore spaces at both ends", "normalise punctuation"]}},
            {"kind": "tf", "q": "“When a patch's context is not found, the harness writes the + lines at the end of "
                                "the file.”",
             "answer": "False.", "why": "Nothing is written; the error goes back to the model.", "key": {"value": False}},
            {"kind": "mc", "q": "What did a patch for an invented line do to <code>invoice.py</code>?",
             "options": ["Deleted it", "Nothing: the patch was refused", "Appended the line", "Rewrote the file"],
             "answer": "B.", "why": "Its old line could not be found, so apply_patch failed without writing.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "The right patch",
        "segment": (112.6, 133.8),
        "figures": [{"t": 125.6, "caption": "The fix from patch run 2, sent as a patch: 277 characters, one line changed, "
                                            "test passes."},
                    {"t": 133.5, "caption": "Applying a patch is a search from strict to loose; no match means refuse."}],
        "body": [
            """<p>In patch run 2 the model read the file and worked out the right fix in its reasoning (remove
<code>+ self.discount()</code> from <code>total()</code>), but the reply hit our token cap before the tool call. Sent
as a patch, that fix is 277 characters (the file is 3,755); <code>apply_patch()</code> changes one line and the test
passes.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Good edits are small: find the old lines, replace them, and
say so clearly when they are not there.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The patch is 277 characters, the file 3,755. What percentage of the file does the "
                                    "patch's size represent, to one decimal?",
             "answer": "7.4%.", "why": "277 / 3,755 ≈ 0.0738.",
             "key": {"parts": [{"label": None, "value": 7.4, "tol": 0.1, "unit": "%"}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Find a patch's old lines in a file, strictest comparison first.",
             "code": """def seek(lines, pattern, start):
    for norm in (lambda s: s, str.rstrip, str.strip):
        for i in range(start, len(lines) - len(pattern) + 1):
            if all(norm(lines[i + j]) == norm(p)
                   for j, p in enumerate(pattern)):
                return i
    return None""",
             "answer": "edit_harness.py's apply_patch() uses it for every chunk of a patch.",
             "why": "Returning None lets the caller refuse the edit with a readable error."},
        ],
    },
]
