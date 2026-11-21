"""Study guide content for How AI Agents Work: Inside the Harness, episode 13: Evaluating Agents.

Build:  python framework/study_guide.py agents a13 --video agents/media/videos/a13_scene/1080p60/EvaluatingAgentsVideo.mp4
Every number comes from code/a13_evaluating_agents (2026-11-21): eval.py, four tasks × two attempts with Qwen3 4B on
Ollama (results_2026-11-21.json), and episode 4's runs.
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 13",
    "title": "Evaluating Agents",
    "tagline": "How to tell if a change made the agent better",
    "duration": "2:16",
    "intro": """<p>An agent's behaviour varies from run to run, so a single run says little. An <b>evaluation</b> is a fixed
set of tasks, a <b>grader</b> for each, and several <b>attempts</b> per task. This lesson builds a small suite from four
earlier tasks, explains <b>pass@1</b> and <b>pass^k</b>, and shows what variance and cost mean for comparing changes.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–12 (the tasks come from episodes 2, 4, 5 and
6). Code: <code>code/a13_evaluating_agents</code> (<code>eval.py</code> needs a local Ollama with
<code>qwen3:4b</code>).</div>""",
}

CONCEPTS = [
    {
        "title": "Tasks, graders, attempts",
        "segment": (7.7, 56.4),
        "figures": [{"t": 29.7, "caption": "An evaluation: a fixed set of tasks, a grader for each, several attempts per "
                                           "task."},
                    {"t": 44.0, "caption": "Our suite: four tasks from episodes 2, 5, 4 and 6, run with episode 2's harness."},
                    {"t": 56.1, "caption": "Graders check outcomes: tests pass and the test file is unchanged; answers "
                                           "contain the facts."}],
        "body": [
            """<p>The same model on the same task behaved differently from run to run throughout this series (three
search runs, three different rewrite disasters, a right answer by accident). One run is an anecdote. An evaluation
fixes the tasks, grades each attempt automatically, and repeats.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Graders check the <b>outcome</b>, not the transcript. For the bug fixes the test must pass
<b>and</b> the test file must be unchanged, otherwise an agent could “pass” by editing the test. For the questions the
final answer must contain the facts (<code>study_guide.py</code> and <code>42</code>; <code>1.4.2</code>).</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Evaluate outcomes with automatic graders, and guard against
the cheap ways to fake them.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why does the fix grader also check that the test file is unchanged?",
             "answer": "Because an agent could make the test pass by weakening or deleting it instead of fixing the code.",
             "why": "The grader must measure the intended outcome, not just a green test."},
            {"kind": "tf", "q": "“A grader should read the transcript to decide if the agent did well.”",
             "answer": "False.", "why": "Ours check the files and the final answer.", "key": {"value": False}},
        ],
    },
    {
        "title": "Scores: pass@1 and pass^k",
        "segment": (56.9, 95.4),
        "figures": [{"t": 67.1, "caption": "Eight attempts: server 2/2, the two fixes 1/2 each, search 0/2; overall 0.50."},
                    {"t": 83.3, "caption": "pass@1 = 0.50; pass^2 = only the server task."},
                    {"t": 95.1, "caption": "The invoice fix twice: FAIL after 4 calls and 7,754 tokens, PASS after 6 calls "
                                           "and 11,053."}],
        "body": [
            """<p>Two attempts per task: <code>server</code> passed both, <code>stats</code> and <code>fix</code> passed
once each, <code>search</code> failed both. <b>pass@1</b>, the share of attempts that succeed, is 4/8 = 0.50.
<b>pass^k</b> asks whether all k attempts succeeded: only <code>server</code>. If users need the agent to work every
time, pass^k is the number that matters.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Variance is large: the same invoice fix failed after 4 calls and 7,754 tokens, then passed after 6
calls and 11,053 tokens.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Report both how often an agent succeeds and how reliably;
they can tell very different stories.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A task passes 3 of 5 attempts. What is its pass@1?",
             "answer": "0.6.", "why": "3 / 5.", "key": {"parts": [{"label": None, "value": 0.6, "tol": 0.01, "unit": None}]}},
            {"kind": "tf", "q": "“That task has pass^5 = true.”",
             "answer": "False.", "why": "Two attempts failed.", "key": {"value": False}},
            {"kind": "number", "q": "If each attempt independently succeeds with probability 0.5, what is the chance that "
                                    "2 attempts both succeed?",
             "answer": "0.25.", "why": "0.5 × 0.5.", "key": {"parts": [{"label": None, "value": 0.25, "tol": 0.01, "unit": None}]}},
        ],
    },
    {
        "title": "Cost, and how to use an eval",
        "segment": (95.9, 127.7),
        "figures": [{"t": 107.4, "caption": "8 attempts: 2 h 47 min on our CPU, 29,864 tokens; 2 per task is too few to "
                                            "beat luck."},
                    {"t": 120.1, "caption": "Change one thing, rerun the same suite, compare (episode 4: 0/3 → 2/2)."},
                    {"t": 127.4, "caption": "Graders in code."}],
        "body": [
            """<p>The eight attempts took 10,003 seconds (2 h 47 min) and 29,864 tokens on our CPU. With results this
noisy, two attempts per task cannot tell a real improvement from luck; real suites use many more tasks and
attempts.</p>""",
            "{fig0}",
            """<p>The method is simple: change one thing (instructions, a tool, the model), rerun the same suite and
compare. Episode 4 did this informally: search-first instructions got 0 of 3 right; adding “read around matches,
follow values back” got 2 of 2. Codex can be driven the same way with <code>codex exec --json</code>, which runs
non-interactively and prints events as JSON lines.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>An eval turns “it seems better” into a number you can
compare, if you can afford enough attempts.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The 8 attempts took 10,003 seconds. What was the average per attempt, in minutes, "
                                    "to one decimal?",
             "answer": "About 20.8.", "why": "10,003 / 8 ≈ 1,250 s ≈ 20.8 min.",
             "key": {"parts": [{"label": None, "value": 20.8, "tol": 0.1, "unit": "min"}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Write a grader that requires the tests to pass and the test "
                                  "file to be unchanged.",
             "code": """def tests_pass_unchanged(test_file):
    def grade(work, answer):
        same = (work / test_file).read_text() == original(test_file)
        ok = subprocess.run(["python3", test_file], cwd=work).returncode == 0
        return ok and same, ("" if same else "test file was modified")
    return grade""",
             "answer": "eval.py uses it for the stats and fix tasks.",
             "why": "Return a reason with the verdict, so failures can be read later."},
        ],
    },
]
