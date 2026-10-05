"""Study guide content for LLMs in Practice, episode 11: Evaluating LLMs.

Build:  python framework/study_guide.py llms-in-practice p11 --video llms-in-practice/media/videos/p11_scene/1080p60/EvaluationVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every score and answer comes from code/p11_evaluation (Qwen2.5-0.5B/1.5B-Instruct in float32 and 3B-Instruct in
bfloat16, greedy; transformers 4.57.1, torch 2.14.0, CPU). "By hand" = an answer counted only if both the final answer
and its stated reasoning are right.
"""

LESSON = {
    "series": "LLMs in Practice",
    "label": "Episode 11",
    "title": "Evaluating LLMs",
    "tagline": "How we know it's better",
    "duration": "2:37",
    "intro": """<p>This lesson answers one question: after changing a prompt, a model or its precision, how do you
know the result is <b>better</b>? You need an <b>evaluation</b>: questions with known answers and an automatic
<b>check</b>, run after every change. A real test of three models shows the traps: a check that is too strict, a test
that is too easy, and scores that hide wrong reasoning, which you only find by <b>reading the outputs</b>.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 3 (sampling), 5 (RAG) and 10 (quantization)
of this series. Code: <code>code/p11_evaluation</code> (downloads three Qwen2.5 models, about 10 GB; about 15 minutes on
a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "A test you can run again",
        "segment": (8, 36),
        "figures": [{"t": 34.5, "caption": "Questions with known answers, the handbook in the prompt, three models."}],
        "body": [
            """<p>You changed a prompt, swapped a model, quantized it: is it better? <i>“It looked fine on a few
tries”</i> is not an answer. You need an <b>evaluation</b>, a test you can run again and again. Start with questions
that have <b>known answers</b>: here, 20 easy questions about the bakery handbook, with the handbook in the prompt (as
in RAG), and three models: Qwen2.5 0.5B, 1.5B and 3B.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>An evaluation = fixed questions + expected answers + an
automatic check.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why is “I tried three questions and it looked fine” a poor way to compare two "
                                   "prompts?",
             "answer": "Too few, unchosen and unrepeatable examples: differences can be luck, and you cannot rerun it "
                       "after the next change.",
             "why": "A fixed test set gives comparable numbers every time."},
            {"kind": "tf", "q": "“For a fair comparison, both models must get exactly the same questions and the same "
                                "context.”",
             "answer": "True.", "why": "Otherwise you are measuring the difference in inputs, not in models."},
        ],
    },
    {
        "title": "The check decides the score",
        "segment": (36, 66),
        "figures": [{"t": 53.0, "caption": "Exact match: 1 / 20, because full-sentence answers like “The bakery opens at "
                                           "7:30 AM on weekdays.” fail."},
                    {"t": 65.5, "caption": "“Contains the fact”: 20 / 20 for every model. The easy test is "
                                           "saturated."}],
        "body": [
            """<p>The simplest check, <b>exact match</b>, gave the 1.5B model <b>1 / 20</b>. Its answers were right,
just written as sentences: <i>“The bakery opens at 7:30 AM on weekdays.”</i> The test was wrong, not the model.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Checking whether the answer <b>contains</b> the expected fact gives <b>20 / 20</b> to all three models.
That is a problem too: a test everyone passes cannot tell models apart. It is <b>too easy</b> (saturated).</p>""",
            """<div class="box key"><b class="t">Key idea</b>A useful test has a fair check and room to separate good from
better.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Expected answer “monday”; the model says “The bakery is closed on Mondays.” Which check "
                                "marks it correct?",
             "options": ["Exact match only", "Contains the expected word (lower-cased)", "Both", "Neither"],
             "answer": "B.", "why": "“mondays” contains “monday”; exact match compares the whole sentence."},
            {"kind": "short", "q": "All models score 20 / 20. What does that tell you, and what should you do?",
             "answer": "The test is saturated and cannot rank the models; add harder questions.",
             "why": "Scores at the ceiling hide real differences."},
        ],
    },
    {
        "title": "Harder questions, and reading the answers",
        "segment": (66, 107),
        "figures": [{"t": 85.0, "caption": "Ten harder questions: 3, 3 and 6 out of 10."},
                    {"t": 105.8, "caption": "Reading the answers: automatic 3, 3, 6; by hand 2, 1, 5."}],
        "body": [
            """<p>Ten harder questions, each combining two facts (a cake for 12 delivered on Sunday: total? Can I order
a birthday cake on Thursday for Saturday? Open at 8:30 on a Saturday?), separate the models: <b>3, 3 and 6</b> out of
10. The bigger model pulls ahead.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>But <b>read the answers</b>. The 1.5B model said <i>“… making the total cost 42 euros”</i> and still
scored, because “39” appears in its working. The 0.5B model said <i>“No, … closed on Saturdays”</i>: right word, wrong
reason. The 1.5B model said <i>“Yes, … for an order of 25 euros, delivery would not qualify”</i>: right facts, marked
wrong for starting with “yes”. Checked by hand (final answer <i>and</i> its stated reason must be right), the real scores are <b>2, 1 and
5</b>: the 1.5B model's “No” for Monday 8:00, for example, came with the reason “it opens at 7:30 on weekdays”, which
contradicts its own answer.</p>""",
            """<div class="box key"><b class="t">Key idea</b>An automatic score is only as good as its checker. Always read
the failures, and a sample of the passes.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "The first version of this test asked “What is the total for a cake for 8 people "
                                   "delivered on a Tuesday?” and expected <i>31</i> (28 + 3 euros delivery). What was "
                                   "wrong with the test itself?",
             "answer": "A cake for 8 costs 28 euros, below the 30-euro minimum for delivery: the correct answer is that "
                       "it cannot be delivered.",
             "why": "Test sets have bugs too. It was caught by reading the answers carefully, and replaced with a cake "
                    "for 12 on a Tuesday (39 + 3 = 42 euros)."},
            {"kind": "mc", "q": "Which is a <b>false positive</b> of the automatic check?",
             "options": ["The 1.5B “42 euros” answer, scored correct because it contains 39",
                         "The 1.5B “Yes, … would not qualify” answer, scored wrong",
                         "The 3B “31 euros” answer", "The 0.5B “No, closed on Mondays” answer"],
             "answer": "A.", "why": "A false positive is a wrong answer counted as right. B is a false negative."},
            {"kind": "tf", "q": "“Automatic scores of 3, 3 and 6 prove the 3B model is exactly twice as good.”",
             "answer": "False.", "why": "With 10 questions the scores are noisy, and the checker made errors (by hand: 2, "
                                       "1, 5)."},
        ],
    },
    {
        "title": "Better checks, noise, and keeping the test",
        "segment": (107, 157),
        "figures": [{"t": 120.5, "caption": "Ways to make checking more reliable."},
                    {"t": 141.0, "caption": "An evaluation is a loop: ask, check, count, and print the failures."}],
        "body": [
            """<p>Make checking easier: ask for a <b>structured answer</b> (a single number, or yes or no); use
<b>several checks</b>; a stronger model can grade answers against a <b>rubric</b>, but check the grader too; and always
<b>read a sample</b> by hand.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Mind the <b>noise</b>: with 10 questions, one answer is 10%. With sampling, run each question several
times; small differences on small test sets often mean nothing. Keep your test set and run it after <b>every
change</b>: a new prompt, a new model, a quantized version. That is how you know it is better, and not just
different.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Treat your test set like code tests: versioned, rerun on every
change, and read when it fails.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Model X scores 7 / 10 and model Y 6 / 10. How many questions is the difference, and "
                                    "why is it weak evidence?",
             "answer": "One question.", "why": "A single answer can flip by chance (e.g. a near-tie, episode 10); you need "
                                              "more questions or repeated runs."},
            {"kind": "code", "q": "<b>Try it yourself.</b> In <code>code/p11_evaluation/evaluate.py</code>, change the "
                                  "system prompt to ask for structured answers, then rerun and compare the hard-set "
                                  "scores and outputs.",
             "code": """msg = [{"role": "system", "content": "Answer from the handbook below. Reply with only "
                                "a number in euros, or only yes or no.\\n\\n" + HANDBOOK},
       {"role": "user", "content": question}]""",
             "answer": "Your scores will change: answers become short (“39”, “no”), so the checks stop rewarding numbers "
                       "that only appear in the reasoning and stop penalising “yes, but no”. Read the outputs to see "
                       "whether the models also answer more or less correctly without room to reason.",
             "why": "The answer format changes both the model's behaviour and what the checker can judge; that is why "
                    "you rerun the whole test after any change."},
        ],
    },
]
