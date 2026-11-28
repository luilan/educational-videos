"""Study guide content for The Frontier: Self-Improvement and Memory, episode 5: Diagnosing Its Own Mistakes.

Build:  python framework/study_guide.py frontier x05
Every number comes from code/x05_diagnosing_its_own_mistakes (self_rewrite.py, confirm.py, 2026-11-28) and the
abstract of arXiv 2611.21885.
"""

LESSON = {
    "series": "The Frontier: Self-Improvement and Memory",
    "label": "Episode 5",
    "title": "Diagnosing Its Own Mistakes",
    "tagline": "First find what goes wrong, then edit the weights behind it",
    "duration": "2:22",
    "intro": """<p>In episode 4 a small model edited its own weights blind and gained nothing that lasted. A second
November 2026 preprint describes a stronger recipe: the model first <b>diagnoses</b> its failure, can <b>run
itself</b> to see which weights matter, and only then rewrites them. We gave the same small model those two steps.
This time the edits held up on 1,000 fresh problems, a real improvement. But the evidence that the model understood
what it was doing is weak, and this lesson shows both halves.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 3 and 4 of this series (random baseline,
self-editing loop, held-out tests). Code: <code>code/x05_diagnosing_its_own_mistakes</code> (Python with PyTorch and
transformers ≥ 5.2; about 2.7 hours for the main run on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Diagnose, probe, then edit",
        "segment": (7.7, 57.5),
        "figures": [{"t": 28.5, "caption": "Self-rewriting (arXiv 2611.21885, 16 Nov 2026, preprint): a 27B model "
                                           "diagnoses its own failure and rewrites the weights behind it: +6.6 points on "
                                           "unseen problems."},
                    {"t": 46.6, "caption": "Our version adds two steps to episode 4's loop: a one-sentence diagnosis, "
                                           "and a probe in which the model is run to see which norm vectors matter."},
                    {"t": 57.3, "caption": "Its diagnosis: a place-value mistake; almost the same sentence in 19 of 25 "
                                           "rounds."}],
        "body": [
            """<p>“Language Models Can Rewrite Their Own Weights” (Fan et al., arXiv 2611.21885, 16 November 2026, a
preprint) reports that “an open 27B model, given its weights, its graded answers and the ability to run itself,
diagnoses its own failure and rewrites the weights behind it”, and becomes 6.6 points more accurate on problems it
never saw. In one model family, the smaller models made no edit or broke themselves.</p>""",
            "{fig0}",
            """<p>We added the same two ingredients to episode 4's loop, with the same small model (Qwen3.5-0.8B), task,
edit language and budget (25 rounds × 4 proposals):</p>
<ol><li><b>Diagnose</b>: the model reads its wrong practice answers and writes one sentence about its mistakes.</li>
<li><b>Probe</b> (“run itself”): for each of its 49 normalization vectors, the harness scales it by 1.1 and
measures how the log-probability of the <i>right</i> answers to the wrong problems changes. The 10 largest effects
are shown to the model.</li>
<li><b>Edit</b>: 4 proposals given the diagnosis and the probe table; the best is kept if the practice score
rises.</li></ol>""",
            "{fig1}",
            """<p>The diagnosis was always some version of “a place-value mistake… adding the last digit of each number
instead of the tens digit… which causes the sum to be off by 100”, almost the same sentence in 19 of 25 rounds:
generic, and the same whatever its current wrong answers were.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Self-rewriting adds two steps before the edit: a diagnosis
of the failures, and measurements from running the model itself.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "What does the probe measure, in one sentence?",
             "answer": "For each normalization vector, how much scaling it by 1.1 changes the log-probability of the "
                       "right answers to the problems the model got wrong.",
             "why": "It is a cheap, gradient-free sensitivity test of each vector."},
            {"kind": "number", "q": "How many forward passes over the wrong problems does one probe need (one per "
                                    "vector, plus the unchanged model)?",
             "answer": "50.", "why": "49 vectors + 1 baseline.",
             "key": {"parts": [{"label": None, "value": 50, "tol": 0, "unit": None}]}},
        ],
    },
    {
        "title": "A real gain",
        "segment": (58.0, 93.2),
        "figures": [{"t": 67.3, "caption": "Six edits kept: practice 22.9% → 35.4%; held-out (200) 25.5% → 32.0%."},
                    {"t": 80.8, "caption": "On 1,000 fresh problems: 34.8% → 42.4%; 87 problems newly right, 11 newly "
                                           "wrong."},
                    {"t": 93.0, "caption": "Same 1,000 problems: episode 4's blind self-edits 34.1%, random edits 36.1%, "
                                           "the probe alone 35.8%, the full loop 42.4%."}],
        "body": [
            """<p>Six edits were kept. The practice score rose from 22.9% to 35.4%, and this time the held-out score
moved too: from 25.5% to 32.0%.</p>""",
            "{fig0}",
            """<p>Two hundred problems give about ±3 points of noise, so we checked the final model on <b>1,000 fresh
problems</b> never used before (they happen to be easier, 34.8% for the original). The edited model scored
<b>42.4%</b>. Problem by problem: 87 that the original missed are now right, and only 11 that it got right are now
wrong. With 87 against 11, chance is ruled out.</p>""",
            "{fig1}",
            """<p>The baselines on the same 1,000 problems: episode 4's blind self-edits 34.1% (30 gained, 37 lost);
random edits 36.1% (35 gained, 22 lost); the probe alone, choosing edits with no language model, 35.8% (it found
one useful edit and then had nothing new to try). Only the full loop improved clearly.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>With a diagnosis and a self-probe, the loop found edits that
generalize to fresh problems, well beyond the baselines.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "87 problems were gained and 11 lost out of 1,000. What is the net change in "
                                    "accuracy, in points?",
             "answer": "7.6.", "why": "(87 − 11) / 1,000 = 7.6%; 34.8% + 7.6 = 42.4%.",
             "key": {"parts": [{"label": None, "value": 7.6, "tol": 0, "unit": None}]}},
            {"kind": "tf", "q": "“Comparing models on the same 1,000 problems is fair even though those problems are "
                                "easier than the 200.”",
             "answer": "True.", "why": "Every model faces the same problems, so their difficulty affects all of them "
                                       "equally.",
             "key": {"value": True}},
        ],
    },
    {
        "title": "Did it understand?",
        "segment": (93.7, 116.5),
        "figures": [{"t": 104.8, "caption": "Only 3 of 84 valid edits picked the probe's top suggestion; the scale "
                                            "direction matched the probe 31 times of 63: a coin flip."},
                    {"t": 116.3, "caption": "The kept edits: two early norm vectors scaled to about a quarter, one × 1.5, "
                                            "one × 1.1. Four vectors, 4,096 numbers."}],
        "body": [
            """<p>A real gain does not mean the model knew what it was doing. Of its 84 valid proposals, 41 targeted a
vector in the probe's top 10, but only 3 chose the top entry. And when it scaled a vector, the direction (up or
down) agreed with the sign its own probe had measured 31 times out of 63, no better than a coin flip. The diagnosis,
as we saw, barely changed.</p>""",
            "{fig0}",
            """<p>What it kept, as net factors on the stored vectors w: layer 1's post-attention norm × 0.27, layer 2's ×
0.25, layer 6's × 1.5 (the same edit episode 4's run kept), and layer 3's input norm × 1.1 (the edit the probe
baseline also kept). Four vectors, 4,096 of the model's 752 million numbers.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>The edits worked, but the model did not follow its own
measurements; working and understanding are separate questions.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Its scale directions agreed with the probe 31 times in 63. What percentage is that "
                                    "(nearest whole number)?",
             "answer": "49%.", "why": "31 / 63 ≈ 0.49, what a coin flip would give.",
             "key": {"parts": [{"label": None, "value": 49, "tol": 1, "unit": "%"}]}},
        ],
    },
    {
        "title": "What this shows",
        "segment": (117.0, 133.8),
        "figures": [{"t": 129.2, "caption": "With a diagnosis and a self-probe the loop found edits that generalize; one "
                                            "run, one small model, one task."},
                    {"t": 133.6, "caption": "In code: probe, diagnose, propose, then the same test and keep."}],
        "body": [
            """<p>Giving the model a diagnosis step and measurements of itself changed the outcome: the loop found edits
that hold up on fresh problems, where blind self-edits and random edits did not. This is one run, with one small
model, on one task; a second run with another seed could look different. It also does not show that the model
understood its edits: its choices barely tracked its own probe. Episode 7 takes the kept edits apart one by one to
see which matter and why.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Measure self-improvement on fresh problems against baselines,
and treat “it worked” and “it understood” as separate claims.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Name one experiment that would test whether the gain depends on the diagnosis step.",
             "answer": "Run the same loop with the probe table but without the diagnosis (or with a fixed, wrong "
                       "diagnosis), several seeds each, and compare on the same fresh problems.",
             "why": "Remove one ingredient at a time, as in episode 3."},
            {"kind": "code", "q": "<b>Try it yourself.</b> One round of self-rewriting.",
             "code": """effects = probe(model, wrong)                  # run itself: which vectors matter
diagnosis = model.generate(wrong_answers)
edit = parse(model.generate(diagnosis + table(effects)))
undo = apply(model, edit)
if score(model, practice) <= best: undo()      # same test and keep""",
             "answer": "self_rewrite.py runs it (diagnose) and the probe-only baseline (probe); confirm.py re-scores "
                       "every final model on 1,000 fresh problems.",
             "why": "Diagnosis and self-measurement are two extra steps in front of the same loop."},
        ],
    },
]
