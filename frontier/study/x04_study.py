"""Study guide content for The Frontier: Self-Improvement and Memory, episode 4: A Model That Edits Itself.

Build:  python framework/study_guide.py frontier x04
Every number comes from code/x04_a_model_that_edits_itself (self_edit.py, 2026-11-28) and the Self-WSP abstract
(arXiv 2611.28833).
"""

LESSON = {
    "series": "The Frontier: Self-Improvement and Memory",
    "label": "Episode 4",
    "title": "A Model That Edits Itself",
    "tagline": "The editor and the edited become one network",
    "duration": "2:25",
    "intro": """<p>In episode 2 a large model edited a small one. Now the editor and the edited are the <b>same
network</b>: a small language model reads its own weights, proposes an edit, and continues as the edited model, so
after every accepted edit it is a slightly different editor. A November 2026 preprint, Self-WSP, reported large gains
this way. We ran a small-scale version on our CPU with the same model, and compared it with the random baseline from
episode 3. At our scale, self-editing raised the score on the practice problems but not on held-out ones, and was no
better than random edits. We also made a mistake on the way, and the lesson shows that too.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 2 and 3 of this series (the edit loop and the
random baseline); How LLMs Work on normalization layers. Code: <code>code/x04_a_model_that_edits_itself</code> (Python
with PyTorch and transformers ≥ 5.2; about 1.5 hours per run on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Self-editing, and how we tested it",
        "segment": (7.7, 58.9),
        "figures": [{"t": 27.6, "caption": "Self-WSP (arXiv 2611.28833, 21 Nov 2026, preprint): Qwen3.5-0.8B edits its "
                                           "own weights; GSM8K 11.8% → 62.2%, sometimes changing only 16 parameters."},
                    {"t": 41.9, "caption": "Our version: the same model on a CPU, adding three 3-digit numbers: 23% of "
                                           "48 practice problems, 26% of 200 held-out problems."},
                    {"t": 58.7, "caption": "Each round it sees its own norm weights and wrong answers, writes 4 edits; "
                                           "the best is kept if it helps, and the edited model writes the next round."}],
        "body": [
            """<p><b>Self-WSP</b> (Ling, Zhong, Zhang and Huang, arXiv 2611.28833, 21 November 2026, a preprint) asks
whether a language model can improve itself by weight-space programming. Qwen3.5-0.8B “reads its own weights,
proposes an edit, and then continues with the edited weights as both problem solver and editor.” The abstract reports
GSM8K (school maths) rising from 11.8% to 62.2% and MATH500 from 7.0% to 32.2%, “sometimes by changing only 16
parameters”.</p>""",
            "{fig0}",
            """<p>We used the same model, Qwen3.5-0.8B (752 million parameters), on our CPU, on a task where answers can
be checked: adding three 3-digit numbers. At the start it got 23% (11 of 48) of the <b>practice</b> problems right
and 25.5% of 200 <b>held-out</b> problems. Practice problems are used to accept edits; held-out problems are only
used to judge.</p>""",
            "{fig1}",
            """<p>Each round the model is shown a summary of its own <b>RMSNorm vectors</b> (49 vectors of 1,024 numbers;
each normalization layer multiplies channel i by 1 + w[i]) and its wrong practice answers, and writes 4 edits as
JSON: scale a whole vector by a factor, or set one entry. We score each edit on the practice problems and keep the
best one if it beats the current score. The edited model then writes the next round's edits. 25 rounds, 100
proposals.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>In self-editing the editor changes with every accepted
edit; held-out problems, never used to accept edits, are the only fair judge.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A norm entry w = 0.25. By what factor does the layer multiply that channel?",
             "answer": "1.25.", "why": "Qwen3.5's RMSNorm multiplies by 1 + w.",
             "key": {"parts": [{"label": None, "value": 1.25, "tol": 0, "unit": None}]}},
            {"kind": "tf", "q": "“Held-out problems may be used to decide which edits to keep, as long as they are not "
                                "shown to the model.”",
             "answer": "False.", "why": "Any problem used to choose edits is no longer a fair test: the choice fits "
                                        "it.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "What it wrote, and what it achieved",
        "segment": (59.4, 93.3),
        "figures": [{"t": 71.0, "caption": "95 of 100 replies were valid edits, mostly scaling a whole vector up by "
                                           "10–50%; 28 of 95 on the final norm."},
                    {"t": 80.3, "caption": "Three edits kept: practice 22.9% → 35.4%, held-out 25.5% → 26.0%."},
                    {"t": 93.1, "caption": "One kept edit set final-norm entry 114 to −31.6: the channel's multiplier "
                                           "went from 4.48 to −30.6, kept because one more practice problem was right."}],
        "body": [
            """<p>The model could do the mechanics: 95 of its 100 replies were valid edits (the others were malformed,
or “No update needed”). But they were alike. 76 scaled a whole vector, 51 of them by a factor between 1.1 and 1.5;
88% of its factors were above 1. 28 of the 95 touched the final normalization layer.</p>""",
            "{fig0}",
            """<p>Three edits were kept: layer 0's post-attention norm × 1.1, layer 6's × 1.5, and one entry of the
final norm set to −31.6. The <b>practice</b> score rose from 22.9% to <b>35.4%</b>. The <b>held-out</b> score went
from 25.5% to <b>26.0%</b>.</p>""",
            "{fig1}",
            """<p>The third edit shows what the loop rewards. Entry 114 of the final norm was 3.48, a multiplier of
4.48; the model set it to −31.6, a multiplier of −30.6: that channel flipped sign and became 6.8 times larger. The
entry was not among the values it had been shown. It was kept because one more practice problem came out right
(+2.1 points).</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Edits kept for a small practice gain can be arbitrary;
only the held-out score says whether the model really improved.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The practice set has 48 problems. How many percentage points is one problem worth "
                                    "(1 decimal)?",
             "answer": "2.1.", "why": "100 / 48 ≈ 2.08.",
             "key": {"parts": [{"label": None, "value": 2.1, "tol": 0.05, "unit": None}]}},
            {"kind": "number", "q": "The practice score rose by 12.5 points. How many more practice problems is that?",
             "answer": "6.", "why": "From 11 to 17 of 48.",
             "key": {"parts": [{"label": None, "value": 6, "tol": 0, "unit": None}]}},
        ],
    },
    {
        "title": "Against random edits, and our mistake",
        "segment": (93.8, 117.1),
        "figures": [{"t": 107.7, "caption": "The same loop with 100 random edits: practice 31.2%, held-out 27.5%. "
                                            "Proposals that beat the current model: 3 of 100 (self), 5 of 100 (random)."},
                    {"t": 116.9, "caption": "Our first run described w as a gain; re-run with the correct description: "
                                            "the same picture."}],
        "body": [
            """<p>The random baseline from episode 3 used the same loop and the same edit language, with 100 random
edits. It ended at 31.2% on practice and <b>27.5%</b> on the held-out problems. Of the model's 100 proposals, 3
beat the model it already was; of the random ones, 5. With 200 held-out problems the score has a standard error of
about 3 points, so neither run's held-out change (+0.5 and +2.0) is distinguishable from noise.</p>""",
            "{fig0}",
            """<p><b>Our mistake.</b> In the first run our prompt called the stored numbers “gains”, but Qwen3.5's
normalization multiplies by 1 + w. That misled the model about its own weights (the random editor reads no
description, so it was unaffected). We stopped that run after 17 rounds (practice 31.2%, held-out 26.0%) and re-ran
with the correct description. The result above is the re-run, and the picture did not change.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>At our scale, the model's self-edits were no better than
random edits in the same loop.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why does a held-out set of 200 problems have about ±3 points of noise at 26%?",
             "answer": "The standard error of a proportion is √(p(1 − p)/n) = √(0.26 × 0.74 / 200) ≈ 0.031.",
             "why": "Differences smaller than that can come from which problems happened to be chosen."},
            {"kind": "mc", "q": "Why was the random baseline unaffected by our description mistake?",
             "options": ["It used a different model", "It never reads the description", "It had more rounds",
                         "It only edited the final layer"],
             "answer": "B.", "why": "Random edits ignore the prompt; only the model's proposals depend on it.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "What this shows",
        "segment": (117.6, 137.6),
        "figures": [{"t": 130.7, "caption": "At our scale: a search the practice set could fool. The Self-WSP abstract "
                                            "itself says random search sometimes performs comparably."},
                    {"t": 137.4, "caption": "In code: the model writes the edit, we apply it, score it, and keep it only "
                                            "if the practice score goes up."}],
        "body": [
            """<p>At our scale (one kind of edit, 100 proposals, 48 practice problems), self-editing behaved like a
search that the practice set could fool, no better than dice. We did not reproduce the preprint's large gains, and
our setup is much smaller than theirs. But the preprint's own abstract points the same way: “WSP driven by random
search sometimes performs comparably”, and “the model shows little understanding of the edits it makes”.</p>
<p>The next episode tests the more ambitious version: a model that first diagnoses its own mistakes, then edits the
weights behind them, and is judged on problems it never saw.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A self-improvement claim needs a held-out test and a random
baseline under the same loop; without them, practice gains can be noise.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Name two changes to our setup that would make a real self-improvement effect easier "
                                   "to detect.",
             "answer": "For example: a larger practice set (less noise when accepting edits), a larger held-out set, "
                       "more rounds, several runs with different seeds, richer edit types.",
             "why": "Noise in the accept step and in the final test hides small effects."},
            {"kind": "code", "q": "<b>Try it yourself.</b> The self-edit loop.",
             "code": """edit = parse(model.generate(describe(model) + wrong_answers))   # the model edits itself
undo = apply(model, edit)
if score(model, practice) > best:
    best = score(model, practice)          # keep: the new model is the next editor
else:
    undo()""",
             "answer": "self_edit.py runs it (self) and the random baseline (random); try 96 practice problems.",
             "why": "The same propose-test-keep loop as episodes 2 and 3, with the model as its own editor."},
        ],
    },
]
