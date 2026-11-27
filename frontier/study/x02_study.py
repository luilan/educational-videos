"""Study guide content for The Frontier: Self-Improvement and Memory, episode 2: Weight-Space Programming.

Build:  python framework/study_guide.py frontier x02
Every number comes from code/x02_weight_space_programming (wsp.py, 2026-11-27): Claude (Opus, headless, no tools)
repairs episode 1's addition model after three kinds of damage; five runs in all.
"""

LESSON = {
    "series": "The Frontier: Self-Improvement and Memory",
    "label": "Episode 2",
    "title": "Weight-Space Programming",
    "tagline": "A large model reads a network's weights like code, and fixes them",
    "duration": "2:27",
    "intro": """<p><b>Weight-space programming</b> (WSP) means a language model reading the raw weights of a
network and editing them directly, like code, with no gradients. In November 2026 a stream of preprints reported
frontier models doing this to small networks. This lesson tries it on our own tiny addition model. We damage it in three
ways and let Claude repair it through a simple loop: read, propose, test, keep. Two damages are undone exactly; the third
is undone in two of three runs. The lesson ends with what such repairs do and do not show.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 1 of this series (the model and its weight
file); How LLMs Work episode 3 (embeddings); Deep Dive episode 14 (LayerNorm). Code:
<code>code/x02_weight_space_programming</code> (Python with PyTorch, plus the Claude Code CLI as the editor; each run
takes under two minutes).</div>""",
}

CONCEPTS = [
    {
        "title": "Weight-space programming, and our loop",
        "segment": (7.5, 42.6),
        "figures": [{"t": 21.4, "caption": "Preprints from November 2026 on weight-space programming: repairing, "
                                           "auditing and improving networks by editing their weights directly."},
                    {"t": 42.3, "caption": "The loop: break the model in secret; Claude reads the architecture, 300 "
                                           "graded answers and the weights; it proposes edits; we test them and keep "
                                           "only what helps. No gradients; the test set stays hidden."}],
        "body": [
            """<p>In November 2026 several preprints described frontier language models reading a small network's
weights as text and editing them: repairing a quantized network (arXiv 2611.33031), auditing clinical networks
(2611.05341), updating predictors from experimental feedback (2611.07319), and a sceptical study of whether the models
really reason about weights (2611.08070). The common idea is to treat a weight file like source code.</p>""",
            "{fig0}",
            """<p>Our version uses episode 1's addition model (99.2% on new questions). We damage it in secret. The
editor, Claude (Opus, run headless with no tools), gets three things: the architecture, the damaged model's answers to
<b>300 practice questions</b> with the wrong ones listed, and the weights as text (small tables in full, big matrices as
per-row norms). It replies with edits in a fixed format: set a number, scale a row, swap two rows, or replace a row. Our
harness applies them and re-scores the practice questions, and keeps an edit <b>only if the practice score rises</b>.
There are up to 4 rounds. No gradients are computed, and the 1,000 test questions are never shown; they judge the
final model.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Weight-space programming is a propose-test-keep loop where
the proposals come from reading the weights and the failures, not from gradients.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why does the harness score edits on practice questions but report the final result on "
                                "separate test questions?",
             "options": ["Test questions are harder", "So that the final score is not inflated by choosing edits on "
                         "the same questions", "To save time", "Because Claude asked for it"],
             "answer": "B.", "why": "Keeping edits that raise the practice score selects for luck on those questions "
                                    "(episode 1); unseen questions give an honest score.",
             "key": {"choice": 1}},
            {"kind": "tf", "q": "“In this loop, Claude computes gradients of the loss to decide its edits.”",
             "answer": "False.", "why": "It only reads numbers and graded answers as text and proposes edits; no "
                                        "gradient is computed anywhere.",
             "key": {"value": False}},
            {"kind": "short", "q": "Why show the big matrices only as per-row norms?",
             "answer": "They hold about 99,000 numbers, too many for a prompt; row norms still reveal rows that were "
                       "scaled or zeroed.",
             "why": "The text view trades detail for size; small tables are shown in full."},
        ],
    },
    {
        "title": "Two repairs undone exactly",
        "segment": (43.1, 79.1),
        "figures": [{"t": 61.6, "caption": "Damage 1: one LayerNorm gain set to 0 (test 78%). Claude: it is exactly "
                                           "0 while the others sit between 0.71 and 0.88; it sets 0.79. One number, 8 "
                                           "seconds, test 99.1%."},
                    {"t": 78.8, "caption": "Damage 2: the embeddings of 3 and 8 swapped (test 39%). Claude reads the "
                                           "errors, swaps the rows back: every weight identical to the original."}],
        "body": [
            """<p><b>Damage 1</b> zeroes <code>blocks.0.ln1.weight[13]</code> (it was 0.81; test accuracy 77.7%).
Claude's first reply: this gain “is exactly 0.000. Every other gain in that LayerNorm sits between about 0.71 and 0.88,
so this one was almost certainly zeroed.” It sets it to 0.79, typical of its neighbours. One number, 8 seconds;
practice 83.3% → 99.7%, test 99.1% (the intact model: 99.2%). The damage stood out in the weights themselves.</p>""",
            "{fig0}",
            """<p><b>Damage 2</b> swaps the embedding rows of the digits 3 and 8 (test 39.3%). Nothing looks odd in
the numbers, so the clue is in the failures: Claude concludes that “the model reads every input 3 as an 8 and every 8
as a 3” and swaps the two rows back. That changes 128 numbers and restores <b>every weight exactly</b>: test 99.2%,
like the original. Here the diagnosis came from the graded answers, not from statistics of the weights.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Damage can be found from odd values in the weights or from
the pattern of wrong answers; the best repairs here were tiny and exact.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many percentage points did the one-number repair recover on the test set "
                                    "(77.7% → 99.1%)?",
             "answer": "21.4 points.", "why": "99.1 − 77.7.",
             "key": {"parts": [{"label": None, "value": 21.4, "tol": 0.05, "unit": "points"}]}},
            {"kind": "number", "q": "Each embedding row has 64 numbers. How many numbers does swapping two rows "
                                    "change?",
             "answer": "128.", "why": "2 × 64.",
             "key": {"parts": [{"label": None, "value": 128, "tol": 0, "unit": None}]}},
            {"kind": "short", "q": "Why couldn't the swap be found from the weight statistics alone?",
             "answer": "Swapping two rows keeps every value and every row norm the same, so nothing looks unusual; "
                       "only the behaviour (3 read as 8 and vice versa) reveals it.",
             "why": "Some damage is invisible in the numbers and visible only in the outputs."},
        ],
    },
    {
        "title": "The hard case, and different runs",
        "segment": (79.6, 116.2),
        "figures": [{"t": 99.6, "caption": "Damage 3, first run: the sign of 7's embedding flipped (practice 59%). The "
                                           "diagnosis is right, but the edits miss; round 4 chases a red herring."},
                    {"t": 115.9, "caption": "Two fresh runs fix it in one round with tok.weight[7] × −1. One reply's "
                                            "reason: if digits are encoded periodically, a sign flip adds 5, so 7 "
                                            "reads as 2."}],
        "body": [
            """<p><b>Damage 3</b> multiplies the embedding of the digit 7 by −1 (test 58.7%). In the first run Claude
diagnosed it correctly from the errors: “an input digit 7 were read as 2: 70+26=46, 71+39=60…”. But its edits missed.
It scaled the row by 0.6 (no change; it then noted that LayerNorm cancels a uniform scale), zeroed one large entry
(+0.3 points, kept), set that entry to +2.516 (no change), and in round 4 halved an unrelated MLP neuron with a large
norm, a red herring. After 4 rounds: practice 59.3%, test 61.2%.</p>""",
            "{fig0}",
            """<p>The same damage, run twice more from scratch, was fixed <b>in one round both times</b> by scaling the
row by −1: test 99.2%, every weight identical to the original. One reply explained the clue: if the embeddings encode
digits periodically (mod 10), flipping the sign acts like adding 5, and 7 + 5 = 12 ends in 2. That is the model's
hypothesis, not something we verified, but it fits every error it listed. The lesson: <b>one run is not a
result</b>. The same input can lead to a clean fix or to a wrong path.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>A correct diagnosis does not guarantee a correct repair,
and repeated runs can disagree; report how many runs succeeded.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "If a sign flip adds 5 (mod 10), which digit is a flipped 3 read as?",
             "answer": "8.", "why": "3 + 5 = 8.",
             "key": {"parts": [{"label": None, "value": 8, "tol": 0, "unit": None}]}},
            {"kind": "tf", "q": "“Scaling a token's embedding row by 0.6 has almost no effect in this model because "
                                "the first operation on it is a LayerNorm.”",
             "answer": "True (almost).", "why": "LayerNorm divides by the vector's spread, so a uniform scale mostly "
                                                "cancels; the position embedding added before it keeps the cancellation "
                                                "from being exact.",
             "key": {"value": True}},
            {"kind": "short", "q": "Across all five runs, what fraction of repairs succeeded, and why does it matter to "
                                   "say so?",
             "answer": "4 of 5 runs (both single-run damages and 2 of 3 flip runs); a single run could have shown "
                       "either success or failure, so only the count gives an honest picture.",
             "why": "Language-model editors are not deterministic."},
        ],
    },
    {
        "title": "What repairs do and don't show",
        "segment": (116.7, 139.8),
        "figures": [{"t": 131.9, "caption": "The scoreboard: one zero fixed (1 number), swap fixed (128), sign flip "
                                            "fixed in 2 of 3 runs (64). Repairing is not teaching something new."},
                    {"t": 139.5, "caption": "The loop in code: describe the weights, ask for edits, apply, test, keep "
                                            "only what helps."}],
        "body": [
            """<p>In our runs a frontier model found and undid <b>structural damage</b>: a zeroed gain, swapped rows, a
flipped sign. The edits were tiny (1 to 128 numbers out of 102,413) and, when they worked, exact. Papers this month
describe the same pattern; one summarizes earlier experiments: frontier models “can fix structural bugs by modifying
as little as 0.016% of parameters, but cannot instill new capabilities” (arXiv 2611.04268).</p>
<p>Two cautions. First, repeated runs differ, as damage 3 showed. Second, every case here was a <b>repair</b>: the
right weights existed and had been disturbed in a simple way. Improving a model beyond what training gave it, the claim
of the self-rewriting papers, is a different and harder question. The next episode asks how much of a WSP loop's
success comes from the reasoning, and how much from the loop itself, by running the same loop with random edits.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Undoing simple damage is evidence that weights can be read
and edited like code; it is not yet evidence that a model can improve itself.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What percentage of the model's 102,413 numbers did the swap repair change? "
                                    "(Two decimal places.)",
             "answer": "0.12%.", "why": "128 / 102,413 = 0.00125.",
             "key": {"parts": [{"label": None, "value": 0.12, "tol": 0.01, "unit": "%"}]}},
            {"kind": "mc", "q": "Which result would be the strongest evidence of self-improvement?",
             "options": ["Undoing a zeroed weight", "Swapping two rows back",
                         "Raising accuracy on unseen questions above the original trained model",
                         "Writing a good diagnosis"],
             "answer": "C.", "why": "Repairs restore what training already produced; beating the original on unseen "
                                    "data would show real improvement.",
             "key": {"choice": 2}},
            {"kind": "code", "q": "<b>Try it yourself.</b> The propose-test-keep loop.",
             "code": """for round in range(4):
    reply = ask(describe(weights) + graded(practice))
    edited = apply(weights, reply["edits"])
    if score(edited, practice) > score(weights, practice):
        weights = edited""",
             "answer": "wsp.py implements ask (the Claude CLI with a JSON schema), describe, apply and the scores, "
                       "and logs every round to runs/.",
             "why": "The editor proposes; only the practice score decides what stays."},
        ],
    },
]
