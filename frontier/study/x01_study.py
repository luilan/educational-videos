"""Study guide content for The Frontier: Self-Improvement and Memory, episode 1: The Frozen Model.

Build:  python framework/study_guide.py frontier x01
Every number comes from code/x01_the_frozen_model (frozen.py + recheck.py, 2026-11-27).
"""

LESSON = {
    "series": "The Frontier: Self-Improvement and Memory",
    "label": "Episode 1",
    "title": "The Frozen Model",
    "tagline": "A trained model is a file of numbers. What if it edited them itself?",
    "duration": "2:20",
    "intro": """<p>In November 2026 two preprints reported language models that <b>edit their own weights</b>
and get better, with no training run. This series rebuilds such experiments at a size a CPU can run, and tests them
sceptically. This first lesson looks at what is being edited. A trained model is a file of numbers. We change one by
hand, then every one in turn, and see how few of them matter on their own. We also see how easy it is to mistake
noise for an improvement.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episodes 3 (embeddings), 9
(transformer blocks) and 12 (building a tiny GPT); Deep Dive episode 14 (LayerNorm). Code:
<code>code/x01_the_frozen_model</code> (Python with PyTorch; the full scan takes about 15 minutes on 12 cores).</div>""",
}

CONCEPTS = [
    {
        "title": "A fourth way to improve a model",
        "segment": (7.5, 43.4),
        "figures": [{"t": 25.2, "caption": "Two preprints from November 2026: a 27B model rewrites its own weights "
                                           "(+6.6 points on unseen problems); five days later a 0.8B model does it too."},
                    {"t": 43.1, "caption": "The usual ways to improve a frozen model, and the new one: edit the "
                                           "weights directly, by reasoning about them."}],
        "body": [
            """<p>On 16 November 2026, <i>Language Models Can Rewrite Their Own Weights</i> (arXiv 2611.21885)
reported an open 27-billion-parameter model that was given its own weights, its graded answers and the ability to run
itself. It diagnosed why it failed and rewrote the weights behind the failure. There were no gradients and no training
run. Its edit made it <b>6.6 points</b> more accurate on problems it had never seen. On 21 November,
<i>Self-WSP</i> (arXiv 2611.28833) reported the same kind of self-editing in Qwen3.5-0.8B. Both are single preprints
that nobody has replicated yet.</p>""",
            "{fig0}",
            """<p>Why is this new? A trained model is <b>frozen</b>. The usual ways to improve it are to train it again,
which is a whole new run, or to change what happens around it: a better prompt, a harness, more time to think. Those
last three help only while the task runs; nothing about the model itself changes. The papers describe a fourth way:
the model, or another model, changes the weights <b>directly, by reasoning about them</b>, and the change stays.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Prompts, harnesses and thinking time improve one task;
editing the weights changes the model itself, and the change persists.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which of these improvements persists after the task ends?",
             "options": ["A better system prompt", "More reasoning tokens", "An edit to the weights",
                         "A retry loop in the harness"],
             "answer": "C.", "why": "The others change what surrounds a frozen model; only a weight edit changes the "
                                    "model.",
             "key": {"choice": 2}},
            {"kind": "tf", "q": "“The self-rewriting result has been confirmed by several independent labs.”",
             "answer": "False.", "why": "As of this video it is a single preprint (16 Nov 2026) with no published "
                                        "replication.",
             "key": {"value": False}},
            {"kind": "short", "q": "What makes editing the weights by reasoning different from training them?",
             "answer": "Training changes the weights with gradients computed from a loss on many examples; here a "
                       "model reads the weights, decides on a change and writes it, with no gradients or training run.",
             "why": "The edit comes from reasoning about the numbers, not from backpropagation."},
        ],
    },
    {
        "title": "Weights are a file of numbers",
        "segment": (43.9, 72.6),
        "figures": [{"t": 59.6, "caption": "The tiny addition model (99.2% right on new questions): one 418,615-byte "
                                           "file holding 102,413 numbers in 30 tables."},
                    {"t": 72.3, "caption": "Setting one number (blocks.0.ln1.weight[13], 0.81) to zero: 99.2% → "
                                           "77.7%, and 27 + 15 now comes out as 41."}],
        "body": [
            """<p>The model in this lesson is the tiny GPT from Reinforcement Learning for LLMs (2 transformer blocks,
64-dimensional), trained here on 7,100 two-digit additions until it gets <b>99.2%</b> of 1,000 new questions right.
Frozen, it is one file, <code>model.pt</code>, of 418,615 bytes: <b>102,413 numbers</b> (4 bytes each, plus a little
bookkeeping) in 30 tables. The tables are the embeddings, the attention and MLP weight matrices, their biases, the
normalization layers and the output layer.</p>""",
            "{fig0}",
            """<p>Nothing stops you editing that file. Number 13 of the first block's first LayerNorm scale is 0.81.
Set it to 0 and the model gets only <b>77.7%</b> right; <code>27+15=</code> now gives <code>41.</code>. A LayerNorm
scale multiplies one of the 64 channels of every token's vector, so zeroing it deletes that channel from everything
the first attention layer sees.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>A frozen model is just numbers in a file; changing one can
change what it does.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "About how many bytes do 102,413 numbers take as 32-bit floats? (Round to the "
                                    "nearest thousand.)",
             "answer": "About 410,000 bytes.", "why": "102,413 × 4 = 409,652; the file adds a few kilobytes of names "
                                                      "and shapes (418,615 in total).",
             "key": {"parts": [{"label": None, "value": 410000, "tol": 2000, "unit": "bytes"}]}},
            {"kind": "number", "q": "The edit cut accuracy from 99.2% to 77.7%. How many of the 1,000 test questions "
                                    "did it break, at least?",
             "answer": "215.", "why": "992 − 777 = 215 (more if it also fixed some).",
             "key": {"parts": [{"label": None, "value": 215, "tol": 0, "unit": None}]}},
            {"kind": "short", "q": "Why can zeroing a single LayerNorm scale hurt more than zeroing a single number "
                                   "in a big weight matrix?",
             "answer": "The scale multiplies one channel for every token, so zeroing it removes that channel "
                       "everywhere; one matrix entry is only one of many terms in a sum.",
             "why": "Some numbers act on everything; most act on one connection."},
        ],
    },
    {
        "title": "How many numbers matter on their own",
        "segment": (73.1, 104.9),
        "figures": [{"t": 88.2, "caption": "Every number zeroed in turn, tested on 250 questions: 96,699 change nothing, "
                                           "163 cost a point or more, none costs over 50 points."},
                    {"t": 104.6, "caption": "The numbers that matter are in the embeddings and normalization scales; "
                                            "none of the 99,456 attention and MLP numbers matters on its own."}],
        "body": [
            """<p>Zeroing each of the 102,413 numbers in turn and testing on 250 questions gives a clear picture.
<b>96,699</b> edits (94%) change nothing; 5,163 raise the score (next concept); 388 cost under a point; 150 cost 1–10
points; 13 cost 10–50 points; none costs more than 50. So only <b>163</b> numbers cost a point or more on their own.</p>""",
            "{fig0}",
            """<p>Where are they? 101 are in the embeddings (1,472 numbers), 61 in the normalization scales (320) and 1
in the output layer. <b>None</b> of the 33,280 attention numbers or the 66,176 MLP numbers costs a point when zeroed
alone. What the model knows is spread across thousands of numbers, so a useful edit usually has to change many of
them together, which is why editing weights by hand is hard and why the papers' results are surprising.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Most single numbers barely matter; knowledge is spread out,
and the few sensitive numbers sit where every token passes through.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What percentage of the single-number edits changed nothing? (Round to a whole "
                                    "number.)",
             "answer": "94%.", "why": "96,699 / 102,413 = 0.944.",
             "key": {"parts": [{"label": None, "value": 94, "tol": 1, "unit": "%"}]}},
            {"kind": "number", "q": "How many numbers are in the attention and MLP tables together?",
             "answer": "99,456.", "why": "33,280 + 66,176.",
             "key": {"parts": [{"label": None, "value": 99456, "tol": 0, "unit": None}]}},
            {"kind": "tf", "q": "“Since no single MLP number matters, the MLP layers could be deleted without harm.”",
             "answer": "False.", "why": "Each number matters little alone, but together they do the computation; "
                                        "removing whole layers would break the model.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "A gain you picked is not a gain",
        "segment": (105.4, 130.4),
        "figures": [{"t": 123.5, "caption": "5,163 edits raised the 250-question score, 5,152 of them by one question. "
                                            "The best 20 on the other 750 questions: 4 better, 9 the same, 7 worse."},
                    {"t": 130.1, "caption": "Editing a model by hand: load the weights, change one number, test again."}],
        "body": [
            """<p><b>5,163</b> single edits raised the score on the 250 probe questions. But 5,152 of them did so by
exactly one question, and 11 by two. To find out whether they are real, take the 20 best and test them on the
<b>other 750</b> test questions, which played no part in choosing them: 4 are better than the unedited model, 9 are the
same and 7 are worse. The “improvements” were noise: one borderline question flipping either way.</p>""",
            "{fig0}",
            """<p>This is the most important habit for the whole series. When you try many edits and keep the best by
some score, the winner's score is inflated, because some edits got lucky on exactly those questions. A real gain must
show on questions that were never used to pick the edit. The self-improvement papers report gains on unseen problems
for this reason, and episodes 3 and 7 test whether ours hold up.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Choose edits on one set of questions and judge them on
another; a gain on the set you chose with is not evidence.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "On 250 questions, how many percentage points is one question worth?",
             "answer": "0.4 points.", "why": "1 / 250 = 0.004.",
             "key": {"parts": [{"label": None, "value": 0.4, "tol": 0, "unit": "points"}]}},
            {"kind": "mc", "q": "You try 1,000 random edits and keep the one with the best score on 50 test "
                                "questions. What should you expect on fresh questions?",
             "options": ["The same gain", "A bigger gain", "A smaller gain, possibly none", "Exactly zero change"],
             "answer": "C.", "why": "The winner was picked partly for luck on those 50 questions; that part does not "
                                    "carry over.",
             "key": {"choice": 2}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Zero one number and measure the damage.",
             "code": """state = torch.load("model.pt")
state["blocks.0.ln1.weight"][13] = 0.0
model.load_state_dict(state)
print(accuracy(model, test))          # 0.992 before, 0.777 after""",
             "answer": "frozen.py scans every number this way; recheck.py tests the best edits on held-out "
                       "questions.",
             "why": "A weight edit is just an assignment into a tensor."},
        ],
    },
]
