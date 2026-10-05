"""Study guide content for How LLMs Work: Deep Dive, episode 20: Learning-Rate Warmup and Schedules.

Build:  python framework/study_guide.py deep-dive d20 --video deep-dive/media/videos/d20_scene/1080p60/SchedulesVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d20_lr_schedules/lr_schedules.py (a 4-layer tiny GPT trained 3,000 steps with AdamW;
torch 2.14.0, CPU; seed 1337) or the schedule functions evaluated at the exercise steps.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 20",
    "title": "Learning-Rate Warmup and Schedules",
    "tagline": "How the step size should change over a run",
    "duration": "2:16",
    "intro": """<p>This lesson answers one question: should the learning rate stay fixed during training? The same tiny
GPT is trained for 3,000 steps with five schedules. The constant learning rates end near 1.60; schedules that
<b>decay</b> the learning rate end lower (cosine 1.551, linear 1.554), and <b>warmup-stable-decay</b> (WSD) does best
(1.542), with most of its gain arriving in the short decay at the end.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 16 (warmup and stability) and 19
(AdamW). Code: <code>code/d20_lr_schedules</code> (PyTorch only; five tiny models, about 25 minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Five schedules, and warmup",
        "segment": (8, 52),
        "figures": [{"t": 39.2, "caption": "Five schedules: constant 0.001 and 0.003; cosine, linear and warmup-stable-decay, "
                                           "all peaking at 0.003."},
                    {"t": 51.1, "caption": "Warmup: 200 steps from near zero to the peak."}],
        "body": [
            """<p>Almost no large model keeps its learning rate fixed: it rises at the start and falls toward the end.
The experiment trains the same tiny GPT for 3,000 steps, five times: two <b>constant</b> learning rates (0.001 and 0.003)
and three schedules peaking at 0.003, all ending at 10% of the peak: warmup + <b>cosine</b> decay, warmup + <b>linear</b>
decay, and <b>warmup-stable-decay</b> (WSD): warmup, a long stable phase at the peak, then a short linear decay over the
last 600 steps.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Each schedule starts with <b>warmup</b>: 200 steps rising linearly from near zero to the peak. As
episode 16 showed, the first updates are the riskiest, and keeping them small protects the model.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A schedule is a multiplier of the peak learning rate as a
function of the step: warm up, hold, decay.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With 200 warmup steps and peak 0.003, what is the learning rate at step 99 (the code "
                                    "uses (step + 1) / 200)?",
             "answer": "0.0015.", "why": "100 / 200 of the peak."},
            {"kind": "number", "q": "At step 1,600 (halfway through the decay), what are the cosine, linear and WSD learning "
                                    "rates?",
             "answer": "0.00165, 0.00165 and 0.003.", "why": "Cosine and linear are both at 55% of the peak halfway; WSD is "
                                                            "still in its stable phase until step 2,400."},
        ],
    },
    {
        "title": "Constant vs decaying",
        "segment": (52, 83),
        "figures": [{"t": 67.0, "caption": "Constant learning rates: 0.003 is faster early (1.90 vs 2.00 at step 500); both end "
                                           "near 1.60."},
                    {"t": 82.3, "caption": "Cosine and linear decay pull ahead as the learning rate shrinks: 1.551 and "
                                           "1.554."}],
        "body": [
            """<p>The higher constant learning rate learns faster early: validation loss 1.90 at step 500 against 2.00.
But by the end they nearly meet: <b>1.598</b> and <b>1.605</b>. A fixed step size eventually stops helping.</p>""",
            """<p>The decaying schedules pull ahead as the learning rate shrinks: cosine <b>1.551</b>, linear
<b>1.554</b>. One way to picture it: smaller steps let the model settle into a lower point that big steps keep jumping
over.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A high learning rate makes fast early progress; a decaying one
lets the loss settle lower at the end.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“The constant learning rate of 0.003 ended clearly better than 0.001.”",
             "answer": "False.", "why": "1.598 vs 1.605: it was ahead early (1.90 vs 2.00 at step 500), but they nearly met."},
            {"kind": "number", "q": "What is the cosine schedule's learning rate at step 2,700?",
             "answer": "About 0.000376.", "why": "warm × (0.1 + 0.45 (1 + cos(π · 2,500 / 2,800))) × 0.003, computed with the "
                                                "episode's function."},
        ],
    },
    {
        "title": "Warmup-stable-decay",
        "segment": (83, 119),
        "figures": [{"t": 101.6, "caption": "WSD tracks the constant run (1.606 at step 2,400), then drops to 1.542 in the "
                                            "600-step decay."},
                    {"t": 117.7, "caption": "Final losses: WSD 1.542, cosine 1.551, linear 1.554, constant 1.598 and 1.605."}],
        "body": [
            """<p>For most of the run, WSD stays at the peak, and its loss tracks the constant run: <b>1.606</b> at step
2,400. Then, in the last 600 steps, the learning rate falls to 10% of the peak and the loss drops sharply to
<b>1.542</b>, the best of the five.</p>""",
            """<p>That shape is practical: you can keep training at the peak for as long as you like, and decay only when
you want a finished model. The schedule matters, not just the peak value: about <b>0.06</b> of loss here (1.598 → 1.542),
with the same peak and the same number of steps.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Much of the benefit of decay arrives during the decay itself:
WSD gains 0.064 in its last 600 steps.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Order the five schedules from best to worst final loss: <i>constant 0.003 · cosine · "
                                   "WSD · constant 0.001 · linear</i>.",
             "answer": "WSD (1.542) → cosine (1.551) → linear (1.554) → constant 0.003 (1.598) → constant 0.001 (1.605).",
             "why": "From the episode's run."},
            {"kind": "short", "q": "Why is WSD convenient if you don't know in advance how long you will train?",
             "answer": "Its learning rate does not depend on the total length until the final decay: you can keep training "
                       "at the peak and start the decay whenever you decide to stop.",
             "why": "Cosine and linear schedules need the total number of steps from the start."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Evaluate the episode's WSD function at a few steps.",
             "code": """for s in (0, 99, 1600, 2700, 2999):
    print(s, 3e-3 * SCHEDULES["warmup-stable-decay"](s))""",
             "answer": "0.000015, 0.0015, 0.003, 0.00165, 0.000304.",
             "why": "Warmup to step 199, stable until 2,399, then a linear decay to 10%."},
        ],
    },
]
