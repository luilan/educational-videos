"""Study guide content for How LLMs Work: Deep Dive, episode 23: Scaling Laws.

Build:  python framework/study_guide.py deep-dive d23 --video deep-dive/media/videos/d23_scene/1080p60/ScalingLawsVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d23_scaling_laws/scaling_laws.py (tiny GPTs on Tiny Shakespeare; torch 2.14.0, CPU) or from
the fitted power laws evaluated at the exercise values.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 23",
    "title": "Scaling Laws",
    "tagline": "Why bigger models are a safe bet",
    "duration": "2:25",
    "intro": """<p>This lesson answers one question: why do labs keep training bigger models on more data? Because the
loss falls predictably. Six tiny GPTs on the same data line up on a straight line in log-log axes: a <b>power law</b>,
loss ≈ 4.24 × N<sup>−0.071</sup>. One model trained on doubling amounts of data gives another, loss ≈ 10.68 ×
tokens<sup>−0.121</sup>. The biggest model falls slightly above its line, short of data. Fitted on small runs, such laws
let labs predict the loss of much larger ones.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episode 17 (the loss) and Foundations
(logarithms). Code: <code>code/d23_scaling_laws</code> (PyTorch only; about 45 minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Loss vs model size",
        "segment": (8, 70),
        "figures": [{"t": 37.0, "caption": "Six tiny GPTs, 7,697 to 1,198,273 parameters, the same 4,096,000 tokens: loss "
                                           "2.280 down to 1.612."},
                    {"t": 69.3, "caption": "In log-log axes, a straight line: loss ≈ 4.24 × N^−0.071; the biggest model sits "
                                           "above it."}],
        "body": [
            """<p>Six tiny GPTs are trained on the same 4,096,000 tokens, from 7,697 to 1,198,273 parameters (excluding
embeddings). Validation loss: 2.280, 2.070, 1.879, 1.774, 1.684, 1.612. With logarithmic axes the points lie almost on a
straight line: a <b>power law</b>, loss ≈ <b>4.24 × N<sup>−0.071</sup></b>. Every 10× more parameters multiplies the
loss by about <b>0.85</b> (10<sup>−0.071</sup>).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The biggest model sits a little above the line: <b>1.612</b> measured, 1.577 predicted. A likely reason:
it saw only about 3.4 training tokens per parameter, so it is starting to run short of data.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A power law y = a·x<sup>b</sup> is a straight line in log-log
axes: log y = log a + b · log x.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What loss does the fitted law predict for 10 million parameters?",
             "answer": "About 1.35.", "why": "4.24 × (10⁷)<sup>−0.071</sup> ≈ 1.350 (if the data were sufficient).", "key": {'parts': [{'label': None, 'value': 1.35, 'tol': 0.027, 'unit': None}]}},
            {"kind": "number", "q": "By what factor does the loss shrink for 100 times more parameters?",
             "answer": "About 0.72.", "why": "0.85² ≈ 0.72, or 100<sup>−0.071</sup>.", "key": {'parts': [{'label': None, 'value': 0.72, 'tol': 0.0144, 'unit': None}]}},
        ],
    },
    {
        "title": "Loss vs data",
        "segment": (70, 92),
        "figures": [{"t": 90.4, "caption": "One 453,857-parameter model as its tokens double: 2.406 → 1.586; loss ≈ 10.68 × "
                                           "tokens^−0.121."}],
        "body": [
            """<p>One model (width 96, 4 layers, 453,857 parameters) is measured as its training tokens double from 256,000
to 8,192,000: 2.406, 2.165, 1.987, 1.798, 1.684, 1.586. Another power law: loss ≈ <b>10.68 × tokens<sup>−0.121</sup></b>;
every doubling multiplies the loss by about <b>0.92</b>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>More data lowers the loss predictably too, until the model's
size becomes the limit.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "By the data law, what is the loss at 16,384,000 tokens (one more doubling)?",
             "answer": "About 1.43 predicted.", "why": "10.68 × 16,384,000<sup>−0.121</sup> ≈ 1.431. Note that the measured "
                                                      "1.586 at 8.2M tokens already sits above the line's value.", "key": {'parts': [{'label': None, 'value': 1.43, 'tol': 0.0286, 'unit': None}]}},
            {"kind": "number", "q": "Four times more tokens multiplies the loss by about how much?",
             "answer": "About 0.85.", "why": "0.92² ≈ 0.846 (= 4<sup>−0.121</sup>).", "key": {'parts': [{'label': None, 'value': 0.85, 'tol': 0.017, 'unit': None}]}},
        ],
    },
    {
        "title": "Real scaling laws",
        "segment": (92, 137),
        "figures": [{"t": 107.8, "caption": "The same shape on real language models; Chinchilla: about 20 tokens per parameter "
                                            "for a fixed compute budget."},
                    {"t": 119.1, "caption": "Fit on small, cheap runs; extrapolate to a model 1,000× larger."}],
        "body": [
            """<p>Research labs found the same shape on real language models, over many orders of magnitude of size,
data and compute. A widely cited result, from the <b>Chinchilla</b> paper (2022), is that for a fixed compute budget the
best results come from about <b>20 training tokens per parameter</b>. Our biggest model had 3.4.</p>""",
            """<p>That is what makes huge training runs possible to plan: fit the law on small, cheap runs and predict the
loss of a model a thousand times larger before spending the money. The laws describe the loss, not every ability, and
they bend when something runs short, as our biggest model showed. In code, fitting one is fitting a straight line through
the logarithms.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Scaling laws turn “bigger is better” into a number you can
plan with.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "At about 20 tokens per parameter, how many tokens would our 1,198,273-parameter model "
                                    "need? And a 7-billion-parameter model?",
             "answer": "About 24 million; 140 billion.", "why": "20 × 1,198,273 ≈ 23,965,460; 20 × 7 × 10⁹.", "key": {'parts': [{'label': 'our model (millions)', 'value': 24, 'tol': 0.5, 'unit': None}, {'label': '7B model (billions)', 'value': 140, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“A scaling law guarantees a bigger model will be better at every task.”",
             "answer": "False.", "why": "It predicts the average loss; specific abilities may not follow it smoothly.", "key": {'value': False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Fit the law to only the four smallest models. How close is its "
                                  "prediction for the 1,198,273-parameter model?",
             "code": """a, s = fit_power_law(*zip(*sizes[:4]))
print(a * 1198273 ** s)""",
             "answer": "The four-point fit is 4.81 × N^−0.083, predicting 1.506: well below the measured 1.612, because "
                       "the big model is short of data (computed from the measured losses).",
             "why": "Extrapolations assume nothing else (here, data) becomes the limit."},
        ],
    },
]
