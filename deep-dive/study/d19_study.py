"""Study guide content for How LLMs Work: Deep Dive, episode 19: Adam and AdamW.

Build:  python framework/study_guide.py deep-dive d19 --video deep-dive/media/videos/d19_scene/1080p60/AdamVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d19_adam/adam.py (Adam written by hand and checked against torch.optim.Adam; a 4-layer tiny
GPT trained 1,500 steps; torch 2.14.0, CPU) or the same formulas on the exercise inputs.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 19",
    "title": "Adam and AdamW",
    "tagline": "How far to move each weight",
    "duration": "2:46",
    "intro": """<p>This lesson answers one question: once backprop has given every weight a gradient, how far should each
weight move? Plain gradient descent moves it by learning rate × gradient, so its steps follow the gradient's scale.
<b>Adam</b> normalizes each weight's step by the typical size of its gradients, using running averages m and v. In a tiny
GPT it beats SGD and momentum. <b>AdamW</b> adds weight decay separately from that normalization; putting the decay into
the gradient instead (Adam + L2) collapses the weights at the same coefficient.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episode 18 (backprop) and Foundations
(gradient descent). Code: <code>code/d19_adam</code> (PyTorch only; eight tiny models, about 25 minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Why plain gradient descent struggles",
        "segment": (8, 41),
        "figures": [{"t": 19.0, "caption": "Gradients → optimizer → weight updates. Almost every LLM uses AdamW."},
                    {"t": 39.6, "caption": "Loss × 1,000: SGD's first step grows from 0.0244 to 24.4; Adam's stays 0.0100."}],
        "body": [
            """<p>Backprop gives every weight a gradient; the <b>optimizer</b> decides how far to move each one. Plain
gradient descent (SGD) uses w ← w − lr · g, so the step depends on the scale of the gradient. Multiply the loss by 1,000
and SGD's first step (largest change in any weight, lr 0.01) grows from <b>0.0244</b> to <b>24.4</b>. Adam's first step
is <b>0.0100</b> both times.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>SGD's step size depends on the gradient's scale; Adam's is set
by the learning rate.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With SGD and lr 0.01, a weight's gradient is 50. How far does it move?",
             "answer": "0.5.", "why": "lr × g = 0.01 × 50."},
            {"kind": "tf", "q": "“Multiplying the loss by 1,000 multiplies SGD's update by 1,000.”",
             "answer": "True.", "why": "The gradient scales by 1,000 and SGD's update is proportional to it (0.0244 → "
                                      "24.4)."},
        ],
    },
    {
        "title": "Adam's idea",
        "segment": (41, 71),
        "figures": [{"t": 62.5, "caption": "m: running average of the gradient; v: of its square; step = lr · m̂ / (√v̂ + ε)."},
                    {"t": 69.4, "caption": "Our 10-line Adam matches torch.optim.Adam after 10 steps to 3 × 10⁻⁸."}],
        "body": [
            """<p>Adam keeps two running averages for every weight: <b>m</b>, the average gradient (m ← 0.9 m + 0.1 g),
and <b>v</b>, the average squared gradient (v ← 0.999 v + 0.001 g²). The step is <b>lr · m̂ / (√v̂ + ε)</b>: the
direction comes from m, the size is normalized by how big that weight's gradients usually are. Both averages start at
zero, so early on they are divided by (1 − 0.9<sup>t</sup>) and (1 − 0.999<sup>t</sup>), the <b>bias correction</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Our own Adam, about ten lines, matches PyTorch's after ten steps to within <b>3 × 10<sup>−8</sup></b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Adam = momentum (m) + a per-weight step size (1/√v): every
weight moves about lr per step at first, whatever the scale of its gradient.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "At step 1, m̂ = g and v̂ = g². What is the step for every weight, and why is the first "
                                   "step 0.0100 in the episode?",
             "answer": "lr · g / |g| = lr · sign(g): every weight moves by exactly lr (0.01), up or down.",
             "why": "Checked: with lr 0.05 the first step is 0.0500 for every weight."},
            {"kind": "number", "q": "At step 1, by what factors do the bias corrections scale m and v?",
             "answer": "10 and 1,000.", "why": "1 / (1 − 0.9) and 1 / (1 − 0.999)."},
        ],
    },
    {
        "title": "Training the tiny GPT",
        "segment": (71, 89),
        "figures": [{"t": 88.4, "caption": "1,500 steps: SGD lr 0.1 2.476; lr 1.0 blew up; momentum 1.940; Adam 1.708; our "
                                           "Adam 1.708."}],
        "body": [
            """<p>The same 4-layer tiny GPT, 1,500 steps. Validation loss: SGD with lr 0.1 <b>2.476</b>; with lr 1.0 it
blows up (NaN); SGD with momentum 0.9 and lr 0.1 <b>1.940</b>; Adam with lr 0.001 <b>1.708</b>; our hand-written Adam
exactly the same, 1.708.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>On transformers, Adam trains faster and more reliably than
SGD, with less tuning of the learning rate.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Order from best to worst validation loss: <i>SGD lr 0.1 · Adam · SGD + momentum</i>.",
             "answer": "Adam (1.708) → SGD + momentum (1.940) → SGD lr 0.1 (2.476).", "why": "From the episode's run."},
        ],
    },
    {
        "title": "Weight decay: Adam + L2 vs AdamW",
        "segment": (89, 140),
        "figures": [{"t": 103.2, "caption": "Adam + L2 adds 0.1 · w to the gradient; AdamW shrinks the weight directly by "
                                            "lr · 0.1 · w."},
                    {"t": 125.8, "caption": "Weight size: no decay 173.8, Adam + L2 4.0, AdamW 127.0; val loss 1.666, 3.307, "
                                            "1.682."}],
        "body": [
            """<p><b>Weight decay</b> pulls weights toward zero to keep them small. <b>Adam + L2</b> adds wd · w to the
gradient; <b>AdamW</b> shrinks each weight directly, w ← w − lr · wd · w, separately from the gradient step.</p>""",
            """<p>With Adam the first way backfires: the decay term is normalized like any gradient, so a weight whose
loss gradient is small gets pushed toward zero by about lr each step, whatever its size. With wd 0.1 and lr 0.003, the
total size of the weights collapses from <b>173.8</b> (no decay) to <b>4.0</b>, and the loss rises to <b>3.307</b>.
AdamW: size <b>127.0</b>, loss <b>1.682</b> (no decay: 1.666).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>To be fair, the same coefficient means very different things in the two methods; Adam + L2 would need a
much smaller one. AdamW keeps decay separate from Adam's normalization, so it behaves predictably: that is why it became
the default.</p>""",
            """<div class="box key"><b class="t">Key idea</b>In AdamW, weight decay is a plain shrink by (1 − lr · wd) per
step, independent of the gradients.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With AdamW, lr 0.003 and wd 0.1, by what factor would decay alone shrink a weight over "
                                    "1,500 steps?",
             "answer": "About 0.64.", "why": "(1 − 0.0003)<sup>1500</sup> ≈ 0.638."},
            {"kind": "mc", "q": "Why did Adam + L2 collapse the weights?",
             "options": ["The learning rate was too small", "The decay term was normalized by √v, so it moved every "
                         "weight by about lr per step", "AdamW has a bug", "Weight decay was turned off"],
             "answer": "B.", "why": "When wd · w dominates the gradient, Adam's normalization turns it into steps of about lr "
                                   "toward zero."},
        ],
    },
    {
        "title": "The cost, and the code",
        "segment": (140, 158),
        "figures": [{"t": 149.8, "caption": "GPT-2 small: 475 MiB of weights, plus 949 MiB for Adam's m and v."},
                    {"t": 157.0, "caption": "Three lines: update m, update v, step by m over √v; AdamW decays separately."}],
        "body": [
            """<p>The price: two extra numbers per weight. For GPT-2 small in float32, <b>475 MiB</b> of weights need
<b>949 MiB</b> more for Adam's m and v. In code, the heart of Adam is three lines: update m, update v, and step by m̂ /
(√v̂ + ε); AdamW adds one separate line for the decay.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Adam triples the memory needed for parameters (weights + m +
v), before counting gradients and activations.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many GB do Adam's m and v take for a 7-billion-parameter model in float32?",
             "answer": "56 GB.", "why": "2 × 7 × 10⁹ × 4 bytes."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Repeat part 2 with lr 0.05. What is Adam's first step for each "
                                  "weight?",
             "code": """opt = torch.optim.Adam([w], lr=0.05)
((x @ w) ** 2).mean().backward(); opt.step()""",
             "answer": "0.0500 for every weight (within 10⁻⁷).", "why": "The first step is lr · sign(g)."},
        ],
    },
]
