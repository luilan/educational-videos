"""Study guide content for How LLMs Work: Deep Dive, episode 33: Quantization, Deeper.

Build:  python framework/study_guide.py deep-dive d33 --video deep-dive/media/videos/d33_scene/1080p60/QuantizationVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d33_quantization/quantization.py (Qwen2.5-0.5B, float32, mean loss on 4 × 512 tokens of
Tiny Shakespeare; transformers 4.57.1, torch 2.14.0) or from the arithmetic shown.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 33",
    "title": "Quantization, Deeper",
    "tagline": "Outliers, groups, and better grids",
    "duration": "2:30",
    "intro": """<p>This lesson answers one question: why does naive 4-bit rounding hurt a model, and how do real tools
make it almost free? The culprit is <b>outliers</b>: a few large values set the scale and crush everything else. Giving
every <b>group</b> of 32 to 128 weights its own scale fixes most of the damage for a fraction of a bit; the <b>NF4</b>
grid places its levels where weights actually are; and for activations, where the outlier is a whole token (the attention
sink), one scale <b>per token</b> makes 8-bit inputs nearly free.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> LLMs in Practice episode 10 (quantization basics) and
Deep Dive episode 12 (attention sinks). Code: <code>code/d33_quantization</code> (Qwen2.5-0.5B, about 1 GB).</div>""",
}

CONCEPTS = [
    {
        "title": "Outliers set the scale",
        "segment": (8, 35),
        "figures": [{"t": 34.0, "caption": "One row of layer 0's down_proj: typical |w| 0.0125, largest 0.170. Four bits with one "
                                           "scale give steps of 0.024: 49% of the weights round to 0."}],
        "body": [
            """<p>Symmetric 4-bit rounding uses the integers −7 to 7: the step is the row's largest absolute value divided
by 7. Rounding to 4 bits makes a 32-bit model 8 times smaller. Done with one scale per row, it also makes it much
worse.</p>""",
            "{fig0}",
            """<p>The reason is outliers. In one row of Qwen2.5-0.5B's first layer the typical (median) weight is 0.0125
and the largest 0.170, 14 times bigger. The step becomes 0.024, so every weight smaller than half a step, 49% of them,
rounds to exactly zero. Across all 304,128 rows, the largest weight is typically 6 times the median, and up to 145
times.</p>""",
            """<div class="box key"><b class="t">Key idea</b>One large value sets the scale for everything that shares it.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A row's largest absolute weight is 0.35. With symmetric 4-bit rounding (−7..7), what is "
                                    "the step, and which weights round to 0?",
             "answer": "Step 0.05; every weight with |w| < 0.025.", "why": "0.35 / 7 = 0.05; values under half a step round "
                                                                         "to 0.", "key": {'parts': [{'label': 'step', 'value': 0.05, 'tol': 0.005, 'unit': None}, {'label': 'rounds to 0 when |w| <', 'value': 0.025, 'tol': 0.0005, 'unit': None}]}},
            {"kind": "number", "q": "How many levels does symmetric 3-bit rounding use?",
             "answer": "7 (−3..3).", "why": "2² − 1 = 3 on each side, plus zero.", "key": {'parts': [{'label': None, 'value': 7, 'tol': 0.5, 'unit': None}]}},
        ],
    },
    {
        "title": "Groups and their cost",
        "segment": (35, 72),
        "figures": [{"t": 55.4, "caption": "Loss with 4-bit weights: full precision 3.393; per row 4.186; groups of 128, 64, 32: "
                                           "3.756, 3.622, 3.570."},
                    {"t": 70.6, "caption": "Scales cost 16 / group bits per weight; at 3 bits, groups of 32 rescue the model "
                                           "(12.5 → 4.52)."}],
        "body": [
            """<p>The fix is groups: every block of, say, 64 consecutive weights gets its own scale, so an outlier only
spoils its own group. Mean loss on Shakespeare, with every linear layer's weights rounded to 4 bits:</p>""",
            """<table><tr><th>4-bit weights</th><th>loss</th><th>weight error</th><th>bits per weight</th></tr>
<tr><td>full precision</td><td>3.393</td><td>–</td><td>32</td></tr>
<tr><td>one scale per row</td><td>4.186</td><td>17.9%</td><td>4.00</td></tr>
<tr><td>groups of 128</td><td>3.756</td><td>13.2%</td><td>4.12</td></tr>
<tr><td>groups of 64</td><td>3.622</td><td>11.8%</td><td>4.25</td></tr>
<tr><td>groups of 32</td><td>3.570</td><td>10.4%</td><td>4.50</td></tr></table>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The scales aren't free: one 16-bit scale per 32 weights adds half a bit per weight. It is worth it, and
more so at lower precision: at 3 bits, one scale per row destroys the model (loss 12.5), and groups of 32 bring it back
to 4.52.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Smaller groups isolate outliers; each scale costs 16 / group
bits per weight.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many gigabytes does a 7-billion-parameter model take at 4 bits with one 16-bit scale "
                                    "per 32 weights? And in 16 bits?",
             "answer": "About 3.9 GB; 14 GB.", "why": "7e9 × 4.5 / 8 = 3.94e9 bytes; 7e9 × 2 = 14e9 bytes.", "key": {'parts': [{'label': '4-bit (GB)', 'value': 3.9, 'tol': 0.078, 'unit': None}, {'label': '16-bit (GB)', 'value': 14, 'tol': 0.5, 'unit': None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Round this row to 4 bits with one scale, then with groups of 4. "
                                  "Which weights survive?",
             "code": """import torch
w = torch.tensor([[0.01, -0.02, 0.015, 0.9, 0.012, -0.008, 0.02, -0.011]])
def quant(w, g):
    x = w.reshape(1, -1, g)
    s = x.abs().amax(-1, True) / 7
    return ((x / s).round() * s).reshape(w.shape)
print(quant(w, 8)); print(quant(w, 4))""",
             "answer": "One scale: only 0.9 survives, the other seven become 0. Groups of 4: the second group keeps all its "
                       "weights (0.0114, −0.0086, 0.0200, −0.0114); the first still loses its three small ones.",
             "why": "The outlier only spoils the group it is in."},
        ],
    },
    {
        "title": "A better grid: NF4",
        "segment": (72, 92),
        "figures": [{"t": 91.0, "caption": "Evenly spaced levels vs NF4 levels, denser near zero. Loss increase at 4 bits, "
                                           "groups of 64: +0.229 vs +0.140."}],
        "body": [
            """<p>Weights follow a bell curve: most are near zero. Evenly spaced levels waste steps on the rare large
values. <b>NF4</b> (from QLoRA) places its 16 levels at quantiles of a normal distribution, so each level covers about
the same share of weights: −1.00, −0.70, −0.53, …, 0, …, 0.56, 0.72, 1.00, times the group's scale. Same 4 bits; the loss
increase over full precision drops from <b>+0.229</b> to <b>+0.140</b>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Put the levels where the values are.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“NF4 is better than evenly spaced 4-bit levels because it stores more bits.”",
             "answer": "False.", "why": "Both use 4 bits (16 levels); NF4 places them better for bell-shaped weights.", "key": {'value': False}},
        ],
    },
    {
        "title": "Activations: the outlier is a token",
        "segment": (92, 131),
        "figures": [{"t": 112.0, "caption": "Inputs to layer 2's down_proj: the first token reaches 1,808; a typical token 1.86 "
                                            "(969x smaller)."},
                    {"t": 130.0, "caption": "8-bit inputs to every linear layer: per tensor 4.709; first token kept 3.475; per "
                                            "token 3.415 (full precision 3.393)."}],
        "body": [
            """<p>To use fast 8-bit arithmetic, the inputs to each layer must be rounded too. There, the outlier is a whole
token: the very first one carries enormous values, the attention sink of episode 12. It has the largest input in 84 of
168 linear layers; in layer 2's down_proj it reaches 1,808, against 1.86 for a typical token.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>With one scale for the whole tensor the step there is 14.2, and 98% of the other tokens round to all
zeros: the loss jumps to 4.709. Keeping the first token in 16 bits brings it back to 3.475; giving every token its own
scale gives 3.415, almost free against 3.393.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Find where the outliers live (rows, groups, tokens) and give
them their own scale.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With one 8-bit scale set by the value 1,808 (−127..127), what is the step, and what does "
                                    "a token whose largest value is 1.86 become?",
             "answer": "Step 14.2; all zeros.", "why": "1,808 / 127 = 14.2; every value below 7.1 rounds to 0.", "key": {'parts': [{'label': None, 'value': 14.2, 'tol': 0.05, 'unit': None}]}},
            {"kind": "mc", "q": "Why does one scale per token work so well here?",
             "options": ["It uses more bits", "Each token's step is set by its own largest value, so the first token cannot "
                                              "crush the others", "It skips the first token", "It rounds weights, not inputs"],
             "answer": "B.", "why": "The outlier is confined to one token, so a per-token scale isolates it.", "key": {'choice': 1}},
        ],
    },
]
