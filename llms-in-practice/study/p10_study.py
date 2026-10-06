"""Study guide content for LLMs in Practice, episode 10: Quantization.

Build:  python framework/study_guide.py llms-in-practice p10 --video llms-in-practice/media/videos/p10_scene/1080p60/QuantizationVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/p10_quantization (Qwen2.5-0.5B-Instruct, simulated round-to-nearest per row;
transformers 4.57.1, torch 2.14.0, CPU); arithmetic checked in Python.
"""

LESSON = {
    "series": "LLMs in Practice",
    "label": "Episode 10",
    "title": "Quantization",
    "tagline": "Shrinking a model to fit a laptop",
    "duration": "2:30",
    "intro": """<p>This lesson answers one question: how do people run big models on laptops and phones? A model is
mostly weights, normally stored with 16 or 32 bits each. <b>Quantization</b> stores each weight with fewer bits: round
it to a small grid of integers and keep one scale per row. On a real 0.5B model, <b>8 bits</b> is a quarter of the
size with practically the same loss; <b>4 bits</b> is an eighth, noticeably worse but usable; below that, the model
falls apart.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Foundations F03 (matrices) and How LLMs Work episode 11
(loss). Code: <code>code/p10_quantization</code> (downloads Qwen2.5-0.5B-Instruct, about 1 GB; CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Where the size comes from",
        "segment": (8, 35),
        "figures": [{"t": 34.0, "caption": "Bits per weight: 32, 16, 8 or 4."}],
        "body": [
            """<p>A model is mostly numbers, its <b>weights</b>. Qwen2.5-0.5B-Instruct has <b>494,032,768</b>. At 32
bits (4 bytes) each, that is about <b>1.98 GB</b>; big models need hundreds of GB. Each weight is normally a 32-bit
number (4 bytes) or 16 bits (2 bytes). <b>Quantization</b> stores each weight with fewer bits.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Memory ≈ number of weights × bits per weight. Halve the bits,
halve the size.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A 7-billion-parameter model is stored in 16-bit. About how many GB is that? And at "
                                    "4 bits?",
             "answer": "About 14 GB; about 3.5 GB.", "why": "7 × 10⁹ × 2 bytes = 14 × 10⁹ bytes; × 0.5 byte = 3.5 × 10⁹ "
                                                          "bytes (plus a little for the scales).", "key": {'parts': [{'label': '16-bit (GB)', 'value': 14, 'tol': 0.5, 'unit': None}, {'label': '4-bit (GB)', 'value': 3.5, 'tol': 0.07, 'unit': None}]}},
            {"kind": "tf", "q": "“Quantization removes weights from the model.”",
             "answer": "False.", "why": "Every weight is kept; each one is stored with fewer bits.", "key": {'value': False}},
        ],
    },
    {
        "title": "Round to a grid, keep a scale",
        "segment": (35, 66),
        "figures": [{"t": 54.5, "caption": "Round each weight in a row to the nearest step between −max and +max; store "
                                           "small integers plus one scale."},
                    {"t": 64.8, "caption": "Real weights from the first layer, before and after 4-bit rounding: two "
                                           "became exactly zero."}],
        "body": [
            """<p>The simplest recipe: take one row of a weight matrix and find its largest absolute value. Round every
weight to the nearest step on a small grid from −max to +max. With <b>8 bits</b> that is <b>255</b> steps
(−127 … 127); with <b>4 bits</b>, only <b>15</b> (−7 … 7). Store the small integers, plus one scale for the row.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The real first weights of the first layer, before and after 4-bit rounding: −0.0019 → 0, −0.0052 →
−0.0082, 0.0188 → 0.0165 … Close, but not the same; two small ones became exactly <b>zero</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>weight ≈ integer × scale. Fewer bits = a coarser grid = bigger
rounding errors.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A row's largest absolute weight is 0.0329. With 4 bits (integers −7 … 7), what is the "
                                    "scale, and what does 0.0188 become?",
             "answer": "Scale 0.0047; 0.0188 → 4 × 0.0047 = 0.0188.",
             "why": "0.0329 / 7 = 0.0047; 0.0188 / 0.0047 = 4.0, already on the grid. (In the real row, the max is "
                    "different, so 0.0188 became 0.0165.)", "key": {'parts': [{'label': 'scale', 'value': 0.0047, 'tol': 5e-05, 'unit': None}, {'label': '0.0188 becomes (dequantized)', 'value': 0.0188, 'tol': 5e-05, 'unit': None}]}},
            {"kind": "number", "q": "How many integer levels does symmetric 3-bit quantization use, and why so few?",
             "answer": "7 (−3 … 3).", "why": "2³⁻¹ − 1 = 3, so the integers run from −3 to 3: every weight in a row has "
                                            "only 7 possible values.", "key": {'parts': [{'label': None, 'value': 7, 'tol': 0.5, 'unit': None}]}},
            {"kind": "short", "q": "Why does a weight smaller than half a step become exactly zero?",
             "answer": "Rounding to the nearest grid point sends it to 0, the closest level.",
             "why": "With 4 bits, any weight smaller than half the scale rounds to 0."},
        ],
    },
    {
        "title": "Measuring the damage",
        "segment": (66, 104),
        "figures": [{"t": 86.4, "caption": "fp32 vs int8: a quarter of the size, practically identical loss."},
                    {"t": 103.0, "caption": "Lower bits: int4 is usable; int3 and int2 fall apart."}],
        "body": [
            """<p>The weight matrices hold <b>72%</b> of the parameters (the rest are mostly the embeddings). Loss on
4,096 tokens of real text (the How LLMs Work narration):</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p><b>fp32</b>: 1,365 MB, loss 2.836. <b>int8</b>: 341 MB (a quarter), weights move about 1.1%, loss
<b>2.837</b>: practically identical. <b>int4</b>: 171 MB (an eighth), weights move 19%, loss rises to <b>3.58</b>;
answers still make sense but are vaguer. <b>int3</b>: loss <b>12.5</b>; <b>int2</b>: <b>16.6</b>, and the model writes
gibberish.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Quality holds up well to about 8 bits, degrades at 4 with
simple rounding, and collapses below that.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many times smaller are the int4 matrices than the fp32 ones?",
             "answer": "8 times.", "why": "32 / 4 = 8; 1,365 MB / 8 ≈ 171 MB.", "key": {'parts': [{'label': None, 'value': 8, 'tol': 0.5, 'unit': None}]}},
            {"kind": "order", "q": "Order from lowest to highest loss: <i>int2 · fp32 · int4 · int8 · int3</i>.",
             "answer": "fp32 (2.836) → int8 (2.837) → int4 (3.582) → int3 (12.491) → int2 (16.610).",
             "why": "Fewer bits, larger rounding errors, higher loss.", "key": {'items': ['fp32', 'int8', 'int4', 'int3', 'int2']}},
        ],
    },
    {
        "title": "Subtleties and better methods",
        "segment": (104, 150),
        "figures": [{"t": 116.0, "caption": "At int8 the loss barely moved, but the greedy answer changed: near-tied "
                                            "tokens flip."},
                    {"t": 133.0, "caption": "Real tools: per-group scales, sensitive weights kept precise, "
                                            "calibration."}],
        "body": [
            """<p>At 8 bits the loss barely moved, but the model's answer to our question still <b>changed</b>. When two
tokens are almost tied, a tiny change can flip the choice, and everything after it (episode 3). Average quality and a
single answer are different things to measure.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Real tools do much better than simple rounding: they use <b>small groups</b> of weights with their own
scale, keep the most <b>sensitive</b> weights at higher precision, and <b>calibrate</b> on sample text. That is why
4-bit versions of big models are popular. In code, the simple version is three lines per matrix: compute the scale
from the largest value, round to integers, multiply back.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Quantize, then measure on your own tasks: an unchanged average
loss does not guarantee unchanged answers.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why does a separate scale for each small group of weights help, compared with one "
                                   "scale per row?",
             "answer": "One large weight no longer stretches the grid for the whole row; each group gets a finer grid "
                       "fitted to its own values.",
             "why": "The scale is set by the largest value; smaller groups have smaller maxima, so smaller steps."},
            {"kind": "code", "q": "<b>Try it yourself.</b> In <code>code/p10_quantization/quantize.py</code>, change the "
                                  "loop to <code>for bits in (6, 5):</code> and run it. What sizes and losses do you get, "
                                  "and where between 8 and 4 bits does quality start to drop?",
             "code": """for bits in (6, 5):""",
             "answer": "int6: 256 MB, weights move 4.4%, loss 2.875. int5: 213 MB, 9.0%, loss 2.944. Quality starts to "
                       "drop around 6 bits and falls faster from 5 to 4 bits (3.582).",
             "why": "Each bit removed halves the number of grid steps, so the average weight change roughly doubles: "
                    "1.1% → 4.4% → 9.0% → 19.2%."},
        ],
    },
]
