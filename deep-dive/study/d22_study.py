"""Study guide content for How LLMs Work: Deep Dive, episode 22: Mixed Precision.

Build:  python framework/study_guide.py deep-dive d22 --video deep-dive/media/videos/d22_scene/1080p60/MixedPrecisionVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d22_mixed_precision/mixed_precision.py (torch.finfo; a 4-layer tiny GPT trained 2,000 steps;
GPT-2 small; transformers 4.57.1, torch 2.14.0, CPU) or the same conversions run on the exercise values.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 22",
    "title": "Mixed Precision",
    "tagline": "Training in 16 bits without losing small updates",
    "duration": "2:31",
    "intro": """<p>This lesson answers one question: how can models train with 16-bit numbers without breaking? 16-bit
formats halve memory and speed up math, but <b>float16</b> has a small range (tiny gradients underflow to zero) and
<b>bfloat16</b> has coarse steps (small updates to a weight of 1.0 vanish). <b>Mixed precision</b> does the matrix math in
16 bits but keeps float32 master weights: in a tiny GPT it matches float32 (1.645 vs 1.644), while pure bfloat16 weights
fall behind (1.677).</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 18 (backprop) and 19 (AdamW), and
Foundations (binary numbers). Code: <code>code/d22_mixed_precision</code> (GPT-2 small, about 500 MB, plus three tiny
models, about 15 minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Three formats",
        "segment": (8, 44),
        "figures": [{"t": 20.8, "caption": "32 bits vs 16 bits: half the memory, and much faster math on modern GPUs."},
                    {"t": 42.5, "caption": "float32, float16 and bfloat16: largest value, step after 1.0, smallest normal "
                                           "number."}],
        "body": [
            """<p>Most large models are trained with 16-bit numbers instead of 32: half the memory, much faster math on
modern GPUs. But 16 bits cannot hold everything. A floating-point number has a sign, an <b>exponent</b> (the range) and a
<b>mantissa</b> (the precision):</p>
<table><tr><th>format</th><th>bits (sign · exponent · mantissa)</th><th>largest</th><th>step after 1.0</th><th>smallest normal</th></tr>
<tr><td>float32</td><td>1 · 8 · 23</td><td>3.4 × 10³⁸</td><td>1.2 × 10⁻⁷</td><td>1.2 × 10⁻³⁸</td></tr>
<tr><td>float16</td><td>1 · 5 · 10</td><td>65,504</td><td>0.00098</td><td>6.1 × 10⁻⁵</td></tr>
<tr><td>bfloat16</td><td>1 · 8 · 7</td><td>3.4 × 10³⁸</td><td>0.0078 (1/128)</td><td>1.2 × 10⁻³⁸</td></tr></table>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>float16 trades range for precision; bfloat16 keeps float32's
range and gives up precision.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "What do 70,000 become in float16 and in bfloat16?",
             "answer": "float16: infinity (overflow, above 65,504). bfloat16: 70,144 (in range, but rounded).",
             "why": "Checked with torch: float16 has a small range, bfloat16 coarse steps."},
            {"kind": "number", "q": "How many GB do the weights of a 7-billion-parameter model take in bfloat16, and in "
                                    "float32?",
             "answer": "14 GB and 28 GB.", "why": "2 and 4 bytes per parameter.", "key": {'parts': [{'label': 'bfloat16 (GB)', 'value': 14, 'tol': 0.5, 'unit': None}, {'label': 'float32 (GB)', 'value': 28, 'tol': 0.5, 'unit': None}]}},
        ],
    },
    {
        "title": "What breaks: rounding and underflow",
        "segment": (44, 81),
        "figures": [{"t": 57.6, "caption": "1.0 + 0.001: float16 keeps it (1.0009766), bfloat16 rounds it away; 1.0 + 0.0001: "
                                           "both lose it."},
                    {"t": 80.4, "caption": "A gradient of 10⁻⁹: 0 in float16; ×1024 then ÷1024 rescues it; bfloat16 keeps "
                                           "it."}],
        "body": [
            """<p><b>Rounding.</b> Coarse steps swallow small updates. 1.0 + 0.001 is 1.0010000 in float32, 1.0009766 in
float16, and exactly 1.0000000 in bfloat16; 1.0 + 0.0001 stays 1.0 in both 16-bit formats. A small learning-rate step on
a 16-bit weight can simply vanish.</p>""",
            """<p><b>Underflow.</b> A gradient of 10<sup>−9</sup> becomes exactly 0 in float16. The fix is <b>loss
scaling</b>: multiply the loss by 1,024 so every gradient is 1,024 times larger, store it, then divide back: 9.9 ×
10<sup>−10</sup> survives. bfloat16's wide range keeps it (1.0 × 10<sup>−9</sup>) without any scaling.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>float16 needs loss scaling against underflow; any 16-bit
format needs higher-precision weights to keep small updates.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In bfloat16, what is 1.0 + 1/256? And 1.0 + 1/128?",
             "answer": "1.0 and 1.0078125.", "why": "The step after 1.0 is 1/128; half a step rounds back to 1.0.", "key": {'parts': [{'label': '1.0 + 1/256', 'value': 1, 'tol': 0.001, 'unit': None}, {'label': '1.0 + 1/128', 'value': 1.0078125, 'tol': 0.0001, 'unit': None}]}},
            {"kind": "tf", "q": "“Loss scaling is needed with bfloat16 for the same reason as with float16.”",
             "answer": "False.", "why": "bfloat16 has float32's exponent range, so tiny gradients do not underflow.", "key": {'value': False}},
        ],
    },
    {
        "title": "Mixed precision in practice",
        "segment": (81, 141),
        "figures": [{"t": 114.2, "caption": "Tiny GPT: float32 1.644; bfloat16 math + float32 weights 1.645; pure bfloat16 "
                                            "weights 1.677."},
                    {"t": 131.5, "caption": "GPT-2 in bfloat16: 237 MiB instead of 475, loss 3.99 instead of 4.00."}],
        "body": [
            """<p>Training therefore mixes precisions: the heavy matrix math runs in 16 bits, but a <b>master copy</b> of
the weights stays in float32, so small updates are not rounded away. The same tiny GPT, 2,000 steps: all float32
<b>1.644</b>; bfloat16 math with float32 weights <b>1.645</b>, the same; pure bfloat16 weights <b>1.677</b>, worse,
likely because small updates were rounded away.</p>""",
            """<p>For using a model, 16 bits are usually enough: GPT-2 in bfloat16 takes <b>237 MiB</b> instead of 475,
and its loss on 1,024 tokens of Shakespeare barely moves (3.9897 vs 3.9996). In code, mixed precision is one context
manager, <code>torch.autocast</code>, around the forward pass; the weights and the optimizer stay in float32.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>16-bit math, 32-bit weights: the speed of the first, the
accuracy of the second.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which setup matched float32 training in the episode?",
             "options": ["Pure bfloat16 weights", "bfloat16 math with float32 master weights", "float16 weights without "
                         "loss scaling", "None of them"],
             "answer": "B.", "why": "1.645 vs 1.644.", "key": {'choice': 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Check that bfloat16 cannot add 1 to 256.",
             "code": """a = torch.tensor(256.0, dtype=torch.bfloat16)
print(a + 1)""",
             "answer": "256.", "why": "Around 256, bfloat16's step is 2: 257 rounds back to 256."},
        ],
    },
]
