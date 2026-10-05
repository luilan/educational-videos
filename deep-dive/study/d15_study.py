"""Study guide content for How LLMs Work: Deep Dive, episode 15: Activations: ReLU, GELU, SwiGLU.

Build:  python framework/study_guide.py deep-dive d15 --video deep-dive/media/videos/d15_scene/1080p60/ActivationsVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d15_activations/activations.py (GPT-2 small and Qwen2.5-0.5B-Instruct, real weights; a
4-layer tiny GPT trained 3,000 steps per activation and seed; transformers 4.57.1, torch 2.14.0, CPU).
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 15",
    "title": "Activations: ReLU, GELU, SwiGLU",
    "tagline": "The bend inside every MLP",
    "duration": "2:45",
    "intro": """<p>This lesson answers one question: what is the “bend” in the middle of every MLP, and why have models
moved from ReLU to GELU to <b>SwiGLU</b>? Without an activation, the MLP's two matrices collapse into one. GPT-2's GELU
keeps small negative values that the model relies on (swap in ReLU and the loss jumps from 4.13 to 7.23). SwiGLU adds a
learned gate; in a tiny GPT with about the same parameter budget it beat GELU, which beat ReLU, in all three seeds.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episode 8 (the MLP) and Deep Dive episode
13 (the residual stream). Code: <code>code/d15_activations</code> (GPT-2 small and Qwen2.5-0.5B, about 1.5 GB, plus nine
tiny models, about 40 minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Why an MLP needs a bend",
        "segment": (8, 46),
        "figures": [{"t": 26.8, "caption": "Widen, bend, narrow. Without the bend, two matrices collapse into one."},
                    {"t": 44.5, "caption": "ReLU (zero for negatives) and GELU (smooth, dipping to −0.17)."}],
        "body": [
            """<p>Every MLP has the same shape: <b>widen</b> (768 → 3,072 in GPT-2), <b>bend</b>, <b>narrow</b> (back to
768). The bend is the activation function. Without it, W₂(W₁x) = (W₂W₁)x: the two matrices would collapse into one, and
the MLP could only compute linear functions.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p><b>ReLU</b> is the simplest bend: negative numbers become 0, positive ones pass. <b>GELU</b> is a smooth
version: large positive numbers pass, large negative ones fade to 0, but in between it dips below zero, down to
<b>−0.17</b> (near x = −0.75). At x = −1: ReLU 0, GELU −0.159, SiLU −0.269.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The activation is the only nonlinear step in the MLP; its
exact shape decides what passes and what is suppressed.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What are ReLU(0.5), GELU(0.5) and SiLU(0.5)?",
             "answer": "0.5, 0.346, 0.311.", "why": "From the code's table; the smooth functions pass less than ReLU near "
                                                    "zero."},
            {"kind": "short", "q": "Why can't an MLP without an activation compute anything a single matrix can't?",
             "answer": "W₂(W₁x) = (W₂W₁)x: the product of two matrices is one matrix.",
             "why": "Stacking linear maps gives a linear map."},
        ],
    },
    {
        "title": "GPT-2 relies on GELU's dip",
        "segment": (46, 69),
        "figures": [{"t": 67.7, "caption": "86% of the hidden values are negative before GELU; swapping in ReLU raises the "
                                           "loss from 4.13 to 7.23."}],
        "body": [
            """<p>In GPT-2's layer 6 MLP, <b>86%</b> of the 3,072 hidden values are negative before GELU. Most of them do
not become zero: only <b>4%</b> end up within 0.01 of zero. Swap every GELU for ReLU, without retraining, and the loss on
512 tokens of Shakespeare jumps from <b>4.13</b> to <b>7.23</b>. The model relies on those small negative values.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>A trained model is tuned to its exact activation; the
negative side of GELU carries information.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Because most hidden values are negative, GELU outputs are mostly exactly zero.”",
             "answer": "False.", "why": "Only 4% end up within 0.01 of zero; GELU maps moderate negative values to small "
                                       "negative outputs (down to −0.17)."},
            {"kind": "mc", "q": "Swapping GPT-2's GELU for ReLU without retraining…",
             "options": ["leaves the loss unchanged", "slightly improves it", "raises it from 4.13 to 7.23",
                         "makes it NaN"],
             "answer": "C.", "why": "Measured on 512 tokens of Shakespeare."},
        ],
    },
    {
        "title": "SwiGLU: a gated MLP",
        "segment": (69, 102),
        "figures": [{"t": 91.7, "caption": "SwiGLU: down(SiLU(gate(x)) × up(x)); Qwen2.5-0.5B: 896 → 4,864 → 896, matched "
                                           "exactly."},
                    {"t": 101.3, "caption": "Three matrices with a narrower middle (344 instead of 512) keep the parameter "
                                            "count about the same."}],
        "body": [
            """<p><b>SwiGLU</b> adds a gate. Two matrices go up: a <b>gate</b> and an <b>up</b> projection. The gate
passes through <b>SiLU</b> (x · sigmoid(x), a smooth bend like GELU) and multiplies the up projection number by number;
a third matrix goes <b>down</b>. In Qwen2.5-0.5B that is 896 → 4,864 → 896, with 13,074,432 parameters per MLP; the
hand computation matches the model exactly (difference 0.0).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Three matrices instead of two would be an unfair comparison, so SwiGLU uses a narrower middle, about two
thirds of the width: in the tiny GPT, 344 instead of 512 (132,096 parameters vs 131,072).</p>""",
            """<div class="box key"><b class="t">Key idea</b>SwiGLU = SiLU(gate · x) ⊙ (up · x), then down: each hidden
unit is a product of two learned signals.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Qwen2.5-0.5B's MLP has three 896 × 4,864 matrices and no biases. How many parameters?",
             "answer": "13,074,432.", "why": "3 × 896 × 4,864."},
            {"kind": "number", "q": "With D = 128, a GELU MLP (128 → 512 → 128, no biases) has 131,072 parameters. What "
                                    "middle width h gives a SwiGLU MLP (3 matrices of 128 × h) the same count?",
             "answer": "About 341 (the code uses 344).", "why": "3 × 128 × h = 131,072 → h ≈ 341.3; 344 is a round "
                                                                "multiple of 8."},
        ],
    },
    {
        "title": "Results, dead neurons, and the code",
        "segment": (102, 152),
        "figures": [{"t": 118.0, "caption": "Three seeds each: ReLU 1.651, GELU 1.620, SwiGLU 1.591 (means); the same order "
                                            "in every run."},
                    {"t": 131.2, "caption": "In the ReLU model, all 2,048 hidden units fired at least once on 40,960 "
                                            "tokens."}],
        "body": [
            """<p>The same 4-layer tiny GPT, trained 3,000 steps with each activation, three seeds each. Validation loss:
ReLU 1.653 / 1.649 / 1.652 (mean <b>1.651</b>), GELU 1.620 / 1.617 / 1.625 (<b>1.620</b>), SwiGLU 1.590 / 1.592 / 1.593
(<b>1.591</b>). The same order in every run: a small gain, at no extra cost.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>A classic worry with ReLU is <b>dead neurons</b>, units that never fire again. In the trained ReLU
model, all 2,048 hidden units fired at least once on 40,960 validation tokens, so at this scale that is not what made
ReLU worse. One common explanation for the gates' advantage: each hidden unit is a product of two learned signals, so
the MLP can switch features on and off depending on the input.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Smooth activations beat ReLU, and gating beats both, by a
small but consistent margin.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Order the activations from best to worst validation loss in the episode: <i>GELU · "
                                   "ReLU · SwiGLU</i>.",
             "answer": "SwiGLU (1.591) → GELU (1.620) → ReLU (1.651).", "why": "Means over three seeds; the same order in "
                                                                              "each seed."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Write SwiGLU's forward pass with plain tensors and check it "
                                  "against Qwen's MLP, as the code does.",
             "code": """gate, up = mlp.gate_proj(x), mlp.up_proj(x)
mine = mlp.down_proj(F.silu(gate) * up)
print((mine - mlp(x)).abs().max())""",
             "answer": "0.0.", "why": "Qwen2's MLP computes exactly down(silu(gate(x)) * up(x))."},
        ],
    },
]
