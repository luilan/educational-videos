"""Study guide content for How LLMs Work: Deep Dive, episode 18: Backprop Through a Transformer.

Build:  python framework/study_guide.py deep-dive d18 --video deep-dive/media/videos/d18_scene/1080p60/BackpropVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d18_backprop/backprop.py (GPT-2 small, real weights; float64 for the gradient check,
float32 for time and memory; transformers 4.57.1, torch 2.14.0, CPU, 8 threads) or the same code on the exercise inputs.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 18",
    "title": "Backprop Through a Transformer",
    "tagline": "One loss number, 124 million gradients",
    "duration": "2:35",
    "intro": """<p>This lesson answers one question: how does one backward pass give every weight of GPT-2 its gradient,
and can we trust it? Backprop is the <b>chain rule</b>, applied from the loss back to the weights. We check autograd
against <b>finite differences</b> on a real GPT-2 weight (they agree to 7 digits, once a precision trap is avoided), look
at gradient sizes layer by layer, and measure the cost: the backward pass takes about <b>2×</b> the time of the forward
pass, and the saved activations take <b>3×</b> the memory of the weights.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Foundations (derivatives, the chain rule) and Deep Dive
episode 17 (the loss and its gradient). Code: <code>code/d18_backprop</code> (GPT-2 small, about 500 MB; uses about 4 GB
of RAM).</div>""",
}

CONCEPTS = [
    {
        "title": "The chain rule",
        "segment": (8, 40),
        "figures": [{"t": 21.2, "caption": "124,439,808 weights, and for each, how the loss would change: one backward pass "
                                           "computes them all."},
                    {"t": 39.3, "caption": "loss = (tanh(w·x) − 3)²: multiply the local derivatives, from the loss back. By "
                                           "hand −3.760292, autograd −3.760291."}],
        "body": [
            """<p>Training needs, for each of GPT-2's 124,439,808 weights, the gradient: how the loss would change if that
weight moved a little. <b>Backprop</b> computes them all in one backward pass by applying the <b>chain rule</b> step by
step, from the loss back to the weights.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>A tiny example: loss = (tanh(w · x) − 3)², with w = 0.5 and x = 2. The local derivatives are 2(h − 3) =
−4.477 for the square, 1 for “− 3”, 1 − tanh² = 0.420 for tanh, and x = 2 for the multiplication. Their product,
<b>−3.760292</b>, is what autograd computes (−3.760291, float32 rounding).</p>""",
            """<div class="box key"><b class="t">Key idea</b>The gradient of the loss with respect to a weight = the product
of the local derivatives along the path from that weight to the loss (summed over paths).</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Same function, but w = 0 (x = 2). What is the gradient?",
             "answer": "−12.", "why": "tanh(0) = 0: 2(0 − 3) × 1 × (1 − 0) × 2 = −12 (checked with autograd).", "key": {'parts': [{'label': None, 'value': -12, 'tol': 0.5, 'unit': None}]}},
            {"kind": "short", "q": "Why does backprop run from the loss backward rather than from each weight forward?",
             "answer": "One loss and many weights: going backward reuses each intermediate derivative for every weight below "
                       "it, so one pass gives all the gradients.",
             "why": "Going forward would need one pass per weight."},
        ],
    },
    {
        "title": "Checking autograd on GPT-2",
        "segment": (40, 73),
        "figures": [{"t": 58.8, "caption": "Layer 5 MLP, weight [100, 200]: autograd −0.0000200195; finite differences "
                                           "−0.0000200195."},
                    {"t": 72.4, "caption": "The first check failed: the library's built-in loss rounds to float32. Our own "
                                           "float64 loss fixed it."}],
        "body": [
            """<p>Take one real weight in GPT-2's layer 5 MLP, entry [100, 200]. Autograd says its gradient is
<b>−0.0000200195</b>. The slow way: nudge the weight up and down by ε = 0.0001 and compute (loss(w + ε) − loss(w − ε)) /
2ε. In float64 the result is <b>−0.0000200195</b>: the same to 7 digits.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>One trap: the first time, the check failed (0.00238 vs −0.00002). The library's built-in loss converts the
logits to float32, whose rounding (about 5 × 10<sup>−7</sup> on a loss of 4.4) swamps a change of 4 × 10<sup>−9</sup>.
Computing the cross-entropy ourselves, in float64, fixed it.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Finite differences are a slow but independent check of
gradients; they need high precision because the changes are tiny.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "To get every gradient of GPT-2 by finite differences (two forward passes per weight), "
                                    "how many forward passes would you need?",
             "answer": "248,879,616.", "why": "2 × 124,439,808, against one forward and one backward pass with backprop.", "key": {'parts': [{'label': None, 'value': 248879616, 'tol': 0.5, 'unit': None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Check another weight: entry [0, 0] of layer 0's attention "
                                  "(<code>model.transformer.h[0].attn.c_attn.weight</code>).",
             "code": """W = model.transformer.h[0].attn.c_attn.weight
g = W.grad[0, 0].item()
with torch.no_grad():
    W[0, 0] += eps; up = lm_loss(model, ids).item()
    W[0, 0] -= 2 * eps; down = lm_loss(model, ids).item()
    W[0, 0] += eps
print(g, (up - down) / (2 * eps))""",
             "answer": "0.0000307327 and 0.0000307328.", "why": "Agreement to 6–7 digits again."},
        ],
    },
    {
        "title": "Every weight, and gradient sizes",
        "segment": (73, 105),
        "figures": [{"t": 87.6, "caption": "One loss number gives all 124,439,808 parameters a gradient, including all 50,257 "
                                           "embedding rows."},
                    {"t": 103.8, "caption": "Gradient size per layer: ~2 to 4 in the early and middle layers, under 1 in the "
                                            "last two."}],
        "body": [
            """<p>One backward pass gives every one of the 124,439,808 parameters its gradient, from a single loss number.
Even all <b>50,257</b> rows of the token embedding get one, though at most 256 different tokens appear in the 256-token
text: the same matrix also produces the output scores for every vocabulary entry.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Gradient sizes, layer by layer: the early and middle layers get the largest (attention 3.27 in layer 0,
up to 4.23 in layer 3), the last two layers the smallest (0.51–0.73). Nothing vanishes on the way back: layer 0 still
gets a large gradient, consistent with the residual stream's direct path
(episode 13).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Backprop gives every parameter a gradient in one pass; in a
pre-norm transformer the gradients reach the first layers intact.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Only the embedding rows of tokens that appear in the text get a gradient.”",
             "answer": "False (for GPT-2).", "why": "GPT-2 ties the embedding to the output layer, and the softmax gives every "
                                                   "vocabulary entry a gradient.", "key": {'value': False}},
            {"kind": "mc", "q": "Which layers got the smallest gradients in this run?",
             "options": ["Layers 0–1", "Layers 3–4", "Layers 10–11", "All about equal"],
             "answer": "C.", "why": "Attention 0.51 and 0.66, MLP 0.65 and 0.73.", "key": {'choice': 2}},
        ],
    },
    {
        "title": "The cost: time and memory",
        "segment": (105, 145),
        "figures": [{"t": 120.8, "caption": "1,024 tokens on this CPU: forward 0.56 s, backward 1.21 s, about twice as long."},
                    {"t": 136.8, "caption": "Activations saved for the backward pass: 1,443 MiB for one sequence, 3× the "
                                            "475 MiB of weights."}],
        "body": [
            """<p>On this CPU, a forward pass over 1,024 tokens takes <b>0.56 s</b>; the backward pass <b>1.21 s</b>, about
twice as long, because it computes gradients for both the activations (to keep going backward) and the weights (to
update them).</p>""",
            """<p>To run backward, the forward pass must keep its intermediate results. Counting exactly what autograd
saves (excluding the weights themselves): <b>1,443 MiB</b> of activations for one 1,024-token sequence, three times the
475 MiB of float32 weights. In code it is one call, <code>loss.backward()</code>; every parameter then holds its
<code>.grad</code>, ready for the optimizer.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Training costs about 3× a forward pass in compute (1 forward +
~2 backward) and needs memory for activations that grows with batch size and sequence length.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "About how much activation memory would a batch of 8 such sequences need?",
             "answer": "About 11,544 MiB (11.3 GiB).", "why": "8 × 1,443 MiB: each sequence keeps its own activations.", "key": {'parts': [{'label': None, 'value': 11544, 'tol': 230.88, 'unit': None}]}},
            {"kind": "number", "q": "GPT-2 small has 124,439,808 float32 parameters. How many MiB is that?",
             "answer": "About 475 MiB.", "why": "124,439,808 × 4 bytes / 2²⁰ ≈ 474.7.", "key": {'parts': [{'label': None, 'value': 475, 'tol': 9.5, 'unit': None}]}},
        ],
    },
]
