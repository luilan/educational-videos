"""Study guide content for How LLMs Work: Deep Dive, episode 13: The Residual Stream.

Build:  python framework/study_guide.py deep-dive d13 --video deep-dive/media/videos/d13_scene/1080p60/ResidualStreamVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d13_residual_stream/residual_stream.py (GPT-2 small, real weights, 512 tokens of Tiny
Shakespeare; an 8-layer tiny GPT trained 1,500 steps; transformers 4.57.1, torch 2.14.0, CPU).
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 13",
    "title": "The Residual Stream",
    "tagline": "Every layer adds; nothing is replaced",
    "duration": "2:51",
    "intro": """<p>This lesson answers one question: how do the layers of a transformer pass information along? Not by
replacing it: each attention and MLP block <b>adds</b> its output to a running vector, the <b>residual stream</b>. In
GPT-2, most updates are 15–34% of the stream's size, so deleting a middle layer barely matters; without the addition,
an 8-layer tiny model cannot learn anything beyond letter frequencies.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episodes 7–8 (the transformer block) and
Foundations (gradients). Code: <code>code/d13_residual_stream</code> (GPT-2 small, about 500 MB, plus two tiny models,
about 10 minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "A stream that every layer writes into",
        "segment": (8, 36),
        "figures": [{"t": 20.5, "caption": "x ← x + attention(x), then x ← x + MLP(x): the running sum is the residual "
                                           "stream."},
                    {"t": 35.0, "caption": "One 768-number vector per token flows from embedding to output; 24 "
                                           "sub-blocks read it and write corrections."}],
        "body": [
            """<p>Inside a transformer, no layer replaces what came before. Each one adds: <code>x = x + attention(x)</code>,
then <code>x = x + MLP(x)</code>. The running sum is the <b>residual stream</b>, the model's shared workspace.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Picture one vector per token (768 numbers in GPT-2) flowing from the embedding to the output. Along the
way sit 24 sub-blocks, an attention and an MLP in each of the 12 layers. Each one <b>reads</b> the stream (through a
normalization) and <b>writes</b> a correction back into it.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The residual stream is a sum: embedding + every block's
output. Each block edits it; none overwrites it.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many sub-blocks write into GPT-2 small's residual stream?",
             "answer": "24.", "why": "12 layers × (1 attention + 1 MLP).", "key": {'parts': [{'label': None, 'value': 24, 'tol': 0.5, 'unit': None}]}},
            {"kind": "short", "q": "Write the residual stream after two layers as a sum of terms.",
             "answer": "x₂ = embedding + attn₀ + mlp₀ + attn₁ + mlp₁ (each computed from the stream as it was at that "
                       "point).",
             "why": "Every block adds its output; nothing is subtracted or replaced."},
        ],
    },
    {
        "title": "How big it is, and how big the edits are",
        "segment": (36, 80),
        "figures": [{"t": 59.0, "caption": "The stream's size entering each layer: 4.6 (embedding), 49.5 after layer 0, up "
                                           "to 216.1 and 554.9 at the end."},
                    {"t": 79.5, "caption": "Each update as a fraction of the stream: 15–34% in layers 1–9; layer 0 and "
                                           "the last layers write more."}],
        "body": [
            """<p>In GPT-2 the stream starts small: an average size of <b>4.6</b>, just the embedding. After layer 0 it
is about 50, then it grows steadily to <b>216</b> entering the last layer and 555 at the end. The first token is
different: its stream grows past <b>3,000</b> (3,067 entering layer 11), the attention sink of episode 12, a huge vector
every head can find.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Compared with the stream they are added to, the updates in layers 1–9 are only <b>15% to 34%</b> of its
size: edits, not rewrites. Layer 0 writes 7.1× (attention) and 5.3× (MLP) the embedding: it builds the stream. The last
layers make bigger changes (up to 1.41×) to prepare the output.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Most layers nudge the stream; the first builds it and the
last ones reshape it for the output.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "By what factor does the average token's stream grow from entering layer 1 (49.5) to "
                                    "entering layer 11 (216.1)?",
             "answer": "About 4.4.", "why": "216.1 / 49.5 ≈ 4.37.", "key": {'parts': [{'label': None, 'value': 4.4, 'tol': 0.088, 'unit': None}]}},
            {"kind": "tf", "q": "“In GPT-2's middle layers, each MLP replaces most of the stream.”",
             "answer": "False.", "why": "Its output is about a quarter of the stream's size (0.24–0.34 in layers 1–9), "
                                       "and it is added, not substituted.", "key": {'value': False}},
        ],
    },
    {
        "title": "Delete a layer; remove the stream",
        "segment": (80, 136),
        "figures": [{"t": 98.3, "caption": "Delete one layer: middle layers 3.74–4.33 (baseline 4.13); the first 7.95, the "
                                           "last 6.12."},
                    {"t": 135.0, "caption": "8-layer tiny GPT: with residuals 2.46 → 1.69; without, stuck at 3.36 (letter "
                                            "frequencies alone: 3.35)."}],
        "body": [
            """<p>Delete a whole layer from GPT-2, letting the stream flow past it. The baseline loss on this Shakespeare
text is <b>4.13</b>. Without any one middle layer, the loss stays between <b>3.74 and 4.33</b>; on this text some
deletions even help. Without the first layer: <b>7.95</b>; without the last: <b>6.12</b>. Because each layer only adds,
deleting one removes one edit; in a plain chain, every layer would depend on the exact output of the one before.</p>""",
            """<p>Now train the same 8-layer tiny GPT with <code>x = x + block(x)</code> or just <code>x = block(x)</code>.
With the residual: 2.46 after 100 steps, <b>1.69</b> after 1,500. Without it: stuck at <b>3.36</b> from the start,
almost exactly the 3.35 you get by predicting from letter frequencies alone. It learned nothing else.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Residual connections make layers robust to removal and,
above all, make deep networks trainable.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which deletion hurt GPT-2 most on this text?",
             "options": ["Layer 6", "Layer 0", "Layer 11", "Layer 4"],
             "answer": "B.", "why": "Loss 7.95 without layer 0, against 6.12 without layer 11 and 3.74–4.33 for middle "
                                   "layers.", "key": {'choice': 1}},
            {"kind": "short", "q": "The model without residuals reached 3.36, almost exactly the letter-frequency loss "
                                   "(3.35). What does that tell you about what it learned?",
             "answer": "Only how common each character is, ignoring the context entirely: its output does not depend on the "
                       "input in any useful way.",
             "why": "A model that predicts the same distribution everywhere scores exactly the unigram loss."},
        ],
    },
    {
        "title": "The gradient highway, and the code",
        "segment": (136, 160),
        "figures": [{"t": 148.3, "caption": "d/dx [x + f(x)] = 1 + f′(x): the 1 carries the learning signal straight back "
                                            "to the early layers."},
                    {"t": 159.4, "caption": "Two plus signs; removing them was the whole experiment."}],
        "body": [
            """<p>Why does training fail without the addition? The derivative of <code>x + f(x)</code> is
<b>1 + f′(x)</b>. The 1 carries the learning signal straight back through all the blocks, so early layers get a usable
gradient however deep the network is. Without it, the gradient must pass through every block's derivative in turn,
and can shrink or scramble on the way.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>The residual stream is also a gradient highway: the identity
path lets every layer learn.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Suppose each of 8 blocks multiplies the gradient by 0.5. What fraction reaches the "
                                    "first block without residuals? With residuals, at least what comes through the "
                                    "identity path?",
             "answer": "0.5⁸ ≈ 0.0039 without; the full gradient (factor 1) through the identity path with residuals.",
             "why": "Products of small factors vanish; the residual adds a path whose factor is exactly 1.", "key": {'parts': [{'label': 'without residuals', 'value': 0.0039, 'tol': 7.8e-05, 'unit': None}, {'label': 'with residuals (at least)', 'value': 1, 'tol': 0.5, 'unit': None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> With <code>run</code> from <code>residual_stream.py</code>, "
                                  "delete layers 3 and 6 together, then layers 2, 3, 6 and 10. Is the damage additive?",
             "code": """for skip in ({3, 6}, {2, 3, 6, 10}):
    lg, _, _ = run(ids, skip=skip)
    print(sorted(skip), F.cross_entropy(lg[0, :-1], ids[0, 1:]).item())""",
             "answer": "Layers 3 and 6: 3.71, lower than either alone (3.83, 3.74). Four middle layers: 4.32, still close "
                       "to the baseline 4.13.",
             "why": "Not additive: the remaining layers' edits interact. But with residuals, removing several middle "
                    "layers degrades gracefully on this text."},
        ],
    },
]
