"""Study guide content for How LLMs Work: Deep Dive, episode 11: Sliding Windows and Sparse Attention.

Build:  python framework/study_guide.py deep-dive d11 --video deep-dive/media/videos/d11_scene/1080p60/SlidingWindowVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d11_sliding_window/sliding_window.py (torch 2.14.0, CPU; tiny GPT, 4 layers, 1,500 steps)
or the same pair-counting formula run on the exercise inputs.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 11",
    "title": "Sliding Windows and Sparse Attention",
    "tagline": "Skipping pairs on purpose, and what it costs",
    "duration": "2:48",
    "intro": """<p>This lesson answers one question: what happens if attention skips most pairs? With a <b>sliding
window</b>, each token sees only the last W tokens, so the work grows like T × W instead of T². Stacked layers let
information hop further back, in theory. In practice, a tiny model with a window did <i>better</i> on Shakespeare, but
completely failed a task that needs exact recall from 128 tokens back. That is why models mix windowed and full
attention layers.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 7 (causal mask) and 10
(FlashAttention). Code: <code>code/d11_sliding_window</code> (PyTorch only; trains six tiny models, about 20 minutes on
a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "The window mask",
        "segment": (8, 40),
        "figures": [{"t": 24.0, "caption": "Full causal attention is a triangle of pairs; a sliding window keeps only the "
                                           "last few."},
                    {"t": 39.2, "caption": "A window of 4 on 8 tokens: a band along the diagonal. Window 4,096 at 131,072 "
                                           "tokens: 6.2% of the pairs."}],
        "body": [
            """<p>Full attention compares every token with every earlier one, so the work grows with the square of the
length: at 131,072 tokens that is <b>8.6 billion</b> pairs, per head, per layer. A <b>sliding window</b> lets each token
look only at the last W tokens (itself included).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The mask becomes a band along the diagonal. With W = 4,096 on 131,072 tokens, 528 million pairs remain:
<b>6.2%</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A window turns the quadratic cost T² into roughly T × W:
linear in the length.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many query-key pairs does a window of 128 allow on 1,024 tokens (full causal: "
                                    "524,800)?",
             "answer": "122,944 (about 23%).", "why": "Token i sees min(i + 1, 128) keys; summing over 1,024 tokens gives "
                                                     "122,944."},
            {"kind": "tf", "q": "“With a window of 4, token 7 can attend to token 3.”",
             "answer": "False.", "why": "It sees tokens 4–7 only: 7 − 3 = 4 is not less than W = 4 (look at row 7 of the "
                                       "printed mask)."},
        ],
    },
    {
        "title": "Reaching further by stacking",
        "segment": (40, 64),
        "figures": [{"t": 62.5, "caption": "With a window of 16, each layer adds 15 tokens of reach: 15, 30, 45, 60, "
                                           "measured with gradients."}],
        "body": [
            """<p>Information can hop. With a window of 16, one layer lets the last token reach 15 tokens back; the next
layer reaches through those, and so on: <b>15, 30, 45, 60</b> after 1–4 layers, measured by checking which inputs
receive a gradient from the last output. Mistral 7B uses a window of 4,096 with 32 layers: in theory, a reach of about
131,072 tokens.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Reach grows with layers × (W − 1), like a receptive field in
a convolutional network, but reach is only a possibility, not a guarantee.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A model has 12 layers with a window of 512. What is its theoretical reach?",
             "answer": "6,132 tokens.", "why": "12 × (512 − 1) = 6,132, by the same rule as 4 × 15 = 60."},
            {"kind": "short", "q": "Why measure reach with gradients rather than by looking at attention weights?",
             "answer": "A gradient is nonzero exactly when an input can influence the output through any path, across "
                       "all layers; attention weights show only one layer's direct links.",
             "why": "The code backpropagates from the last output and counts which inputs receive a gradient."},
        ],
    },
    {
        "title": "Two experiments, two answers",
        "segment": (64, 110),
        "figures": [{"t": 87.0, "caption": "Shakespeare, 256 characters: full 1.780, window 64 1.750, window 16 1.693. "
                                           "The windows did better."},
                    {"t": 109.0, "caption": "Copy 128 symbols: full attention 0.000; both windows 2.775, pure guessing "
                                            "(ln 16 = 2.77)."}],
        "body": [
            """<p>The same tiny GPT is trained for 1,500 steps on 256-character Shakespeare texts. Validation loss: full
attention <b>1.780</b>, window 64 <b>1.750</b>, window 16 <b>1.693</b>, with 44% and 12% of the pairs. The windows did
better: for spelling out Shakespeare character by character, nearby characters matter most, and the window is a helpful
shortcut for a small model trained briefly.</p>""",
            """<p>Now a task that needs long range: 128 random symbols, then the same 128 again. Full attention learns
to copy perfectly: loss <b>0.000</b> on the second half. Both windows are stuck at <b>2.775</b>, pure guessing
(ln 16 = 2.77). Even window 64, whose reach on paper is 252 tokens, never learned to relay the symbols through its
layers.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Windows are cheap and fine for local patterns, but lose exact
recall far back. One benchmark can hide what another reveals.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which task would most likely suffer from a window of 16?",
             "options": ["Predicting the next letter of a word", "Answering a question about the first page of a long "
                         "report", "Closing a bracket opened 3 tokens earlier", "Choosing between “a” and “an”"],
             "answer": "B.", "why": "It needs exact information from far back; the others are local."},
            {"kind": "number", "q": "In the copy task, what loss does pure guessing among 16 symbols give?",
             "answer": "About 2.77.", "why": "ln 16 ≈ 2.773; both windowed models scored 2.775."},
            {"kind": "tf", "q": "“If a window's theoretical reach covers a distance, the model will learn to use it.”",
             "answer": "False.", "why": "Window 64 with 4 layers can reach 252 tokens in theory, but scored 2.775 on the "
                                       "copy task, no better than guessing."},
        ],
    },
    {
        "title": "Mixing, sparse patterns, and the cache",
        "segment": (110, 159),
        "figures": [{"t": 136.0, "caption": "Other sparse patterns: global tokens, strides. Always: choose which pairs to "
                                            "compute."},
                    {"t": 158.0, "caption": "The window mask in one line: not in the future, and less than W back."}],
        "body": [
            """<p>That is why models <b>mix</b>: Gemma 2, for example, alternates sliding-window layers with
full-attention layers. Sliding windows are one kind of <b>sparse attention</b>; other patterns add a few <b>global
tokens</b> that see, and are seen by, everything, or skip with a <b>stride</b>. The idea is always to choose which pairs
are worth computing.</p>""",
            """<p>At inference there is a bonus: with a window, the KV cache only needs the last W tokens, a fixed-size
<b>rolling buffer</b> however long the text grows.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Windowed layers handle the local work cheaply; a few full
layers keep long-range recall.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With a window of 4,096, how many tokens' keys and values must a layer keep in its "
                                    "cache while generating token 100,000?",
             "answer": "4,096.", "why": "Only the last W tokens can be attended to; older entries can be dropped."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Print <code>window_mask(10, 3).int()</code>. How many ones are "
                                  "in it?",
             "code": """print(window_mask(10, 3).int(), window_mask(10, 3).sum())""",
             "answer": "27.", "why": "Rows 0 and 1 have 1 and 2 ones; the other 8 rows have 3: 1 + 2 + 24 = 27."},
        ],
    },
]
