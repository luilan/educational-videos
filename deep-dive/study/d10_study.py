"""Study guide content for How LLMs Work: Deep Dive, episode 10: FlashAttention.

Build:  python framework/study_guide.py deep-dive d10 --video deep-dive/media/videos/d10_scene/1080p60/FlashAttentionVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d10_flash_attention/flash_attention.py (torch 2.14.0, CPU, 8 threads; memory measured as
peak resident memory on Linux) or the same formulas run on the exercise inputs.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 10",
    "title": "FlashAttention",
    "tagline": "Same math, less memory",
    "duration": "2:46",
    "intro": """<p>This lesson answers one question: how can attention avoid storing its huge matrix of scores?
<b>FlashAttention</b> reads the scores in tiles and uses an <b>online softmax</b> (a running max, sum and output, rescaled
whenever a larger score appears). The result is exactly standard attention, but the biggest piece ever stored is one
tile. Measured: 11 MiB instead of 1,557 MiB at 4,096 tokens, and 12 times faster.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 7–8 (the mask; one head, entry by
entry). Code: <code>code/d10_flash_attention</code> (PyTorch only, no downloads; Linux for the memory measurement).</div>""",
}

CONCEPTS = [
    {
        "title": "The matrix, and the obstacle",
        "segment": (8, 43),
        "figures": [{"t": 29.5, "caption": "Standard attention stores every score: 1.5 GiB for 4,096 tokens and 12 heads, "
                                           "in one layer."},
                    {"t": 42.0, "caption": "Softmax needs the largest score and the total of the whole row before any "
                                           "weight is known."}],
        "body": [
            """<p>Standard attention builds the full matrix of scores, every token against every token. For 4,096 tokens
and 12 heads, that is about <b>1.5 GiB</b> (measured) for a single layer; at 131,072 tokens it would be <b>768 GiB</b>.
FlashAttention computes exactly the same result without ever storing that matrix.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The obstacle is <b>softmax</b>: to turn a row of scores into weights you need the row's largest score
(for numerical stability) and its total, before any single weight is known. So it seems the whole row must be kept.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The score matrix grows with the square of the length; storing
it is what makes long contexts run out of memory.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many GiB would the score matrices of one layer take for 8,192 tokens and 12 "
                                    "heads, in float32?",
             "answer": "3 GiB.", "why": "8,192² × 12 × 4 bytes = 3,221,225,472 bytes = 3 GiB.", "key": {'parts': [{'label': None, 'value': 3, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“Doubling the length doubles the memory of the score matrix.”",
             "answer": "False.", "why": "It is T × T: doubling T multiplies it by 4 (99 → 386 → 1,557 MiB in the "
                                       "measurements).", "key": {'value': False}},
        ],
    },
    {
        "title": "Online softmax",
        "segment": (43, 67),
        "figures": [{"t": 66.0, "caption": "Eight scores in two chunks: running max 4.0, sum 1.9067, output 36.2742, the "
                                           "same as the full softmax."}],
        "body": [
            """<p>Read the row in chunks and keep three running numbers: the <b>largest score</b> so far (m), the
<b>sum</b> so far (l) and the <b>output</b> so far. With scores 1, 3, 0.5, 2 | 4, 1.5, 0, 2.5 and values 0, 10, …, 70:
after the first chunk, m = 3.0, l = 1.5853 and the output is 14.3052. The second chunk brings a 4, so the old sum and
output are <b>rescaled</b> by e<sup>3 − 4</sup> before the new terms are added: m = 4.0, l = 1.9067, output
<b>36.2742</b>, exactly the full softmax's answer.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>A softmax-weighted average can be computed in one pass, chunk
by chunk: when the max grows, multiply what you have by e<sup>old max − new max</sup>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Scores 1, 2 | 3 with values 0, 10 | 20. After the first chunk (max 2), what is the "
                                    "output so far? And the final output?",
             "answer": "7.3106 after the first chunk; 15.7521 at the end.",
             "why": "Chunk 1: (e<sup>−1</sup>·0 + 1·10) / (e<sup>−1</sup> + 1) = 7.3106. Then rescale by e<sup>2 − 3</sup> "
                    "and add e<sup>0</sup>·20: the same as the full softmax, 15.7521.", "key": {'parts': [{'label': 'after first chunk', 'value': 7.3106, 'tol': 0.001, 'unit': None}, {'label': 'final', 'value': 15.7521, 'tol': 0.001, 'unit': None}]}},
            {"kind": "short", "q": "Why subtract the running max at all, instead of just adding up e<sup>score</sup>?",
             "answer": "To avoid overflow: e<sup>score</sup> for large scores is too big for floating point. Subtracting "
                       "the max keeps every exponent at most 0.",
             "why": "The standard softmax does the same; the online version just updates the max as it goes."},
        ],
    },
    {
        "title": "Tiles, and the same math",
        "segment": (67, 98),
        "figures": [{"t": 89.0, "caption": "Blocks of 64 queries × 64 keys; future blocks skipped, diagonal blocks masked; "
                                           "the largest piece stored is 16 KiB."},
                    {"t": 97.3, "caption": "The 20-line tiled version matches standard attention to within 4.8 × "
                                           "10⁻⁷."}],
        "body": [
            """<p>FlashAttention applies the online softmax to the whole matrix, in <b>tiles</b>: a block of 64 queries
against a block of 64 keys; compute their scores, update each row's running max, sum and output, move on. Blocks
entirely in the future are skipped; only the diagonal blocks need the mask. The biggest piece ever stored is 64 × 64
numbers, <b>16 KiB</b>, instead of 1,024 × 1,024 = <b>4 MiB</b> for one head of 1,024 tokens.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>It is not an approximation: the episode's 20-line version matches standard attention to within
<b>4.8 × 10<sup>−7</sup></b>, float rounding.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Tiling plus online softmax gives exact attention with memory
that no longer grows with T².</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With 1,024 tokens and blocks of 64, how many query-key blocks does causal tiled "
                                    "attention compute (out of 256)?",
             "answer": "136.", "why": "16 blocks per side; the lower triangle including the diagonal is 16 × 17 / 2 = 136.", "key": {'parts': [{'label': None, 'value': 136, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "How large is one tile of scores with blocks of 128, in float32?",
             "answer": "64 KiB.", "why": "128 × 128 × 4 bytes = 65,536 bytes.", "key": {'parts': [{'label': None, 'value': 64, 'tol': 0.5, 'unit': None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Run <code>tiled</code> from <code>flash_attention.py</code> "
                                  "with blocks of 128 and 16. Does the output change?",
             "code": """for B in (128, 16):
    b, biggest = tiled(q, k, v, B=B)
    print(B, (a - b).abs().max().item(), biggest)""",
             "answer": "No: the difference stays at float rounding (around 10⁻⁷) whatever the block size; only the largest "
                       "tile changes (16,384 or 256 numbers).",
             "why": "The block size is a performance choice, not part of the math."},
        ],
    },
    {
        "title": "Measured, and why it is faster",
        "segment": (98, 144),
        "figures": [{"t": 114.5, "caption": "12 heads: standard 99, 386, 1,557 MiB; fused 8, 4, 11 MiB. At 4,096 tokens, "
                                            "0.74 s → 0.06 s."},
                    {"t": 131.0, "caption": "On a GPU, moving data to and from main memory is the slow part; tiles stay "
                                            "in fast on-chip memory."}],
        "body": [
            """<p>With 12 heads, against PyTorch's fused attention kernel (<code>scaled_dot_product_attention</code>,
which also works block by block), the extra memory is:</p>
<table><tr><th>tokens</th><th>standard</th><th>fused</th></tr>
<tr><td>1,024</td><td>99 MiB</td><td>8 MiB</td></tr>
<tr><td>2,048</td><td>386 MiB</td><td>4 MiB</td></tr>
<tr><td>4,096</td><td>1,557 MiB</td><td>11 MiB</td></tr></table>
<p>and at 4,096 tokens it is <b>12 times faster</b> (0.74 s → 0.06 s, CPU). The outputs agree to 7 × 10<sup>−7</sup>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Why faster, if the math is the same? On a GPU the slow part is <b>moving data</b>, not multiplying it.
Small tiles stay in fast on-chip memory, while the big matrix would travel back and forth to the much slower main
memory. What does not change: every query is still compared with every earlier key, so the <b>work</b> still grows
with the square of the length. Cutting that requires skipping pairs (next episode).</p>""",
            """<div class="box key"><b class="t">Key idea</b>FlashAttention saves memory and memory traffic, not
arithmetic: the work is still quadratic.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which statement about FlashAttention is true?",
             "options": ["It approximates attention to save time", "It reduces the work from T² to T",
                         "It computes exact attention without storing the T × T matrix",
                         "It only works without a causal mask"],
             "answer": "C.", "why": "Same result (to rounding), memory per tile only, and the causal mask is handled by "
                                   "skipping and masking blocks.", "key": {'choice': 2}},
            {"kind": "number", "q": "From 2,048 to 4,096 tokens, by what factor did standard attention's measured memory "
                                    "grow?",
             "answer": "About 4 (386 → 1,557 MiB, ×4.03).", "why": "The score matrix is T × T.", "key": {'parts': [{'label': None, 'value': 4, 'tol': 0.5, 'unit': None}]}},
        ],
    },
]
