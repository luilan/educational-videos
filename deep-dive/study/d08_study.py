"""Study guide content for How LLMs Work: Deep Dive, episode 8: The Attention Matrix, Entry by Entry.

Build:  python framework/study_guide.py deep-dive d08 --video deep-dive/media/videos/d08_scene/1080p60/AttentionMatrixVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d08_attention_matrix/attention_matrix.py (GPT-2 small, real weights, layer 4, head 3,
counting from zero; transformers 4.57.1, torch 2.14.0, CPU) or the same tensors used in the exercises.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 8",
    "title": "The Attention Matrix, Entry by Entry",
    "tagline": "One GPT-2 head, computed by hand",
    "duration": "2:55",
    "intro": """<p>This lesson answers one question: where does each number in an attention diagram come from? One head
of the real GPT-2 is recomputed by hand on <i>The cat sat on the mat because it was tired</i>: queries, keys and values;
one score as a sum of 64 products, divided by 8; the mask and the softmax; the weighted sum of values; and the 12 heads
joined back together. Every step matches the <code>transformers</code> library exactly.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episodes 5–6 (attention) and Deep Dive
episode 7 (the causal mask). Code: <code>code/d08_attention_matrix</code> (GPT-2 small, about 500 MB).</div>""",
}

CONCEPTS = [
    {
        "title": "Queries, keys and values",
        "segment": (8, 42),
        "figures": [{"t": 40.5, "caption": "One matrix turns each normalized 768-number token vector into a query, a key "
                                           "and a value, each split into 12 heads of 64."}],
        "body": [
            """<p>The sentence has ten tokens. We follow <b>layer 4, head 3</b> of GPT-2 small (counting from zero). Each
token's 768-number vector is first normalized (LayerNorm), then multiplied by one matrix that produces its <b>query</b>,
<b>key</b> and <b>value</b>, 768 numbers each. Each is split into <b>12 heads of 64 numbers</b>; head 3 uses numbers
192–255 of each.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>A head is just a 64-number slice of the queries, keys and
values; all 12 heads are computed by the same matrix multiplication.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "GPT-2's query-key-value matrix maps 768 numbers to 3 × 768. How many parameters does "
                                    "it have, including its 2,304 biases?",
             "answer": "1,771,776.", "why": "768 × 2,304 = 1,769,472 weights, plus 2,304 biases."},
            {"kind": "number", "q": "How many numbers per head would a model with 1,024-number vectors and 16 heads "
                                    "have?",
             "answer": "64.", "why": "1,024 / 16 = 64, the same head size as GPT-2."},
        ],
    },
    {
        "title": "One entry, one row",
        "segment": (42, 73),
        "figures": [{"t": 56.0, "caption": "Query of “it” times key of “cat”: 64 products add up to 4.55; divided by 8, "
                                           "0.57."},
                    {"t": 71.5, "caption": "The whole row for “it”: mask the future, apply softmax, and 84% of the "
                                           "attention goes to “cat”."}],
        "body": [
            """<p>One entry: the query of <i>it</i> against the key of <i>cat</i>. Multiply the 64 pairs of numbers and
add them up: <b>4.55</b>. Divide by 8, the square root of 64: <b>0.57</b>. That is one score.</p>""",
            """<p>Do the same for every token up to <i>it</i>: the scores range from −4.33 (<i>the</i>) to +0.57
(<i>cat</i>). The future (<i>was</i>, <i>tired</i>) is masked with −∞, softmax turns the row into weights, and
<b>0.837</b> goes to <i>cat</i>. In this head, <i>it</i> points at the cat.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Each entry is a dot product divided by √(head size); each row
is a softmax over the tokens a query is allowed to see.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "If “it” could see only “The” (score −2.14) and “cat” (score 0.57), what weight would "
                                    "“cat” get?",
             "answer": "About 0.938.", "why": "e<sup>0.57</sup> / (e<sup>0.57</sup> + e<sup>−2.14</sup>) ≈ 0.938; fewer "
                                             "competitors, more weight."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Using <code>qh</code> and <code>kh</code> from "
                                  "<code>attention_matrix.py</code>, compute the same score with a single matrix product "
                                  "and check it.",
             "code": """s = qh[HEAD] @ kh[HEAD].T / 8
print(s[i, j], (qh[HEAD, i] * kh[HEAD, j]).sum() / 8)""",
             "answer": "Both print 0.569 (shown as 0.57).",
             "why": "A matrix product computes every query-key dot product at once; entry (i, j) is exactly the hand "
                    "computation."},
        ],
    },
    {
        "title": "The matrix, and why divide by 8",
        "segment": (73, 108),
        "figures": [{"t": 87.2, "caption": "The full 10 × 10 matrix: 45 entries masked, most rows pointing back at “cat”; "
                                           "identical to transformers."},
                    {"t": 107.0, "caption": "Without dividing by 8, the largest weight per row averages 0.98 instead of "
                                            "0.68: almost one-hot."}],
        "body": [
            """<p>Repeat for every query and you get the full matrix: 10 × 10 = 100 entries, 45 of them masked. The
hand-computed weights match the library's with a largest difference of <b>0.0</b>. In this head, most words look back at
<i>cat</i>, the subject of the sentence.</p>""",
            """<p>Why divide by 8? The dot product of two random 64-number vectors spreads out by about 8 (the code
measures a standard deviation of 8.00, which is √64). Without the division, softmax becomes nearly all-or-nothing: the
largest weight in each row averages <b>0.98</b> instead of <b>0.68</b>. Almost one-hot, so the gradients for every other
token nearly vanish, and training is much harder.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Dividing by √(head size) keeps scores in a range where
softmax stays soft, whatever the head size.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A model uses heads of 128 numbers. By what does it divide each score?",
             "answer": "About 11.31.", "why": "√128 ≈ 11.31."},
            {"kind": "tf", "q": "“Without the division by 8, the row for “it” in this head would still give “cat” about "
                                "84%.”",
             "answer": "False.", "why": "Scores 8 times larger make softmax sharper: “cat” gets 1.000 (checked by running "
                                       "it)."},
            {"kind": "number", "q": "In a causal 10 × 10 attention matrix, how many entries are not masked?",
             "answer": "55.", "why": "10 + 9 + … + 1 = 55; the other 45 are masked."},
        ],
    },
    {
        "title": "Values, heads, and scale",
        "segment": (108, 167),
        "figures": [{"t": 123.5, "caption": "The weights mix the values; 12 heads × 64 = 768 numbers, mixed by one 768 × "
                                            "768 matrix."},
                    {"t": 141.5, "caption": "Other heads: one looks only at the previous token; in 92 of 144, most "
                                            "attention goes to the first token (schematic on the right)."}],
        "body": [
            """<p>The weights then mix the <b>values</b>: the output for <i>it</i> is 0.837 × value(cat) + 0.056 ×
value(The) + …, 64 numbers, mostly cat's. The 12 heads' outputs are joined back into 768 numbers and mixed by one more
768 × 768 matrix. Again identical to the library: a difference of 0.0.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Other heads have other habits. In <b>layer 4, head 11</b>, every token puts all of its attention (1.00)
on the token just before it. In <b>92 of the 144</b> heads, more than half the attention goes to the very first token
(the attention-sinks episode explains why).</p>""",
            """<p>GPT-2 builds 12 × 12 = <b>144</b> of these matrices on every forward pass; at 1,024 tokens that is
<b>150,994,944</b> weights. Keeping that memory in check is the subject of the FlashAttention episode.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A head reads the values of the tokens it attends to; the output
matrix then combines what all the heads found.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put one head's steps in order: <i>softmax · multiply by the values · dot products of "
                                   "queries and keys · divide by √64 · mask the future</i>.",
             "answer": "dot products → divide by √64 → mask the future → softmax → multiply by the values.",
             "why": "This is the order in the code (the mask must come before the softmax)."},
            {"kind": "number", "q": "How many attention weights would GPT-2 compute in one forward pass at 512 tokens?",
             "answer": "37,748,736.", "why": "144 × 512 × 512; a quarter of the 1,024-token figure, since the matrix grows "
                                            "with the square of the length."},
            {"kind": "short", "q": "What does a “previous-token head” make possible for the layers above it?",
             "answer": "Each position now carries information about the token just before it, so later heads can match "
                       "pairs or patterns of tokens, not just single tokens.",
             "why": "It is a building block of induction heads (Deep Dive part 10)."},
        ],
    },
]
