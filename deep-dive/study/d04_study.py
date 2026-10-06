"""Study guide content for How LLMs Work: Deep Dive, episode 4: Why Attention Needs Position.

Build:  python framework/study_guide.py deep-dive d04 --video deep-dive/media/videos/d04_scene/1080p60/PositionVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d04_position/position.py (GPT-2 small, real weights; transformers 4.57.1, torch 2.14.0,
CPU) or the same functions run on the exercise sentences.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 4",
    "title": "Why Attention Needs Position",
    "tagline": "Proof, with real GPT-2 weights, that attention is blind to order",
    "duration": "2:45",
    "intro": """<p>This lesson answers one question: why do transformers need position information at all?
<b>Attention</b> is a weighted average whose weights come from dot products, so shuffling the input just shuffles the
output (<b>permutation equivariance</b>). With GPT-2's real weights and its position embeddings switched off, “dog
bites man” and “man bites dog” give identical vectors. Position embeddings, and to a lesser degree the causal mask,
bring order back; GPT-2's learned table, however, stops at 1,024 positions.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episodes 4–6 (position, attention) and
Foundations F02 (dot product, cosine similarity). Code: <code>code/d04_position</code> (GPT-2 small, about 500 MB).</div>""",
}

CONCEPTS = [
    {
        "title": "Attention is blind to order",
        "segment": (8, 42),
        "figures": [{"t": 41.4, "caption": "Shuffle the tokens and the attention weights are the same numbers, just "
                                           "shuffled."}],
        "body": [
            """<p><i>Dog bites man</i> and <i>man bites dog</i>: same three words, opposite news. But attention has no
idea of order. Each token's output is a <b>weighted average of value vectors</b>, and the weights come from <b>dot
products</b> between queries and keys. Shuffle the input tokens, and you get the same dot products, just shuffled; a
weighted sum does not care about order. This property is called <b>permutation equivariance</b>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Without extra information, attention treats a sentence as a
bag of tokens: reorder the input, and the outputs are simply reordered.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "In your own words: why does reordering the inputs only reorder the outputs of "
                                   "attention?",
             "answer": "Each output is computed from dot products with all tokens and a weighted sum over them; neither "
                       "operation depends on where a token sits in the list.",
             "why": "Sums and dot products treat their inputs as a set."},
            {"kind": "tf", "q": "“Permutation equivariance means the outputs are identical for every order.”",
             "answer": "False.", "why": "The outputs are the same vectors in a different order: each token's own output "
                                       "does not change, only its position in the list.", "key": {'value': False}},
        ],
    },
    {
        "title": "The experiment",
        "segment": (42, 82),
        "figures": [{"t": 68.4, "caption": "GPT-2, position embeddings off, every token seeing every other: dog vs dog "
                                           "1.000000."},
                    {"t": 80.6, "caption": "Position embeddings back on: dog vs dog drops to 0.96."}],
        "body": [
            """<p>Take GPT-2 small, all 12 layers. Switch off its position embeddings, let every token see every other,
and feed in both sentences. The final vector for <i>dog</i> in the first sentence and in the second: cosine similarity
<b>1.000000</b>. The same for <i>man</i>. The biggest difference anywhere is 0.0002, just rounding. The model cannot
tell who bit whom.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Switch the position embeddings back on, and dog in position 1 is no longer the same as dog in position
3: the similarity drops to <b>0.96</b>. Now order can matter.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Position information is the only thing that lets the model
tell “dog bites man” from “man bites dog”.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "With position embeddings off and no mask, what would GPT-2 do with “the cat chased the "
                                "mouse” and “the mouse chased the cat”?",
             "options": ["Clearly different vectors for “cat”", "Identical vectors for “cat”, just reordered",
                         "An error", "Random vectors"],
             "answer": "B.", "why": "Checked by running it: cosine similarity 1.000000.", "key": {'choice': 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Using <code>last_layer</code> from <code>position.py</code>, "
                                  "compare “ cat” (token 1 in the first sentence, token 4 in the second) with positions "
                                  "off and on. Why is the “on” number so close to 1, unlike dog's 0.96?",
             "code": """a = tok(" the cat chased the mouse", return_tensors="pt").input_ids
b = tok(" the mouse chased the cat", return_tensors="pt").input_ids
for pos in (False, True):
    oa, ob = last_layer(a, pos), last_layer(b, pos)
    print(pos, torch.nn.functional.cosine_similarity(oa[1], ob[4], dim=0).item())""",
             "answer": "Off: 1.000000. On: 0.999767, different, but barely by this measure.",
             "why": "GPT-2's final vectors share a few very large dimensions, so cosine similarity between them is high "
                    "overall; it is a blunt measure here. The “largest difference” printed by position.py shows the change "
                    "more clearly."},
        ],
    },
    {
        "title": "The causal mask leaks order",
        "segment": (82, 106),
        "figures": [{"t": 105.3, "caption": "With a causal mask and no position embeddings, the two dogs differ: 0.987."}],
        "body": [
            """<p>Models like GPT use a <b>causal mask</b>: each token sees only the tokens before it. That alone leaks
some order: the first token sees only itself, the last sees everything. With the mask on and no position embeddings,
the two dogs are no longer identical: <b>0.987</b>. Some models are trained this way, without explicit positions, but
it is an indirect signal.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>A causal mask breaks the symmetry a little; explicit position
information makes order a first-class input.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In a 4-token causal mask (as on screen), how many of the 16 query-key pairs are "
                                    "allowed?",
             "answer": "10.", "why": "1 + 2 + 3 + 4 = 10: each token sees itself and all earlier tokens.", "key": {'parts': [{'label': None, 'value': 10, 'tol': 0.5, 'unit': None}]}},
            {"kind": "short", "q": "With a causal mask, why does the first token's output depend on its position?",
             "answer": "It can only attend to itself, while a later token can attend to several; so the set of tokens it "
                       "mixes depends on where it sits.",
             "why": "Position changes what each token can see, even without position embeddings."},
        ],
    },
    {
        "title": "What GPT-2 learned, and its limit",
        "segment": (106, 163),
        "figures": [{"t": 129.5, "caption": "Similarity of GPT-2's position vector 100 to every position 0–400: smooth, "
                                            "and negative far away."},
                    {"t": 148.9, "caption": "A learned table stops at 1,024; rotating queries and keys encodes relative "
                                            "distance."}],
        "body": [
            """<p>GPT-2 learned one vector per position: <b>1,024</b> of them, each with 768 numbers, from scratch. A
smooth pattern emerges: position 100 is almost identical to 101 (0.999), still very close to 110 (0.98), less so to
150 (0.76), and at 300 the similarity is negative (−0.23).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>But a learned table has a hard limit: GPT-2 has no vector for position 1,025, so it cannot read
anything longer. Modern models encode position by <b>rotating</b> the queries and keys, so attention sees
<b>relative distance</b> and the range can be stretched (next episode).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Learned absolute positions work, but are capped by the table
size and say nothing directly about distance between tokens.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many numbers are in GPT-2's position-embedding table?",
             "answer": "786,432.", "why": "1,024 positions × 768 numbers.", "key": {'parts': [{'label': None, 'value': 786432, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“GPT-2 can process a 2,000-token document in one pass.”",
             "answer": "False.", "why": "Its learned position table has 1,024 entries; there is no vector for position "
                                       "1,025 or beyond.", "key": {'value': False}},
        ],
    },
]
