"""Study guide content for How LLMs Work: Deep Dive, episode 9: Multi-Query and Grouped-Query Attention.

Build:  python framework/study_guide.py deep-dive d09 --video deep-dive/media/videos/d09_scene/1080p60/GQAVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d09_gqa/gqa.py (Qwen2.5-0.5B-Instruct in bfloat16, GPT-2 small, and a tiny GPT trained
2,000 steps on Tiny Shakespeare; transformers 4.57.1, torch 2.14.0, CPU) or from the cache-size formula it uses.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 9",
    "title": "Multi-Query and Grouped-Query Attention",
    "tagline": "Sharing keys and values to shrink the KV cache",
    "duration": "2:42",
    "intro": """<p>This lesson answers one question: how do modern models keep the KV cache small? Instead of one key and
value head per query head (<b>multi-head</b>), query heads <b>share</b>: all of them (<b>multi-query</b>) or groups of
them (<b>grouped-query</b>). Qwen2.5-0.5B's measured cache is 7× smaller than without sharing; a tiny model trained with
shared keys and values loses almost nothing, while sharing them in GPT-2 after training breaks it.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> LLMs in Practice episode 2 (KV cache) and Deep Dive
episode 8 (one head, entry by entry). Code: <code>code/d09_gqa</code> (Qwen2.5-0.5B about 1 GB, GPT-2 about 500 MB, and
three tiny models trained in a few minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Sharing keys and values",
        "segment": (8, 41),
        "figures": [{"t": 23.0, "caption": "The KV cache: a key and a value per token, per layer, per head."},
                    {"t": 40.3, "caption": "Multi-head: one K/V head per query head. Grouped-query: groups share one. "
                                           "Multi-query: all share one."}],
        "body": [
            """<p>When a model writes, it keeps a key and a value for every token, in every layer and every head: the
<b>KV cache</b>. On long texts it gets huge, and reading it slows every step. Most modern models let heads <b>share</b>
their keys and values.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In classic <b>multi-head attention</b> (MHA), every query head has its own key and value head. In
<b>multi-query attention</b> (MQA), all query heads share a single one. <b>Grouped-query attention</b> (GQA) sits in
between: the query heads are split into groups, and each group shares one key/value head.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The number of query heads stays the same; only the number of
key/value heads, and so the cache, shrinks.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "A model has 32 query heads and 8 key/value heads. What is it?",
             "options": ["Multi-head attention", "Grouped-query attention, groups of 4", "Multi-query attention",
                         "Grouped-query attention, groups of 8"],
             "answer": "B.", "why": "32 / 8 = 4 query heads share each key/value head."},
            {"kind": "tf", "q": "“Multi-query attention uses a single query head.”",
             "answer": "False.", "why": "It keeps all its query heads; they share a single key and value head."},
        ],
    },
    {
        "title": "Qwen2.5, measured",
        "segment": (41, 76),
        "figures": [{"t": 61.0, "caption": "Qwen2.5-0.5B: 14 query heads share 2 key/value heads; 12,288 bytes of cache "
                                           "per token instead of 86,016."},
                    {"t": 74.6, "caption": "At 32,768 tokens: 384 MiB instead of 2,688 MiB; at 131,072: 1,536 MiB "
                                           "instead of 10,752."}],
        "body": [
            """<p>Qwen2.5-0.5B has 24 layers, <b>14 query heads</b> and only <b>2 key/value heads</b>: groups of 7. Its
cached keys have shape (1, 2, tokens, 64): two heads, not fourteen. The cache per token is</p>
<p style="text-align:center">2 (K and V) × 24 layers × 2 heads × 64 numbers × 2 bytes = <b>12,288 bytes</b>,</p>
<p>exactly what the code measures. With a key and value per query head it would be <b>86,016</b>, seven times more.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>At 32,768 tokens that is <b>384 MiB</b> instead of 2,688 MiB; at 131,072 tokens, <b>1,536 MiB</b>
instead of 10,752 MiB. Each query head still asks its own question; only the keys and values it looks up are shared:
seven different questions asked of the same index.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Cache per token = 2 × layers × key/value heads × head size ×
bytes per number; GQA cuts the key/value heads.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many bytes per token would Qwen2.5-0.5B's cache take in float32 (4 bytes per "
                                    "number)?",
             "answer": "24,576.", "why": "2 × 24 × 2 × 64 × 4; twice the bfloat16 figure."},
            {"kind": "number", "q": "A model has 32 layers, 8 key/value heads of 128 numbers, in bfloat16. How many bytes "
                                    "of cache per token? And with 32 key/value heads?",
             "answer": "131,072 (128 KiB); 524,288 (512 KiB) with 32.",
             "why": "2 × 32 × 8 × 128 × 2 = 131,072; four times the key/value heads, four times the cache."},
            {"kind": "number", "q": "About how many MiB does Qwen2.5-0.5B's cache take for a 1,000-token chat?",
             "answer": "About 11.7 MiB.", "why": "12,288 × 1,000 / 2<sup>20</sup> ≈ 11.7."},
        ],
    },
    {
        "title": "The cost: trained in, or bolted on",
        "segment": (76, 132),
        "figures": [{"t": 110.0, "caption": "Trained from scratch: 8, 2 or 1 key/value heads give val loss 1.673, 1.685, "
                                            "1.704, with an 8× smaller cache."},
                    {"t": 130.5, "caption": "GPT-2 with keys and values averaged per group, no retraining: loss jumps "
                                            "from 3.80 to 6.4–6.7."}],
        "body": [
            """<p>The same tiny GPT (8 query heads, 4 layers) is trained three times for 2,000 steps, with 8, 2 or 1
key/value heads. Validation loss: <b>1.673</b>, <b>1.685</b>, <b>1.704</b>. The cache shrinks from 1,024 numbers per
token to 256, then 128, and the model loses some parameters too (818,241 → 719,169 → 702,657). A loss of 0.03 for an 8×
smaller cache.</p>""",
            """<p>But sharing has to be learned. Take GPT-2 and average its keys and values within each group, with no
retraining: on 1,024 tokens of Shakespeare the loss jumps from <b>3.80</b> to <b>6.65</b> (4 K/V heads), 6.41 (2) and
6.52 (1). Each head was trained to expect its own keys. The grouped-query attention paper converts a trained model the
same way, then fixes it with a short extra round of training (“uptraining”).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Built in from the start, sharing costs little; bolted onto a
trained model, it needs extra training.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why does averaging GPT-2's keys within a group hurt so much, when training with shared "
                                   "keys barely hurts?",
             "answer": "GPT-2's query heads were trained against their own keys; the averaged keys are something none of "
                       "them learned to read. A model trained with shared keys learns queries that work with them.",
             "why": "The weights adapt to the architecture during training; afterwards, they cannot without more "
                    "training."},
            {"kind": "number", "q": "By what factor is the tiny model's cache smaller with 1 key/value head than with 8?",
             "answer": "8.", "why": "1,024 / 128 = 8 numbers per token."},
            {"kind": "tf", "q": "“In the episode's GPT-2 test, fewer key/value heads always gave a higher loss.”",
             "answer": "False.", "why": "4 heads gave 6.65, 2 heads 6.41, 1 head 6.52: all far worse than 3.80, but not "
                                       "in order. Without retraining, the damage is not graded."},
        ],
    },
    {
        "title": "Speed, and the code",
        "segment": (132, 153),
        "figures": [{"t": 141.6, "caption": "Every generated token reads the whole cache from memory: smaller cache, faster "
                                            "steps, more users per GPU."},
                    {"t": 152.2, "caption": "Key and value projections output fewer heads; each is repeated to match its "
                                            "group of queries."}],
        "body": [
            """<p>Generating a token means reading the whole cache from memory, so a smaller cache means faster steps and
room for more users (bigger batches) on the same GPU.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In code, the key and value projections output fewer heads (<code>kv_heads × head size</code>), and each
is repeated with <code>repeat_interleave</code> to match its group of queries. The rest of attention is unchanged.</p>""",
            """<div class="box key"><b class="t">Key idea</b>GQA is a two-line change to attention, and one of the main
reasons long contexts are affordable.</div>""",
        ],
        "exercises": [
            {"kind": "code", "q": "<b>Try it yourself.</b> Train the tiny model in <code>gqa.py</code> with 4 key/value "
                                  "heads (change the loop to <code>for kv in (4,):</code>). How many numbers per token does "
                                  "its cache hold?",
             "code": """for kv in (4,):
    ...""",
             "answer": "512 numbers per token (2 × 4 layers × 4 heads × 16); val loss 1.680, between the 8-head 1.673 and "
                       "the 2-head 1.685 (checked by running it).",
             "why": "The cache is linear in the number of key/value heads."},
            {"kind": "mc", "q": "With 8 query heads and 2 key/value heads, what does "
                                "<code>k.repeat_interleave(4, dim=1)</code> produce from key heads [A, B]?",
             "options": ["[A, B, A, B, A, B, A, B]", "[A, A, A, A, B, B, B, B]", "[A, B]", "[A, A, B, B]"],
             "answer": "B.", "why": "repeat_interleave repeats each head in place, so query heads 0–3 use A and 4–7 use "
                                   "B: consecutive groups."},
        ],
    },
]
