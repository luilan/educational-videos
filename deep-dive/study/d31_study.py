"""Study guide content for How LLMs Work: Deep Dive, episode 31: The KV Cache, Deeper.

Build:  python framework/study_guide.py deep-dive d31 --video deep-dive/media/videos/d31_scene/1080p60/KVCacheVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d31_kv_cache/kv_cache.py (GPT-2 small, greedy generation; transformers 4.57.1, torch 2.14.0,
CPU, 8 threads; times depend on the machine) or the cache-size formula.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 31",
    "title": "The KV Cache, Deeper",
    "tagline": "Prefill, decode, and what the cache really saves",
    "duration": "2:28",
    "intro": """<p>This lesson answers one question: what does the KV cache buy, and what does it cost? GPT-2 writes the
same 200 tokens with and without it, 2.4 times faster with it; without it, every step recomputes the whole text and gets
slower as the text grows. The cache holds keys and values for every layer and token: 73,728 bytes per token. And reading a
prompt in one pass (<b>prefill</b>) is 64 times faster than one token at a time, which is why <b>decoding</b> is the slow
part of inference.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> LLMs in Practice episode 2 (the KV cache) and Deep Dive
episode 9 (grouped-query attention). Code: <code>code/d31_kv_cache</code> (GPT-2 small, about 500 MB).</div>""",
}

CONCEPTS = [
    {
        "title": "What the cache saves",
        "segment": (8, 60),
        "figures": [{"t": 39.5, "caption": "200 tokens, identical output: 7.43 s with the cache, 18.03 s without."},
                    {"t": 58.6, "caption": "Time per new token: ~36 ms with the cache; without, 43 → 88 → 138 ms as the text "
                                           "grows."}],
        "body": [
            """<p>To write each new token, a model needs the keys and values of every token before it. The <b>KV cache</b>
keeps them instead of recomputing them. GPT-2 writes 200 tokens greedily, twice: the output is identical token for token,
but it takes <b>7.43 s</b> with the cache and <b>18.03 s</b> without, 2.4 times slower.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Step by step: with the cache every new token costs about <b>36 ms</b>, however long the text. Without it
the cost keeps growing: 43 ms at 7 tokens, 57.5 at 56, 88 at 106, 138 at 206, because the whole text is recomputed every
time.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The cache turns each new token's cost from “the whole text” into
“one token plus a lookup”; the output is exactly the same.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With the cache, about how many tokens per second does GPT-2 generate here?",
             "answer": "About 27.", "why": "200 tokens / 7.43 s ≈ 26.9."},
            {"kind": "short", "q": "Why does the time per token without the cache grow roughly in proportion to the text "
                                   "length?",
             "answer": "Each step runs the model over the whole text so far, so the work grows with the number of tokens "
                       "(and attention even with its square).",
             "why": "43 → 57.5 → 88 → 138 ms at 7, 56, 106, 206 tokens."},
        ],
    },
    {
        "title": "What the cache holds",
        "segment": (60, 77),
        "figures": [{"t": 76.4, "caption": "12 layers × keys and values of shape (1, 12, tokens, 64): 73,728 bytes per token; "
                                           "72 MiB at 1,024 tokens."}],
        "body": [
            """<p>For each of GPT-2's 12 layers the cache holds keys and values of shape (batch 1, 12 heads, tokens, 64
numbers). That is 2 × 12 × 768 × 4 = <b>73,728 bytes per token</b> in float32: <b>72 MiB</b> at 1,024 tokens, for one
conversation.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Cache size = 2 × layers × (key/value heads × head size) ×
bytes × tokens: it grows with every token and every user.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "GPT-2's cache in float16: how many bytes per token, and how many MiB at 1,024 tokens?",
             "answer": "36,864 bytes; 36 MiB.", "why": "Half of 73,728 bytes per token."},
            {"kind": "mc", "q": "Qwen2.5-0.5B's bfloat16 cache is 12,288 bytes per token (episode 9); GPT-2's in float16 is "
                                "36,864. Why is the larger model's cache smaller?",
             "options": ["It has fewer layers", "Grouped-query attention: 2 key/value heads instead of one per query head",
                         "It uses shorter tokens", "It does not cache values"],
             "answer": "B.", "why": "2 K/V heads × 64 × 24 layers vs 12 heads × 64 × 12 layers."},
        ],
    },
    {
        "title": "Prefill and decode",
        "segment": (77, 139),
        "figures": [{"t": 94.4, "caption": "A 512-token prompt: 0.29 s in one pass (1,738 tokens/s), 18.76 s one token at a "
                                           "time."},
                    {"t": 112.4, "caption": "Prefill keeps the hardware busy; each decode step reads all the weights and the "
                                            "cache for little math."}],
        "body": [
            """<p>Reading a 512-token prompt in one pass, called <b>prefill</b>, takes <b>0.29 s</b>: 1,738 tokens per
second. Feeding the same prompt one token at a time takes <b>18.76 s</b>, 64 times slower. Prefill processes all the
prompt's tokens in parallel, with big matrix multiplications that keep the hardware busy. <b>Decoding</b> can only produce
one token at a time, and each step reads all the weights and the whole cache for very little math: here, 27 tokens per
second.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>That is why a chatbot pauses before its answer (prefill) and then streams it word by word (decode), and
why the cache's size limits context length and the number of users per machine. Shared keys and values (episode 9) and
sliding windows (episode 11) shrink it. In code, the cache is <code>past_key_values</code>: pass it back in and feed
only the newest token.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Prefill is compute-bound and fast per token; decode is
memory-bound and slow per token.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "At the measured rates, how long would prefill of a 2,000-token prompt take, and how long "
                                    "to decode a 200-token answer?",
             "answer": "About 1.2 s and 7.4 s.", "why": "2,000 / 1,738 ≈ 1.15 s; 200 / 27 ≈ 7.4 s (on this CPU; real servers "
                                                       "are far faster but show the same imbalance)."},
            {"kind": "tf", "q": "“Generating tokens is slow mainly because each step does a lot of arithmetic.”",
             "answer": "False.", "why": "Each step does little math for the data it must read; it is limited by moving the "
                                       "weights and cache, not by arithmetic."},
        ],
    },
]
