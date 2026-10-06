"""Study guide content for How LLMs Work: Deep Dive, episode 32: Batching and Paged Attention.

Build:  python framework/study_guide.py deep-dive d32 --video deep-dive/media/videos/d32_scene/1080p60/BatchingVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d32_batching_paging/batching_paging.py (GPT-2 small on an 8-thread CPU, second of two runs;
times vary by machine and run) or the allocation arithmetic it prints.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 32",
    "title": "Batching and Paged Attention",
    "tagline": "Serving many users from one model",
    "duration": "2:11",
    "intro": """<p>This lesson answers one question: how does one model serve many users efficiently? <b>Batching</b>
lets one read of the weights serve many requests: GPT-2 went from 26 to 404 tokens per second. <b>Continuous batching</b>
refills the batch as requests finish, instead of wasting 65% of the work on padding. <b>Paged attention</b> stores the
KV cache in small pages allocated as needed, so the same memory holds 2.5 times as many requests.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 21 (batch size) and 31 (the KV
cache). Code: <code>code/d32_batching_paging</code> (GPT-2 small, about 500 MB).</div>""",
}

CONCEPTS = [
    {
        "title": "Batching",
        "segment": (8, 51),
        "figures": [{"t": 35.1, "caption": "64 tokens per request: 26 tokens/s alone, 153 with 4 requests, 404 with 16."},
                    {"t": 50.0, "caption": "Decoding reads all the weights for little math; a batch reuses that read for many "
                                           "requests."}],
        "body": [
            """<p>A model answering one user at a time wastes most of its hardware. GPT-2 writing 64 tokens for one request
manages <b>26 tokens per second</b> on this CPU; four requests at once, <b>153</b>; sixteen, <b>404</b>: about 15 times the
throughput.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Why? Decoding one token reads all the weights for very little math (episode 31). With a batch, the same
read serves many requests in one pass. On this CPU a batch of one is so inefficient that four requests (1.68 s) even
finished sooner than one (2.42 s).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Batching turns memory-bound decoding into useful parallel work.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With 16 requests at 404 tokens per second in total, how many tokens per second does each "
                                    "user see?",
             "answer": "About 25.", "why": "404 / 16 ≈ 25: nearly the single-request speed, for 16 users at once.", "key": {'parts': [{'label': None, 'value': 25, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“Batching makes each individual request much faster.”",
             "answer": "False.", "why": "It raises total throughput; each request runs at about the same speed (here ~25 vs 26 "
                                       "tokens per second).", "key": {'value': False}},
        ],
    },
    {
        "title": "Uneven requests and continuous batching",
        "segment": (51, 76),
        "figures": [{"t": 64.8, "caption": "16 requests of 20 to 1,000 tokens in a fixed batch: 16,000 slots, 5,570 useful, "
                                           "65% wasted."},
                    {"t": 75.2, "caption": "Continuous batching: finished requests leave, waiting ones join at the next step."}],
        "body": [
            """<p>Real requests have different lengths. Sixteen requests wanting 20 to 1,000 tokens in a fixed batch: the
batch runs until the longest finishes, 16 × 1,000 = 16,000 token slots for 5,570 useful tokens, <b>65% wasted</b> on
requests that are already done.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p><b>Continuous batching</b> fixes that: as soon as a request finishes it leaves, and a waiting request joins
at the next step. The batch stays full of useful work.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Schedule per step, not per batch.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Two requests of 100 and 900 tokens in one fixed batch: what fraction of the slots is "
                                    "wasted?",
             "answer": "About 44%.", "why": "2 × 900 = 1,800 slots for 1,000 useful tokens.", "key": {'parts': [{'label': None, 'value': 44, 'tol': 0.88, 'unit': '%'}]}},
        ],
    },
    {
        "title": "Paged attention",
        "segment": (76, 122),
        "figures": [{"t": 93.3, "caption": "Reserving 2,048 tokens for every request: 7.03 GiB, 40% used."},
                    {"t": 112.9, "caption": "Pages of 16 tokens as needed: 2.81 GiB, 99.0% used: 2.5x more requests in the same "
                                            "memory."}],
        "body": [
            """<p>The other limit is memory: every request needs its KV cache. Reserving room for the maximum length (2,048
tokens) for each of 100 requests of 50 to 1,500 tokens (average 810), in GPT-2's float16 cache, takes <b>7.03 GiB</b>, and
only <b>40%</b> is ever used.</p>""",
            """<p><b>Paged attention</b> borrows an idea from operating systems: store the cache in small pages of 16
tokens, allocated only as a request grows, and keep a table of where each request's pages live. The same 100 requests
need <b>2.81 GiB</b>, 99.0% used: the same memory holds 2.5 times as many requests.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Allocate the KV cache in small pages on demand; waste is at
most one partly filled page per request.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many 16-token pages does a 1,000-token request need, and how many token slots are "
                                    "wasted?",
             "answer": "63 pages; 8 slots.", "why": "⌈1,000 / 16⌉ = 63 pages = 1,008 slots.", "key": {'parts': [{'label': 'pages', 'value': 63, 'tol': 0.5, 'unit': None}, {'label': 'wasted slots', 'value': 8, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "GPT-2's float16 cache is 36,864 bytes per token. How much does reserving 2,048 tokens "
                                    "cost per request?",
             "answer": "72 MiB.", "why": "36,864 × 2,048 = 75,497,472 bytes.", "key": {'parts': [{'label': None, 'value': 72, 'tol': 0.5, 'unit': None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Compute the pages needed with the episode's rounding.",
             "code": """PAGE = 16
for n in (1, 16, 17, 1000):
    print(n, -(-n // PAGE))""",
             "answer": "1 → 1, 16 → 1, 17 → 2, 1000 → 63.", "why": "-(-n // PAGE) is integer division rounded up."},
        ],
    },
]
