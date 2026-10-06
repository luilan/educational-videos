"""Study guide content for How LLMs Work: Deep Dive, episode 27: Sharding the Optimizer State.

Build:  python framework/study_guide.py deep-dive d27 --video deep-dive/media/videos/d27_scene/1080p60/ShardingVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d27_sharding/sharding.py (4 local processes, torch.distributed "gloo", torch 2.14.0) or the
same per-worker arithmetic on the exercise values.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 27",
    "title": "Sharding the Optimizer State",
    "tagline": "Stop storing the same numbers on every worker",
    "duration": "2:08",
    "intro": """<p>This lesson answers one question: why should every data-parallel worker store the same Adam state?
It shouldn't. With <b>optimizer state sharding</b> (ZeRO, FSDP), each worker keeps only its slice: it updates its slice
of the weights, then the workers <b>all-gather</b> the result. A real run with PyTorch's ZeroRedundancyOptimizer gives
identical losses with a quarter of the Adam state per worker; sharding weights and gradients too brings a 7-billion-
parameter model from 112 GB to under 2 GB per worker on 64 workers.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 19 (AdamW) and 25 (data
parallelism). Code: <code>code/d27_sharding</code> (PyTorch only; starts 4 local processes; a few minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Every worker, the same state",
        "segment": (8, 38),
        "figures": [{"t": 20.7, "caption": "Data parallelism copies weights, gradients and Adam's m and v on every worker."},
                    {"t": 37.1, "caption": "Sharded: each worker updates its quarter with its quarter of Adam's state, then "
                                           "all-gather."}],
        "body": [
            """<p>Data parallelism has a wasteful habit: every worker stores the same Adam state, the same two numbers per
parameter, copied on every machine. <b>Sharding</b> gives each worker only its own slice. After the gradients are
averaged, each worker updates only its slice of the weights, using only its slice of Adam's state; then the workers share
the updated slices with an <b>all-gather</b>, so everyone ends the step with the full, identical model.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Each optimizer state entry only needs to live on one worker;
the workers coordinate to rebuild the full weights after the update.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why can each worker update only its own slice of the weights correctly?",
             "answer": "After the all-reduce, every worker has the same averaged gradients; Adam's update for each weight "
                       "uses only that weight's gradient and its own m and v, so a slice can be updated independently.",
             "why": "Adam is element-wise: no weight's update depends on another weight's state."},
        ],
    },
    {
        "title": "Same training, a quarter of the state",
        "segment": (38, 66),
        "figures": [{"t": 53.0, "caption": "4 processes, 100 steps: plain AdamW and ZeroRedundancyOptimizer give identical "
                                           "losses (2.4467)."},
                    {"t": 65.3, "caption": "Adam state per worker: 6.24 MiB plain, 1.56 MiB sharded."}],
        "body": [
            """<p>Four processes train the same 818,241-parameter tiny GPT for 100 steps, once with plain AdamW and once
with PyTorch's <code>ZeroRedundancyOptimizer</code> wrapping AdamW. The losses are identical (largest difference 0.0;
<b>2.4467</b> after 100 steps in both runs). But each worker holds <b>6.24 MiB</b> of Adam state with the plain optimizer
and <b>1.56 MiB</b> sharded: a quarter, with four workers.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Sharding changes where numbers are stored, not what is
computed.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With 8 workers, how much Adam state would each hold for the tiny model?",
             "answer": "About 0.78 MiB.", "why": "6.24 MiB / 8.", "key": {'parts': [{'label': None, 'value': 0.78, 'tol': 0.0156, 'unit': None}]}},
            {"kind": "tf", "q": "“Sharding the optimizer state changes the training result.”",
             "answer": "False.", "why": "The episode's losses were identical to every printed digit.", "key": {'value': False}},
        ],
    },
    {
        "title": "At scale, and the cost",
        "segment": (66, 118),
        "figures": [{"t": 84.2, "caption": "7B parameters on 64 workers: 112 GB copied, 56.9 GB with Adam state sharded, "
                                           "1.8 GB with everything sharded."},
                    {"t": 108.0, "caption": "ZeRO (Microsoft DeepSpeed) and FSDP (PyTorch): no worker stores what another "
                                            "already holds."}],
        "body": [
            """<p>At scale this is what makes training possible. A 7-billion-parameter model with Adam in float32 (16 bytes
per parameter) on 64 workers: everything copied needs <b>112 GB</b> per worker; sharding Adam's state (8 of the 16 bytes)
brings it to <b>56.9 GB</b>; sharding the weights and gradients too, <b>1.8 GB</b> (activations not counted).</p>""",
            """<p>Nothing is free: sharded weights must be gathered just before each layer uses them and released
afterwards, so communication is traded for memory. You will meet these ideas as <b>ZeRO</b>, from Microsoft's DeepSpeed,
and as <b>FSDP</b>, fully sharded data parallel, in PyTorch. In code, sharding the optimizer is one line: wrap AdamW in
<code>ZeroRedundancyOptimizer</code>; the training loop doesn't change.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Shard the optimizer state, then the gradients, then the
weights: each step saves memory and adds communication.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "GPT-2 small (124,439,808 parameters, 16 bytes each) fully sharded over 8 workers: how "
                                    "much per worker?",
             "answer": "About 249 MB.", "why": "124,439,808 × 16 / 8 ≈ 2.49 × 10⁸ bytes.", "key": {'parts': [{'label': None, 'value': 249, 'tol': 4.98, 'unit': None}]}},
            {"kind": "number", "q": "For 7B parameters on 64 workers, show where 56.9 GB comes from.",
             "answer": "8 bytes × 7 × 10⁹ (weights + gradients, full) = 56 GB, plus 8 bytes × 7 × 10⁹ / 64 (Adam's share) "
                       "≈ 0.875 GB.",
             "why": "Only m and v are divided among the workers.", "key": {'self': True}},
            {"kind": "mc", "q": "What does full sharding (weights too) cost?",
             "options": ["Lower accuracy", "More communication: weights gathered before each layer", "More memory",
                         "Nothing"],
             "answer": "B.", "why": "Memory is traded for traffic.", "key": {'choice': 1}},
        ],
    },
]
