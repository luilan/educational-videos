"""Study guide content for How LLMs Work: Deep Dive, episode 25: Data Parallelism.

Build:  python framework/study_guide.py deep-dive d25 --video deep-dive/media/videos/d25_scene/1080p60/DataParallelVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d25_data_parallel/data_parallel.py (4 processes with torch.distributed "gloo" on one CPU
machine; torch 2.14.0) or simple arithmetic on parameter counts.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 25",
    "title": "Data Parallelism",
    "tagline": "Many workers, one model, the same step",
    "duration": "2:27",
    "intro": """<p>This lesson answers one question: how do many workers train one model together? In <b>data
parallelism</b>, every worker holds a full copy of the model and processes its own slice of the batch; an
<b>all-reduce</b> averages the gradients so every copy takes the same step. A real run with 4 processes matches a single
process to within float rounding for 200 steps. The costs: exchanging every gradient every step, and holding the whole
model, gradients and optimizer state on every worker.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 19 (AdamW) and 21 (batch size and
gradient accumulation). Code: <code>code/d25_data_parallel</code> (PyTorch only; starts 4 local processes; a few
minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Copies, slices, and all-reduce",
        "segment": (8, 49),
        "figures": [{"t": 35.0, "caption": "Each worker: a full model copy and a slice of the batch; all-reduce averages the "
                                           "gradients."},
                    {"t": 48.5, "caption": "A real run: 4 processes × 8 sequences vs 1 process × 32 sequences."}],
        "body": [
            """<p>Large models are trained on thousands of GPUs. The simplest way to use many workers is <b>data
parallelism</b>: every worker holds a full copy of the model and takes its own slice of the batch. Each computes
gradients on its slice; then the workers <b>average their gradients</b> (an <b>all-reduce</b>), so every copy takes
exactly the same step and they stay identical.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The episode runs it for real on one machine: 4 worker processes talking through PyTorch's distributed
library (the “gloo” backend; GPUs use “nccl”), each with 8 sequences, compared with one process that takes all 32.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Data parallelism splits the batch, not the model; an
all-reduce keeps the copies in sync.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why must every worker start from the same weights?",
             "answer": "They all apply the same averaged update; if they started differently, they would stay different, "
                       "and averaging gradients from different models would not make sense.",
             "why": "The code seeds every worker identically (torch.manual_seed(0)); DDP also broadcasts rank 0's weights at "
                    "the start."},
            {"kind": "tf", "q": "“In data parallelism, each worker holds a quarter of the model.”",
             "answer": "False.", "why": "Each holds the whole model and a quarter of the batch."},
        ],
    },
    {
        "title": "Exactly the same training",
        "segment": (49, 77),
        "figures": [{"t": 64.8, "caption": "First gradient within 1.5 × 10⁻⁸; losses within 4.8 × 10⁻⁷ for 200 steps: 4.3335 "
                                           "→ 2.3006 in both runs."},
                    {"t": 76.1, "caption": "The average of four averages over equal slices is the average over all 32."}],
        "body": [
            """<p>The first gradient matches the single process to within <b>1.5 × 10<sup>−8</sup></b>, and over 200 steps
the losses never differ by more than <b>4.8 × 10<sup>−7</sup></b>: 4.3335 at the start, 2.3006 at the end, in both runs.
Why exact? The batch loss is an average, and the average of four averages over equal slices is the average over all
32. Splitting the work changes nothing in the math (only the order of float additions).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>With equal slices, data-parallel training is mathematically the
same as one big batch.</div>""",
        ],
        "exercises": [
            {"kind": "code", "q": "<b>Try it yourself.</b> Change the script to 2 workers × 16 sequences (and fewer steps). "
                                  "Does it still match?",
             "code": """WORLD, PER_WORKER, STEPS = 2, 16, 50""",
             "answer": "Yes: first gradient within 1.5 × 10⁻⁸, losses within 4.8 × 10⁻⁷ (step 50: 2.5632 in both).",
             "why": "Any equal split of the 32 sequences gives the same average gradient."},
            {"kind": "short", "q": "What would break if one worker got 4 sequences and another 12, and DDP still took a plain "
                                   "average of their gradients?",
             "answer": "The average would weight the 4 sequences as much as the 12, so it would no longer equal the batch "
                       "average.",
             "why": "Equal slices make the mean of means equal the overall mean."},
        ],
    },
    {
        "title": "The costs",
        "segment": (77, 145),
        "figures": [{"t": 92.1, "caption": "Gradients exchanged every step: 3.1 MiB (tiny model), 0.5 GiB (GPT-2), 26.1 GiB "
                                           "(7 billion parameters)."},
                    {"t": 127.3, "caption": "Every worker holds 16 bytes per parameter with Adam: 7 billion parameters → 112 "
                                            "GB."}],
        "body": [
            """<p><b>Communication.</b> Every step, every worker must exchange all its gradients: 3.1 MiB for the tiny
model, 0.5 GiB for GPT-2, 26.1 GiB for a 7-billion-parameter model (float32). The usual method is a <b>ring
all-reduce</b>: workers pass chunks to their neighbor around a ring, and each sends about 2 × (N − 1)/N × the gradient
size, however many workers there are (4.7 MiB for the tiny model with 4 workers). Libraries start sending during the
backward pass, to hide the wait.</p>""",
            """<p><b>Memory.</b> Every worker needs the whole model: with Adam in float32 that is 16 bytes per parameter
(weights, gradients, and Adam's m and v, 4 bytes each). For 7 billion parameters: 112 GB, more than one GPU holds. The
next episodes split the model itself. In code, data parallelism is two lines: start the process group, wrap the model in
<code>DistributedDataParallel</code>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Data parallelism scales compute, not memory: each worker still
needs room for the whole training state.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With 8 workers, how many GiB does each send per step in a ring all-reduce of GPT-2's "
                                    "0.46 GiB of gradients?",
             "answer": "About 0.81 GiB.", "why": "2 × 7/8 × 0.4636 GiB ≈ 0.81."},
            {"kind": "number", "q": "How much memory does GPT-2 small's training state take per worker (16 bytes per "
                                    "parameter)?",
             "answer": "About 2.0 GB (1.85 GiB).", "why": "124,439,808 × 16 bytes ≈ 1.99 × 10⁹ bytes."},
        ],
    },
]
