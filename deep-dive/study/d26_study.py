"""Study guide content for How LLMs Work: Deep Dive, episode 26: Tensor and Pipeline Parallelism.

Build:  python framework/study_guide.py deep-dive d26 --video deep-dive/media/videos/d26_scene/1080p60/TensorPipelineVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d26_tensor_pipeline/tensor_pipeline.py (2 local processes, torch.distributed "gloo"; torch
2.14.0) or the bubble formula it prints.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 26",
    "title": "Tensor and Pipeline Parallelism",
    "tagline": "Splitting one model across workers",
    "duration": "2:29",
    "intro": """<p>This lesson answers one question: how do you train a model that doesn't fit on one worker? Split the
model itself. <b>Tensor parallelism</b> splits each weight matrix: two workers each compute half of an MLP and an
all-reduce sums their partial outputs (matching the full MLP to 1.2 × 10<sup>−6</sup>). <b>Pipeline parallelism</b> splits
the layers: one worker sends its activations to the next (an exact match), and micro-batches shrink the idle
“bubble”. Real runs combine both with data parallelism.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episode 25 (data parallelism) and How LLMs
Work episode 8 (the MLP). Code: <code>code/d26_tensor_pipeline</code> (PyTorch only; starts 2 local processes; seconds on
a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Tensor parallelism",
        "segment": (8, 76),
        "figures": [{"t": 41.9, "caption": "Worker 0 keeps half the columns of the first matrix (half the hidden units) and "
                                           "the matching half of the rows of the second."},
                    {"t": 58.8, "caption": "Summing the two partial outputs (all-reduce) gives the full MLP's output, to "
                                           "1.2 × 10⁻⁶."}],
        "body": [
            """<p>Data parallelism needs the whole model on every worker; a 7-billion-parameter model with Adam needs 112
GB. So split the model. <b>Tensor parallelism</b> splits each weight matrix. For an MLP 128 → 512 → 128, worker 0 keeps
the first half of the <b>columns</b> of the first matrix, so it computes half the hidden units; GELU acts on each unit
separately, so that is fine. It keeps the matching half of the <b>rows</b> of the second matrix and gets a partial
output. Worker 1 does the same with the other halves.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Summing the two partial outputs with an all-reduce gives exactly the full MLP's output: in a real run with
two processes the largest difference is <b>1.2 × 10<sup>−6</sup></b> (float rounding), and each worker holds 65,792 of the
131,584 parameters. The price is an all-reduce of the layer's output in every layer (256 KiB here; attention is split by
heads the same way), so tensor parallelism usually stays inside one machine, where links between GPUs are fastest.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Split the first matrix by columns and the second by rows: each
worker produces a partial sum, and one all-reduce completes it.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With 4 tensor-parallel workers, how many of the 512 hidden units does each compute?",
             "answer": "128.", "why": "512 / 4, each worker keeping a quarter of the columns and rows."},
            {"kind": "short", "q": "Why does splitting the hidden units work with GELU in between?",
             "answer": "GELU is applied to each hidden unit independently, so each worker can apply it to its own half without "
                       "needing the other half.",
             "why": "A function that mixed hidden units (like a softmax over them) would need communication first."},
        ],
    },
    {
        "title": "Pipeline parallelism and the bubble",
        "segment": (76, 115),
        "figures": [{"t": 92.1, "caption": "Worker 0 runs layers 0–1 and sends its activations (64 KiB per micro-batch) to "
                                           "worker 1 for layers 2–3: an exact match."},
                    {"t": 113.7, "caption": "With 1 batch, each worker idles half the time; with 4 micro-batches, 20%."}],
        "body": [
            """<p><b>Pipeline parallelism</b> splits the layers: worker 0 runs layers 0 and 1, then sends its activations
to worker 1, which runs layers 2 and 3. Only one tensor crosses between them (64 KiB per micro-batch here), and the
output matches the full model exactly (difference 0.0).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>But a pipeline has a <b>bubble</b>: with one batch, worker 1 waits while worker 0 works, then worker 0
waits: half the time idle. Splitting the batch into micro-batches keeps both busy. The idle fraction is
(stages − 1) / (micro-batches + stages − 1): 2 stages with 4 micro-batches, 20%; with 16, 6%; 8 stages need many more
(8 micro-batches: 47%; 32: 18%).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Pipelines communicate little but idle at the start and end;
more micro-batches shrink the bubble.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "4 stages and 12 micro-batches: what fraction of the time is each stage idle?",
             "answer": "20%.", "why": "(4 − 1) / (12 + 4 − 1) = 3 / 15."},
            {"kind": "number", "q": "With 8 stages, how many micro-batches are needed to bring the bubble under 10%?",
             "answer": "64.", "why": "7 / (m + 7) < 0.1 → m > 63."},
        ],
    },
    {
        "title": "Combining them",
        "segment": (115, 138),
        "figures": [{"t": 126.7, "caption": "Tensor parallel inside a machine, pipeline across machines, data parallel on top."},
                    {"t": 136.5, "caption": "In code: a slice of each matrix and one all-reduce; a send and a receive."}],
        "body": [
            """<p>Real training runs combine all three: <b>tensor</b> parallelism inside a machine (heavy, frequent
communication on fast links), <b>pipeline</b> parallelism across machines (little communication), and <b>data</b>
parallelism on top, over many copies of the whole arrangement. In code, tensor parallelism is a slice of each matrix and
one all-reduce; pipeline parallelism is a send and a receive between stages.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Put the most communication-hungry split on the fastest links.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which split communicates in every layer?",
             "options": ["Data parallelism", "Tensor parallelism", "Pipeline parallelism", "None"],
             "answer": "B.", "why": "Every split layer ends with an all-reduce of its output; pipelines communicate only "
                                   "between stages."},
            {"kind": "tf", "q": "“Pipeline parallelism changes the model's output.”",
             "answer": "False.", "why": "Same layers, same order: the episode's pipeline matches the full model exactly."},
        ],
    },
]
