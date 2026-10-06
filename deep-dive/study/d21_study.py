"""Study guide content for How LLMs Work: Deep Dive, episode 21: Batch Size and Gradient Accumulation.

Build:  python framework/study_guide.py deep-dive d21 --video deep-dive/media/videos/d21_scene/1080p60/BatchSizeVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d21_batch_size/batch_size.py (a 4-layer tiny GPT; torch 2.14.0, CPU, 8 threads) or simple
arithmetic on those numbers.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 21",
    "title": "Batch Size and Gradient Accumulation",
    "tagline": "How many examples per step, and what if they don't fit",
    "duration": "2:28",
    "intro": """<p>This lesson answers two questions: how big should a training batch be, and what can you do when it
doesn't fit in memory? Bigger batches give gradients closer to the true one (cosine similarity 0.17 for one sequence,
0.90 for 256), but for the same data they mean fewer steps: here, batches of 8 reached a loss of 1.660 and batches of 128
only 1.930. Big batches win on hardware speed. <b>Gradient accumulation</b> gives the gradient of a big batch with the
memory of a small one.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 18 (backprop) and 19 (AdamW). Code:
<code>code/d21_batch_size</code> (PyTorch only; about 15 minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Noisy gradients",
        "segment": (8, 41),
        "figures": [{"t": 15.4, "caption": "Each step averages the gradient over a batch of sequences."},
                    {"t": 39.6, "caption": "Similarity to the gradient of 4,096 sequences: 0.17 (batch 1), 0.50 (16), 0.72 "
                                           "(64), 0.90 (256)."}],
        "body": [
            """<p>Every training step averages the gradient over a <b>batch</b> of examples. A small batch gives a noisy
estimate of the true gradient. After 300 training steps, each batch's gradient is compared with the average over 4,096
sequences (cosine similarity): batch 1 <b>0.17</b>, 4 0.19, 16 <b>0.50</b>, 64 <b>0.72</b>, 256 <b>0.90</b>. Bigger
batches point more reliably downhill.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>The batch gradient is an estimate of the full-data gradient;
its noise shrinks as the batch grows.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“A batch of 256 gives exactly the true gradient.”",
             "answer": "False.", "why": "Its similarity to the 4,096-sequence gradient is 0.90, not 1: still an estimate.", "key": {'value': False}},
            {"kind": "short", "q": "Why is a single sequence's gradient so different from the average (similarity 0.17)?",
             "answer": "It reflects the quirks of that one piece of text; averaging over many sequences cancels those and "
                       "keeps what they share.",
             "why": "Noise in an average shrinks as more independent samples are added."},
        ],
    },
    {
        "title": "Same data, different batch sizes",
        "segment": (41, 104),
        "figures": [{"t": 70.0, "caption": "48,000 sequences: batch 8 → 1.660, 32 → 1.709, 128 → 1.930 (1.811 with twice the "
                                           "learning rate)."},
                    {"t": 90.1, "caption": "Time per sequence on this CPU: 2.82 ms (batch 8), 1.56 ms (32), 1.68 ms (128)."}],
        "body": [
            """<p>A bigger batch means fewer steps for the same data. With 48,000 training sequences in every run: batch 8,
6,000 steps, loss <b>1.660</b>; batch 32, 1,500 steps, <b>1.709</b>; batch 128, 375 steps, <b>1.930</b>. Doubling the
learning rate for batch 128 helps (<b>1.811</b>) but it still trails. For a fixed amount of data, more, noisier steps
won here.</p>""",
            """<p>So why use big batches? <b>Speed</b>: on this CPU a batch of 8 costs 2.82 ms per sequence, a batch of 32
only 1.56 ms, because the hardware does more work in parallel. Here it was already saturated at 32 (1.68 ms at 128);
GPUs keep gaining up to much larger batches. Small batches use the data best; big batches use the hardware best. Large
language models train with batches of millions of tokens, with learning rates tuned to match.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Batch size trades data efficiency (more steps) against hardware
efficiency (more parallel work per step).</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With 48,000 sequences and batch 64, how many steps?",
             "answer": "750.", "why": "48,000 / 64.", "key": {'parts': [{'label': None, 'value': 750, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "How long would the 48,000 sequences take at batch 8 and at batch 32, at the measured "
                                    "time per sequence?",
             "answer": "About 135 s and 75 s.", "why": "48,000 × 2.82 ms and 48,000 × 1.56 ms.", "key": {'parts': [{'label': 'batch 8 (s)', 'value': 135, 'tol': 2.7, 'unit': None}, {'label': 'batch 32 (s)', 'value': 75, 'tol': 1.5, 'unit': None}]}},
            {"kind": "mc", "q": "At a fixed amount of data, which run reached the lowest loss?",
             "options": ["Batch 128, lr 0.002", "Batch 32", "Batch 8", "Batch 128, lr 0.001"],
             "answer": "C.", "why": "1.660 with 6,000 steps.", "key": {'choice': 2}},
        ],
    },
    {
        "title": "Gradient accumulation",
        "segment": (104, 138),
        "figures": [{"t": 117.4, "caption": "Run micro-batches, divide each loss by their number, call backward without "
                                            "clearing: the gradients add up."},
                    {"t": 129.3, "caption": "4 micro-batches of 8 = one batch of 32, to 1.5 × 10⁻⁸: the memory of 8, the "
                                            "gradient of 32."}],
        "body": [
            """<p>When the batch you want doesn't fit in memory, use <b>gradient accumulation</b>: run several small
micro-batches, divide each loss by their number, and call <code>backward()</code> each time without clearing the
gradients: PyTorch adds them up in <code>.grad</code>. Then take one optimizer step.</p>""",
            """<p>Four micro-batches of 8 give the same gradient as one batch of 32 to within <b>1.5 × 10<sup>−8</sup></b>
(gradient size 1.123). The memory of a batch of 8, the gradient of 32; just no speed-up, since the micro-batches run one
after another.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Gradients are sums, so they can be accumulated across
micro-batches; dividing each loss by the number of micro-batches turns the sum into the average.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "What goes wrong if you forget to divide each micro-batch loss by 4?",
             "answer": "The accumulated gradient is 4 times too large, as if the learning rate were 4 times higher (for SGD; "
                       "Adam is mostly insensitive to this scale).",
             "why": "Each backward adds a full-size gradient; the sum of 4 is 4 × the average."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Accumulate 8 micro-batches of 4 instead, and compare with the "
                                  "batch of 32.",
             "code": """model.zero_grad()
for k in range(8):
    (lm_loss(model, x[4*k:4*k+4], y[4*k:4*k+4]) / 8).backward()
print((big - flat_grad(model)).abs().max())""",
             "answer": "About 10⁻⁸: the same gradient again.",
             "why": "Any split of the batch gives the same average."},
        ],
    },
]
