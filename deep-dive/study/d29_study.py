"""Study guide content for How LLMs Work: Deep Dive, episode 29: State-Space Models.

Build:  python framework/study_guide.py deep-dive d29 --video deep-dive/media/videos/d29_scene/1080p60/StateSpaceVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d29_state_space/state_space.py (4-layer tiny language models; torch 2.14.0, CPU) or the
recurrence formula on the exercise values. The selective-decay values in the third figure are an illustration.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 29",
    "title": "State-Space Models",
    "tagline": "Replacing attention with a running memory",
    "duration": "2:23",
    "intro": """<p>This lesson answers one question: can a language model work without attention? <b>State-space
models</b> keep a fixed-size memory, updated once per token: h ← a·h + (1 − a)·u. With a fixed decay a, a tiny model
matches attention (1.648 vs 1.644); with a <b>selective</b> decay chosen by each token (the idea behind Mamba) it does
better on these short texts (1.591). The big win is at inference: the state never grows, and each new token costs the same
6 µs whether the context is a thousand tokens or a hundred thousand.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 10–11 (FlashAttention, sliding
windows) and LLMs in Practice episode 2 (the KV cache). Code: <code>code/d29_state_space</code> (PyTorch only; about 10
minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "A running memory",
        "segment": (8, 49),
        "figures": [{"t": 35.1, "caption": "h ← a·h + (1 − a)·u: an input s steps ago still counts a^s; a = 0.5 forgets fast, "
                                           "0.99 remembers long."},
                    {"t": 47.7, "caption": "Selective: a depends on each token, so the model chooses what to keep (values "
                                           "illustrative)."}],
        "body": [
            """<p>Attention compares every new token with every earlier one and keeps a cache that grows with the text.
<b>State-space models</b> keep a fixed-size memory updated once per token. The simplest version, for every channel:
<b>h ← a · h + (1 − a) · u</b>, a running, decaying average of the inputs u. With a fixed a per channel, some channels
remember for a long time (a = 0.99) and others only a few steps (a = 0.5).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The key idea behind Mamba-style models is to make a depend on the token itself: a<sub>t</sub> =
sigmoid(W x<sub>t</sub>). The model can then choose, at every step, what to keep and what to forget: a <b>selective</b>
state space.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A recurrence carries the past in a fixed-size state; making its
decay input-dependent lets it decide what is worth keeping.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With a fixed a = 0.9, how much does an input from 10 steps ago still count, relative to "
                                    "when it arrived (a^10)?",
             "answer": "About 0.35.", "why": "0.9¹⁰ ≈ 0.349."},
            {"kind": "number", "q": "After how many steps has a channel with a = 0.99 forgotten half of an input?",
             "answer": "About 69.", "why": "0.99^s = 0.5 → s = ln 0.5 / ln 0.99 ≈ 69."},
        ],
    },
    {
        "title": "The experiment",
        "segment": (49, 84),
        "figures": [{"t": 67.8, "caption": "Validation loss: attention 1.644, fixed decay 1.648, selective decay 1.591."},
                    {"t": 82.5, "caption": "On 64-character texts nearby characters matter most; exact recall from far back is "
                                           "where attention keeps its edge."}],
        "body": [
            """<p>The tiny model's attention is replaced by each recurrence (no position embedding is needed: a
recurrence knows the order by itself) and trained 2,000 steps. Validation loss: attention <b>1.644</b>, fixed decay
<b>1.648</b> (about the same), selective decay <b>1.591</b>, better than attention here.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>“Here” means 64-character texts, where nearby characters matter most. A fixed-size state must squeeze the
whole past into a few numbers, and research has found that exact recall from far back is where attention keeps its edge
(compare the copy task of episode 11).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Recurrences can match or beat attention on local patterns; long,
exact lookup is their weak spot.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which model reached the lowest validation loss in the episode?",
             "options": ["Attention", "Recurrence, fixed decay", "Recurrence, selective decay", "All equal"],
             "answer": "C.", "why": "1.591 vs 1.644 and 1.648."},
            {"kind": "short", "q": "Why does the recurrent model not need a position embedding?",
             "answer": "It processes tokens one after another, so the order is built into how the state is updated; "
                       "attention, by itself, is blind to order (episode 4).",
             "why": "Each update depends on the previous state, so swapping tokens changes the result."},
        ],
    },
    {
        "title": "Inference: memory and time",
        "segment": (84, 135),
        "figures": [{"t": 96.8, "caption": "Per layer: attention's KV cache 0.98 MiB after 1,000 tokens, 97.66 MiB after 100,000; "
                                           "the recurrent state stays 0.50 KiB."},
                    {"t": 114.8, "caption": "One new token, one layer: attention 37.9 µs → 11,334 µs as the context grows; the "
                                            "recurrence 6.3–6.4 µs."}],
        "body": [
            """<p>The big win is at inference. Per layer (128 numbers per token, float32), attention's cache grows with
every token: <b>0.98 MiB</b> after a thousand tokens, <b>97.66 MiB</b> after a hundred thousand. The recurrent state stays
at <b>0.50 KiB</b>, forever.</p>""",
            """<p>Time to process one new token in one layer, on this CPU: attention takes 37.9 µs with a thousand tokens
of context, 312.6 µs with ten thousand, 11,334 µs with a hundred thousand; the recurrence about <b>6.4 µs</b> every time.
So some recent models mix the two: AI21's Jamba interleaves Mamba layers with a few attention layers, cheap memory for
most of the work and exact lookup where it's needed.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Constant memory and constant time per token, at the price of
compressing the past.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How large would one layer's KV cache be after 1,000,000 tokens (128 numbers, float32, "
                                    "keys and values)?",
             "answer": "About 977 MiB.", "why": "2 × 1,000,000 × 128 × 4 bytes ≈ 1.02 × 10⁹ bytes."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Run the recurrence by hand on a constant input u = 1 with a = 0.5, "
                                  "for 5 steps from h = 0.",
             "code": """h = 0.0
for _ in range(5):
    h = 0.5 * h + 0.5 * 1.0
    print(h)""",
             "answer": "0.5, 0.75, 0.875, 0.9375, 0.96875.", "why": "The state approaches the input, closing half the gap each "
                                                                    "step."},
        ],
    },
]
