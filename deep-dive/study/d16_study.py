"""Study guide content for How LLMs Work: Deep Dive, episode 16: Pre-Norm, Post-Norm, and Stability.

Build:  python framework/study_guide.py deep-dive d16 --video deep-dive/media/videos/d16_scene/1080p60/PrePostNormVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d16_pre_post_norm/pre_post_norm.py (a 12-layer tiny GPT on Tiny Shakespeare; torch 2.14.0,
CPU; seed 1337).
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 16",
    "title": "Pre-Norm, Post-Norm, and Stability",
    "tagline": "Same parts, different order",
    "duration": "2:28",
    "intro": """<p>This lesson answers one question: does it matter whether the normalization comes before or after each
block? <b>Post-norm</b> (the original Transformer, BERT) normalizes the stream itself: x = norm(x + block(x)).
<b>Pre-norm</b> (GPT-2, Llama, Qwen) normalizes only what the block reads: x = x + block(norm(x)). At a low learning rate
they train equally well; at three times the learning rate, post-norm stalls within ten steps unless the learning rate is
warmed up, while pre-norm trains either way.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 13 (residual stream) and 14
(normalization). Code: <code>code/d16_pre_post_norm</code> (PyTorch only; ten 12-layer tiny models, about an hour on a
CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Two orders",
        "segment": (8, 54),
        "figures": [{"t": 36.7, "caption": "Post-norm: x = norm(x + block(x)). Pre-norm: x = x + block(norm(x)), plus one "
                                           "final norm."},
                    {"t": 53.2, "caption": "Pre-norm keeps the residual stream a clean highway; post-norm interrupts it 24 "
                                           "times in 12 layers."}],
        "body": [
            """<p>The original Transformer and BERT put the normalization <b>after</b> each block: add the block's output
to the stream, then normalize the stream itself. GPT-2, Llama and Qwen put it <b>before</b>: normalize only what the block
reads, and add its output to the stream untouched. Pre-norm then needs one final norm before the output layer.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In pre-norm, the residual stream of episode 13 is a clean highway: nothing but additions from the
embedding to the end. In post-norm, every layer's sum passes through a norm, so the direct path is interrupted 24 times in
a 12-layer model (after each attention and each MLP).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Pre-norm normalizes the block's input; post-norm normalizes the
stream. Only pre-norm keeps the identity path intact.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many times is the residual stream normalized in a 32-layer post-norm transformer "
                                    "(one attention and one MLP per layer)?",
             "answer": "64.", "why": "Two norms per layer, each applied to the stream itself.", "key": {'parts': [{'label': None, 'value': 64, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“In pre-norm, the residual stream is never normalized before the output layer.”",
             "answer": "False.", "why": "Pre-norm adds one final norm before the output layer; inside the layers the "
                                       "stream itself is never normalized.", "key": {'value': False}},
        ],
    },
    {
        "title": "Learning rate, and warmup",
        "segment": (54, 101),
        "figures": [{"t": 81.8, "caption": "Learning rate 0.003, no warmup: pre-norm 1.67; post-norm stuck at 3.36 from step "
                                           "100 to 1,500."},
                    {"t": 99.6, "caption": "With 300 warmup steps: post-norm 1.72, pre-norm 1.66."}],
        "body": [
            """<p>A 12-layer tiny GPT, 1,500 steps. Validation loss at learning rate 0.001: pre-norm <b>1.67</b>,
post-norm <b>1.68</b>. No difference. At three times the learning rate (0.003): pre-norm 1.67 again; post-norm stuck at
<b>3.36</b> from step 100 to the end, the letter-frequency loss of episode 13. It learned nothing else.</p>""",
            """<p>The classic fix is <b>warmup</b>: start the learning rate near zero and raise it linearly over the first
300 steps. With warmup at 0.003, post-norm trains: <b>1.72</b>; pre-norm reaches 1.66. Post-norm transformers were
famously hard to train without warmup.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Post-norm needs warmup at higher learning rates; pre-norm does
not, though warmup can still help.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With 300 warmup steps and a peak learning rate of 0.003, what is the learning rate at "
                                    "step 150 (linear warmup, as in the code)?",
             "answer": "About 0.0015.", "why": "Halfway through the warmup: 0.003 × 151 / 300 ≈ 0.00151 (the code uses "
                                              "(step + 1) / warmup).", "key": {'parts': [{'label': None, 'value': 0.0015, 'tol': 5e-05, 'unit': None}]}},
            {"kind": "mc", "q": "Which run failed to learn?",
             "options": ["Pre-norm, lr 0.003, no warmup", "Post-norm, lr 0.001, no warmup",
                         "Post-norm, lr 0.003, no warmup", "Post-norm, lr 0.003, with warmup"],
             "answer": "C.", "why": "Stuck at 3.36; the other three reached 1.67–1.72.", "key": {'choice': 2}},
        ],
    },
    {
        "title": "Where it goes wrong, and why pre-norm won",
        "segment": (101, 139),
        "figures": [{"t": 118.9, "caption": "First 100 steps at lr 0.003: post-norm stalls at 3.38 by step 10; pre-norm keeps "
                                            "falling to 2.40."},
                    {"t": 137.6, "caption": "In code, the whole difference is where the norm sits."}],
        "body": [
            """<p>At initialization, in this small model, both versions get similar gradients at every layer (the last
layer's gradient is 0.9 times the first's in both). The trouble comes with the <b>first updates</b>: within ten steps
post-norm's training loss stalls at <b>3.38</b> and never recovers (3.28 at step 100), while pre-norm keeps falling (3.11
at step 10, 2.40 at step 100). Warmup makes those first updates small.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>That is why modern models use pre-norm: it trains stably at higher learning rates and with less careful
tuning, even when the network gets very deep. In code the difference is where the norm sits:
<code>x = x + block(norm(x))</code> versus <code>x = norm(x + block(x))</code>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Instability shows up in the first steps; pre-norm and warmup
both protect them.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Post-norm's loss stalled at about the letter-frequency value. What does that suggest the "
                                   "model ended up predicting?",
             "answer": "Roughly the same distribution of characters everywhere, ignoring the context.",
             "why": "A context-free prediction scores exactly the letter-frequency (unigram) loss, 3.35 here."},
            {"kind": "code", "q": "<b>Try it yourself.</b> In <code>pre_post_norm.py</code>, train post-norm at lr 0.003 "
                                  "with a shorter warmup of 50 steps. Does it still train?",
             "code": """print(train(pre=False, lr=3e-3, warmup=50))""",
             "answer": "Yes: 2.40 / 1.90 / 1.75 at 100 / 500 / 1,500 steps, between no warmup (stuck at 3.36) and 300 "
                       "steps (1.72).",
             "why": "Even a short warmup protects the first updates; how short it can be depends on the model and "
                    "learning rate."},
        ],
    },
]
