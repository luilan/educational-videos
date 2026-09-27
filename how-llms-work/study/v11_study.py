"""Study guide content for How LLMs Work, episode 11: Training: Learning from Mistakes.

Build:  python framework/study_guide.py how-llms-work v11
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 11",
    "title": "Training: Learning from Mistakes",
    "tagline": "Loss, gradients, and millions of tiny nudges",
    "duration": "2:38",
    "intro": """<p>This lesson answers the question left open in episode 1: where do the model's numbers come from?
A new model starts with <b>random weights</b>. Training shows it text, measures how wrong each next-token guess was with
one number, the <b>loss</b>, and nudges every weight a tiny bit in the direction that makes the loss smaller. Then it
does that again, millions of times.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on episode 1 (a probability for every
token) and episode 10 (logits and softmax). The maths is in Foundations F05 (Exponentials and Logarithms) for the log in
the loss, F10 (Slopes and Gradients) for gradient descent, F12 (Neural Networks in Three Minutes) and F13 (PyTorch and
Autograd) for backpropagation and the code, and F14 (Train vs Validation Data) for the loss curve.</div>""",
}

CONCEPTS = [
    {
        "title": "The text is its own answer key",
        "segment": (8, 31),
        "figures": [{"t": 16.3, "caption": "A new model: random weights, and a real sample of its output."},
                    {"t": 30.0, "caption": "Hide the next token, let the model guess, compare with the truth."}],
        "body": [
            """<p>A freshly created model is <b>pure noise</b>: every weight is random, so it writes gibberish like
<code>lGk'kFYAe:cLs:QzoYSky</code> (the real untrained tiny GPT). <b>Training</b> makes those numbers useful.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The recipe: take a huge amount of text, <b>hide the next token</b>, let the model predict it, and
<b>compare its guess with the truth</b>. The text itself provides the answers, so nobody labels anything
(<b>self-supervised</b> learning). Every position is an exercise: from <i>“The”</i> predict <i>cat</i>, from
<i>“The cat”</i> predict <i>sat</i>, …</p>""",
            """<div class="box key"><b class="t">Key idea</b>Training data is just text: at every position the model
predicts the next token, and <b>the text itself says what the right answer was</b>.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Where do the correct answers for training come from?",
             "options": ["People label each sentence by hand", "The next token of the text itself",
                         "A bigger model grades each guess", "A dictionary of correct sentences"],
             "answer": "B.", "why": "Hide the next token and the text already contains the answer. That is why any "
                                   "text can be used, in huge amounts."},
            {"kind": "tf", "q": "“A freshly created model already writes rough English; training only fixes its "
                                "mistakes.”",
             "answer": "False.", "why": "Its weights are random, so it writes random characters "
                                        "(<code>lGk'kFYAe…</code>). Everything it knows comes from training."},
            {"kind": "number", "q": "<i>“The cat sat on the mat”</i> is 6 tokens. How many next-token exercises does "
                                    "it give?",
             "answer": "5.", "why": "Every token after the first is a target, predicted from all the tokens before it: "
                                   "cat, sat, on, the, mat."},
            {"kind": "short", "q": "For <i>“To be or not to be”</i>, what are the input and the correct answer of "
                                   "the 4th exercise?",
             "answer": "Input “To be or not”, answer “to”.",
             "why": "The exercises are To → be, To be → or, To be or → not, To be or not → to, …"},
        ],
    },
    {
        "title": "Cross-entropy: scoring a guess",
        "segment": (31, 57),
        "figures": [{"t": 47.5, "caption": "90 % for “mat” gives a loss of about 0.1; 1 % gives about 4.6."},
                    {"t": 56.5, "caption": "One loss per position, averaged over the batch (illustrative)."}],
        "body": [
            """<p>To learn, the model needs a number for how wrong it was: the <b>cross-entropy loss</b>,
<b>loss = −ln p(correct token)</b>, the negative natural log of the probability it gave to the right token.
<i>mat</i> at 90 % gives −ln 0.9 ≈ 0.105; at 1 %, −ln 0.01 ≈ 4.605.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The loss is 0 only for a certain, correct guess (p = 1) and grows without limit as p → 0, so
confident mistakes cost a lot. We <b>average</b> it over every position in a <b>batch</b>. Lower is better: training is
the <b>search for weights that make it small</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b><b>loss = −ln p(correct token)</b>, averaged over every
position in the batch: near 0 when confident and right, large when the right token got a tiny probability.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The model gave the correct token <i>mat</i> a probability of 0.5. What is the loss?",
             "answer": "≈ 0.69.", "why": "−ln 0.5 = 0.693: between the video's 0.1 (p = 0.9) and 4.6 (p = 0.01)."},
            {"kind": "mc", "q": "The correct next token is <i>mat</i>. Which prediction has the <b>largest</b> loss?",
             "options": ["p(mat) = 0.9", "p(mat) = 0.3, p(rug) = 0.6", "p(mat) = 0.05, p(rug) = 0.9",
                         "p(mat) = 0.5, p(rug) = 0.5"],
             "answer": "C.", "why": "Only p(mat) counts: −ln 0.05 ≈ 3.0, against 0.11, 1.20 and 0.69. How the rest of "
                                   "the probability is spread does not change the loss."},
            {"kind": "tf", "q": "“A cross-entropy loss can be negative if the model is very sure and right.”",
             "answer": "False.", "why": "p is at most 1, so ln p ≤ 0 and −ln p ≥ 0. The best possible loss is "
                                        "exactly 0, at p = 1."},
            {"kind": "number", "q": "Three positions; the correct tokens got probabilities 0.9, 0.5 and 0.01. What is "
                                    "the average loss?",
             "answer": "≈ 1.80.", "why": "(0.105 + 0.693 + 4.605) / 3 = 1.80. The one bad prediction contributes most "
                                        "of it."},
        ],
    },
    {
        "title": "Gradient descent: feeling our way downhill",
        "segment": (57, 83),
        "figures": [{"t": 69.5, "caption": "The loss landscape (illustrative)."},
                    {"t": 82.3, "caption": "Gradient descent: step after step, downhill."}],
        "body": [
            """<p>Picture the loss as a <b>landscape</b>: each point is one setting of <b>all</b> the weights, its
height the loss. We want a <b>deep valley</b>, but with millions of weights (millions of dimensions) we cannot look
around: we feel our way downhill.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The <b>gradient</b> gives one slope per weight: which way the loss goes <b>up</b>, and how steeply.
Every weight steps a little the other way, <b>w ← w − step size × gradient</b>, again and again:
<b>gradient descent</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The gradient points uphill; <b>gradient descent</b> steps
every weight a little <b>against</b> it, over and over.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "What does one point of the loss landscape stand for, and what does its height mean?",
             "answer": "One setting of all the weights; its height is the loss with those weights.",
             "why": "A real model's landscape has one direction per weight: millions of dimensions."},
            {"kind": "mc", "q": "For one weight, the gradient is +2.0. What does that tell us?",
             "options": ["Set this weight to 2.0", "Increasing this weight raises the loss, so decrease it",
                         "The loss is 2.0", "This weight is twice as important as the others"],
             "answer": "B.", "why": "The sign says which way is uphill, the size how steep. We step the other way."},
            {"kind": "number", "q": "Step size 0.1. New value of (a) w = 0.50, gradient +2.0; (b) w = −1.00, gradient −3.0?",
             "lines": 2,
             "answer": "(a) 0.30. (b) −0.70.",
             "why": "w − 0.1 × gradient: 0.50 − 0.20 = 0.30 and −1.00 + 0.30 = −0.70. A negative gradient means "
                    "the weight goes up."},
            {"kind": "tf", "q": "“For a real LLM we could just try every setting of the weights and keep the best.”",
             "answer": "False.", "why": "With millions of weights there are far too many settings. We only feel the "
                                        "slope where we stand, and step downhill."},
        ],
    },
    {
        "title": "The learning rate",
        "segment": (83, 91),
        "figures": [{"t": 90.9, "size": "small", "caption": "Too small: tiny steps, forever. Too large: overshoot and bounce."}],
        "body": [
            """<p>The size of the step is the <b>learning rate</b>. <b>Too small</b>, and training takes forever,
creeping down the slope. <b>Too large</b>, and each step <b>overshoots</b> the bottom of the valley, lands on the other
side, and training bounces around instead of settling.</p>""",
            "{fig0}",
            """<p>A toy example: one weight with loss = w², whose slope at w is 2w. From w = 1, a learning rate of 0.1
gives 1 → 0.8 → 0.64 → …, creeping towards the bottom at 0. A learning rate of 1.1 gives 1 → −1.2 → 1.44 → …: every
jump crosses the valley and is bigger than the last.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The <b>learning rate</b> is the step size. Too small:
slow. Too large: overshoot and bounce.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "A run's loss goes down steadily, but very slowly. What is the most likely fix?",
             "options": ["Lower the learning rate", "Raise the learning rate a bit", "Train on less text",
                         "Stop computing gradients"],
             "answer": "B.", "why": "Steady but slow progress is the sign of steps that are too small."},
            {"kind": "tf", "q": "“If the loss jumps up and down and never settles, the learning rate is too small.”",
             "answer": "False.", "why": "That is the sign of a learning rate that is too <i>large</i>: every step "
                                        "overshoots the valley."},
            {"kind": "number", "q": "Same toy loss w² (slope 2w), now starting at w = 3. Where is w after one step "
                                    "with a learning rate of (a) 0.1 and (b) 1.5?",
             "lines": 2,
             "answer": "(a) 2.4. (b) −6.",
             "why": "The slope at 3 is 6. (a) 3 − 0.1 × 6 = 2.4, a bit closer to 0. (b) 3 − 1.5 × 6 = −6: across "
                    "the valley and further from the bottom than before."},
            {"kind": "number", "q": "Same toy loss, from w = 1, with a learning rate of 0.5. Where is w after one step?",
             "answer": "0.", "why": "1 − 0.5 × 2 = 0, exactly the bottom. On this toy bowl 0.5 is a perfect step "
                                   "size; real landscapes are never this simple."},
        ],
    },
    {
        "title": "Backpropagation",
        "segment": (91, 109),
        "figures": [{"t": 108.5, "caption": "Forward pass left to right; the backward pass carries gradients back from "
                                            "the loss, at about twice the cost."}],
        "body": [
            """<p>Gradient descent needs a gradient for <b>every</b> weight, millions of them, at every step.
<b>Backpropagation</b> gets them all at once.</p>""",
            "{fig0}",
            """<p>The <b>forward pass</b> runs the model from the embedding through the blocks and logits to the loss,
keeping its intermediate results. The <b>backward pass</b> then starts at the <b>loss</b> and works <b>backwards</b>
through every layer, applying the <b>chain rule</b> from calculus: multiply the slopes, layer by layer. Reusing the
saved results, it costs only about <b>twice</b> as much as a forward pass.</p>""",
            """<div class="box key"><b class="t">Key idea</b><b>Backpropagation</b> = the chain rule, applied from the
loss backwards through every layer. One forward pass plus one backward pass (≈ 2× the forward cost) gives the gradient
of every weight.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put these in the order the backward pass works through them: <i>embedding · "
                                   "logits · loss · blocks</i>.",
             "answer": "loss → logits → blocks → embedding.",
             "why": "It starts where the forward pass ended, at the loss, and works back towards the input."},
            {"kind": "number", "q": "Chain rule: the loss changes 3 times as fast as a logit, and that logit changes "
                                    "0.5 times as fast as one weight. What is the loss's gradient for that weight?",
             "answer": "1.5.", "why": "Multiply the slopes along the chain: 3 × 0.5 = 1.5."},
            {"kind": "number", "q": "A forward pass takes 10 ms. Roughly how long does a training step (forward + "
                                    "backward) take?",
             "answer": "About 30 ms.", "why": "The backward pass costs about twice the forward pass: 10 + 20 = 30 ms. "
                                             "The weight update itself is cheap."},
            {"kind": "tf", "q": "“Backpropagation finds each weight's gradient by nudging it and re-running the model.”",
             "answer": "False.", "why": "That would need one run per weight, millions per step. Backpropagation gets "
                                        "every gradient from a single backward pass."},
        ],
    },
    {
        "title": "The training loop in code",
        "segment": (109, 132),
        "figures": [{"t": 124.0, "caption": "The loop: batch, forward, loss, backward, step. Adam gives each weight "
                                            "its own step size (illustrative sizes)."},
                    {"t": 131.5, "caption": "The same loop in PyTorch: forward pass, loss, backward pass, step."}],
        "body": [
            """<p>Training repeats one cycle: take a <b>batch</b> of text, run the <b>forward pass</b>, compute the
<b>loss</b>, run the <b>backward pass</b>, take a small <b>step</b>. Over and over, across billions of tokens. In
practice the step uses a smarter update rule, <b>Adam</b>, which adapts the step size for each weight.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<pre class="code">for step in range(max_steps):
    x, y = get_batch(train_data)     # text, and the next tokens
    logits = model(x)                # forward pass
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    opt.zero_grad()
    loss.backward()                  # backpropagation
    opt.step()                       # nudge every weight</pre>""",
            """<p>PyTorch does the calculus: <code>loss.backward()</code> fills in the gradient of every weight, and
<code>opt.step()</code> (the optimizer, such as Adam) moves every weight. <code>opt.zero_grad()</code> first clears the
previous step's gradients, because PyTorch adds new gradients to old ones.</p>""",
            """<div class="box key"><b class="t">Key idea</b>One training step = <b>batch → forward → loss → backward →
step</b>. In PyTorch: <code>model(x)</code>, <code>F.cross_entropy</code>, <code>loss.backward()</code>,
<code>opt.step()</code>.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put one training step in order: <i>step · loss · batch · backward pass · "
                                   "forward pass</i>.",
             "answer": "batch → forward pass → loss → backward pass → step.",
             "why": "The loss needs the forward pass, the gradients need the loss, and the step needs the gradients."},
            {"kind": "mc", "q": "What does Adam do differently from plain gradient descent?",
             "options": ["It measures the loss in a different way", "It adapts the step size for each weight",
                         "It no longer needs gradients", "It uses a bigger batch of text"],
             "answer": "B.", "why": "It still follows the gradients, but each weight gets its own step size."},
            {"kind": "short", "q": "In the code above, which line does backpropagation, and which line actually "
                                   "changes the weights?",
             "answer": "<code>loss.backward()</code>, then <code>opt.step()</code>.",
             "why": "<code>backward()</code> only computes gradients; the weights move in <code>opt.step()</code>."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Here is the same loop for a “model” with a single weight "
                                  "<code>w</code> and the loss (w − 1)², lowest at w = 1. Its gradient is 2(w − 1). "
                                  "(a) Before running, predict the three printed lines (step, w, loss, gradient). "
                                  "(b) Delete the line <code>opt.zero_grad()</code> and run again. Which numbers change, "
                                  "and why? (c) Put it back and replace <code>SGD</code> with "
                                  "<code>torch.optim.Adam([w], lr=0.1)</code>. How far does w move in the first step, "
                                  "compared with SGD?",
             "code": """import torch

w = torch.tensor(3.0, requires_grad=True)     # one weight, a bad start
opt = torch.optim.SGD([w], lr=0.1)            # plain gradient descent

for step in range(3):
    loss = (w - 1.0) ** 2                     # forward pass + loss (lowest at w = 1)
    opt.zero_grad()
    loss.backward()                           # backpropagation: fills in w.grad
    print(step, round(w.item(), 3), round(loss.item(), 3), round(w.grad.item(), 3))
    opt.step()                                # w <- w - lr * gradient""",
             "answer": "(a) <code>0 3.0 4.0 4.0</code>, <code>1 2.6 2.56 3.2</code>, <code>2 2.28 1.638 2.56</code> "
                       "· (b) the gradients become 7.2 and 8.96 · (c) 0.1 instead of 0.4",
             "why": """(a) At w = 3 the gradient is 2 × 2 = 4, so w moves 0.1 × 4 = 0.4 to 2.6; then 0.32 to 2.28.
The steps shrink as the slope flattens. (b) Without <code>zero_grad</code> the gradients pile up: 4 + 3.2 = 7.2, then
7.2 + 1.76 = 8.96, so the steps are too big (w jumps to 1.88). (c) w becomes 2.9: Adam's first step is about the
learning rate, 0.1, however steep the slope, because it scales each weight's step by that weight's own gradients."""},
        ],
    },
    {
        "title": "A real loss curve",
        "segment": (132, 147),
        "figures": [{"t": 146.7, "caption": "The real training run of the tiny GPT (818,241 parameters)."}],
        "body": [
            """<p>This is the real loss curve of the tiny GPT of episode 12, which predicts one of 65 characters at a
time. It starts at about <b>4.4</b>: no better than guessing. Giving all 65 characters the same probability, p = 1/65,
gives a loss of ln 65 ≈ 4.17 (the dashed line).</p>""",
            "{fig0}",
            """<p>It falls <b>fast at first</b>, <b>then slower</b>, and ends near <b>1.59</b>: the
<b>validation loss</b>, measured on text the model <b>never trained on</b>. It shows what the model learned, not what it
memorised.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Start ≈ 4.4 (guessing ≈ 4.17), a fast drop, a slow tail,
end ≈ 1.59 on text the model never saw: the <b>validation loss</b>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "All 65 characters get the same probability. What is the loss of each prediction?",
             "answer": "≈ 4.17.", "why": "p = 1/65 ≈ 0.015 and −ln(1/65) = ln 65 ≈ 4.17: the dashed “random "
                                        "guessing” line."},
            {"kind": "number", "q": "The loss went 4.41 → 2.20 in 250 steps, then → 1.59 by step 5,000. How big is each "
                                    "drop?",
             "answer": "2.21, then only 0.61.", "why": "Fast at first, then slower: the easy patterns are learned "
                                                       "first."},
            {"kind": "mc", "q": "Why is the validation loss measured on text the model never trained on?",
             "options": ["It is faster to compute", "To see if what it learned carries over to new text, rather "
                         "than being memorised", "The training text has no correct answers",
                         "To make the loss look smaller"],
             "answer": "B.", "why": "A model could score well on text it has memorised. New text shows what it really "
                                   "learned."},
            {"kind": "number", "q": "Which p, given to the right character every time, gives a loss of 1.59? Compare "
                                    "with guessing.",
             "answer": "p ≈ 0.20 (1 in 5).", "why": "−ln p = 1.59, so p = e<sup>−1.59</sup> ≈ 0.20, against "
                                                    "1/65 ≈ 0.015 for guessing: about 13 times better."},
        ],
    },
]
