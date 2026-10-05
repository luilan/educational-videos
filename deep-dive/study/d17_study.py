"""Study guide content for How LLMs Work: Deep Dive, episode 17: Cross-Entropy, Deeper.

Build:  python framework/study_guide.py deep-dive d17 --video deep-dive/media/videos/d17_scene/1080p60/CrossEntropyVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d17_cross_entropy/cross_entropy.py (GPT-2 small, real weights; transformers 4.57.1, torch
2.14.0, CPU) or the same formulas on the exercise inputs.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 17",
    "title": "Cross-Entropy, Deeper",
    "tagline": "What the loss number really measures",
    "duration": "2:32",
    "intro": """<p>This lesson answers one question: what exactly is the “loss” every training run prints? It is the
average of <b>−ln(probability of the right token)</b>. The same number can be read as <b>perplexity</b> (e<sup>loss</sup>)
or as <b>bits</b> (loss ÷ ln 2). The log punishes confident mistakes hard, rewards honest probabilities (GPT-2 is well
calibrated, except for one formatting habit), and its gradient is beautifully simple: softmax minus one-hot.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episode 10 (training) and Foundations
(logarithms, softmax). Code: <code>code/d17_cross_entropy</code> (GPT-2 small, about 500 MB).</div>""",
}

CONCEPTS = [
    {
        "title": "The loss, token by token",
        "segment": (8, 47),
        "figures": [{"t": 40.5, "caption": "GPT-2 on two sentences: “Paris” gets 3% (loss 3.43); by the second sentence "
                                           "“Rome” gets 58% (0.55)."},
                    {"t": 46.4, "caption": "The printed loss is the average over the tokens: 2.344."}],
        "body": [
            """<p>The loss of one token is <b>−ln(p)</b>, where p is the probability the model gave to the token that
actually came next. GPT-2 on “The capital of France is Paris, and the capital of Italy is Rome.”: after “The capital of
France is”, “Paris” gets <b>3%</b> (loss 3.43). By the second sentence the model has caught on: “Rome” gets <b>58%</b>
(loss 0.55), and “ of” after “ capital” 64% (0.45).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The number training prints is the <b>average</b> over all predicted tokens: 2.344 for this sentence
(14 predictions).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Cross-entropy = average of −ln(probability of the right
token). Lower is better; 0 means certainty and correctness.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The model gives the right token 25%. What is its loss?",
             "answer": "About 1.39.", "why": "−ln 0.25 = ln 4 ≈ 1.386."},
            {"kind": "tf", "q": "“A loss of 0 is possible only if the model gives the right token 100%.”",
             "answer": "True.", "why": "−ln p = 0 only when p = 1."},
        ],
    },
    {
        "title": "Perplexity, bits, and confident mistakes",
        "segment": (47, 87),
        "figures": [{"t": 66.4, "caption": "Perplexity e^2.344 = 10.4; 3.38 bits per token; Shakespeare: 1.87 bits per "
                                           "character."},
                    {"t": 85.6, "caption": "−ln p explodes near zero: 90% → 0.11, 10% → 2.30, 1% → 4.61, 0.1% → 6.91."}],
        "body": [
            """<p>Two other names for the same number. <b>Perplexity</b> = e<sup>loss</sup> = e<sup>2.344</sup> =
<b>10.4</b>: as if the model were choosing evenly among about 10 tokens at each step. <b>Bits</b> = loss ÷ ln 2 =
<b>3.38</b> bits per token. On 1,024 tokens of Shakespeare GPT-2's loss is 4.000 (perplexity 54.6), which is
<b>1.87 bits per character</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The log punishes confident mistakes hard: right token at 90% → 0.11, 10% → 2.30, 1% → 4.61, 0.1% →
6.91. On Shakespeare, the worst 10% of tokens make up <b>32%</b> of the total loss.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Perplexity and bits are the same measurement in different
units; a few very wrong predictions can dominate the average.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A model's loss is 1.0. What are its perplexity and its bits per token?",
             "answer": "Perplexity about 2.72; about 1.44 bits.", "why": "e¹ ≈ 2.718; 1 / ln 2 ≈ 1.443."},
            {"kind": "number", "q": "Perplexity 54.6 on Shakespeare. What loss is that?",
             "answer": "4.0.", "why": "ln 54.6 ≈ 4.000, matching the measured loss."},
            {"kind": "short", "q": "Why does a model that is wrong but unsure lose less than one that is wrong and sure?",
             "answer": "The loss depends only on the probability given to the right token; an unsure model leaves more "
                       "probability for it, so −ln p is smaller.",
             "why": "Confidence in the wrong answer means little probability left for the right one."},
        ],
    },
    {
        "title": "Calibration",
        "segment": (87, 116),
        "figures": [{"t": 99.6, "caption": "When GPT-2's top guess has ~30% probability, it is right 33% of the time; ~70%, "
                                           "right 67%."},
                    {"t": 115.4, "caption": "At 96% confidence it is right only 67%: 95% of those mistakes are a line break "
                                            "the file doesn't have."}],
        "body": [
            """<p>Because the log rewards honest probabilities, a model trained on cross-entropy tends to be
<b>calibrated</b>: when its top guess has about 30% probability, GPT-2 is right 33% of the time; around 70%, right
67%. Buckets on 1,023 Shakespeare tokens: confidence 0.11 → right 0.14, 0.29 → 0.33, 0.49 → 0.46, 0.70 → 0.67.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Except at the top: in the 244 tokens where GPT-2 is on average <b>96%</b> sure, it is right only
<b>67%</b> of the time. Of the 81 confident mistakes, 95% are the same one: GPT-2 expects a blank line after each line
of Shakespeare (a line break token), and this file has none. A mismatch between the text and what the model is used to,
not random overconfidence.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Cross-entropy rewards calibrated probabilities; systematic
miscalibration often points to a mismatch between the data and what the model expects.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "A perfectly calibrated model says 40% for its top guess on 500 tokens. About how many of "
                                "those guesses are right?",
             "options": ["50", "200", "400", "500"],
             "answer": "B.", "why": "40% of 500 = 200."},
            {"kind": "tf", "q": "“GPT-2 was overconfident on Shakespeare at every confidence level.”",
             "answer": "False.", "why": "Below 0.8 it was close to calibrated (even slightly underconfident at 0.11 → "
                                       "0.14 and 0.29 → 0.33); only the top bucket was far off."},
        ],
    },
    {
        "title": "The gradient, and the code",
        "segment": (116, 143),
        "figures": [{"t": 132.0, "caption": "Gradient w.r.t. the logits = softmax − one-hot; autograd agrees to every digit "
                                            "shown."},
                    {"t": 142.0, "caption": "log_softmax, pick the right token, negate, average: what F.cross_entropy "
                                            "does."}],
        "body": [
            """<p>For each logit, the gradient of the loss is the predicted probability, minus 1 for the right token:
<b>softmax − one-hot</b>. With 5 logits and the right token at index 2: [+0.612, +0.098, −0.985, +0.231, +0.044], exactly
what autograd computes. The right token's logit is pushed up, every other logit down in proportion to its probability.
That is the signal that starts every backward pass.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>∂loss/∂logitᵢ = pᵢ − 1[i = right]: simple, bounded between
−1 and 1, and cheap to compute.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Softmax gives the right token 0.7. What is the gradient for its logit? For a wrong "
                                    "token with 0.2?",
             "answer": "−0.3 and +0.2.", "why": "0.7 − 1 and 0.2 − 0."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Check that the gradient sums to zero over the logits, for any "
                                  "input.",
             "code": """z = torch.randn(5, requires_grad=True)
F.cross_entropy(z[None], torch.tensor([2])).backward()
print(z.grad.sum())""",
             "answer": "0 (up to rounding, about 10⁻⁸).", "why": "Softmax sums to 1 and the one-hot sums to 1, so their "
                                                                "difference sums to 0."},
        ],
    },
]
