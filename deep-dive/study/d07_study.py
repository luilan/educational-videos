"""Study guide content for How LLMs Work: Deep Dive, episode 7: Causal Masking, in Detail.

Build:  python framework/study_guide.py deep-dive d07 --video deep-dive/media/videos/d07_scene/1080p60/CausalMaskVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d07_causal_mask/causal_mask.py (GPT-2 small with real weights; an 818,241-parameter tiny
GPT on Tiny Shakespeare; transformers 4.57.1, torch 2.14.0, CPU) or the same functions run on the exercise inputs.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 7",
    "title": "Causal Masking, in Detail",
    "tagline": "How a model is kept from reading the answer",
    "duration": "3:05",
    "intro": """<p>This lesson answers one question: during training a GPT sees the whole text at once, so what stops it
from reading the next word? The <b>causal mask</b>: before the softmax, every score for a future token is set to
<b>−∞</b>, so it gets exactly zero attention. The lesson checks it on GPT-2's real weights and trains the same tiny model
with and without the mask: without it, the model learns to copy instead of predict.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episodes 5–6 (attention, softmax) and Deep
Dive episode 4 (the mask leaks order). Code: <code>code/d07_causal_mask</code> (GPT-2 small, about 500 MB, plus two tiny
models trained in a few minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "The mask, by hand",
        "segment": (8, 45),
        "figures": [{"t": 21.5, "caption": "During training the whole text is visible: one triangle of numbers keeps "
                                           "each token from seeing the ones after it."},
                    {"t": 43.8, "caption": "Scores above the diagonal become −∞; softmax turns them into exactly 0, and "
                                           "each row still sums to 1."}],
        "body": [
            """<p>A GPT is trained to predict the next token, but during training it is given the whole text at once,
including every answer. What keeps it honest is the <b>causal mask</b>.</p>""",
            """<p>Take the attention scores for four tokens: each row is a query, each column a key. The cells above the
diagonal are the <b>future</b>. The mask sets those scores to <b>−∞</b>; then softmax turns each row into weights. Since
e<sup>−∞</sup> = 0, the future gets exactly zero weight, and each row still sums to 1. Row 2, for example, has scores
1.0 and 2.0 for the tokens it may see, giving weights 0.269 and 0.731.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Mask before the softmax, with −∞: the weights are then a
proper distribution over the past only.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In a causal mask for 64 tokens, how many of the 64 × 64 scores are set to −∞?",
             "answer": "2,016.", "why": "The strict upper triangle: 64 × 63 / 2 = 2,016; the other 2,080 are allowed.", "key": {'parts': [{'label': None, 'value': 2016, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "Row 3 has scores 0.5, 1.0, 2.0 for the tokens it may see. What weight goes to "
                                    "token 3?",
             "answer": "0.629.", "why": "e<sup>2</sup> / (e<sup>0.5</sup> + e<sup>1</sup> + e<sup>2</sup>) = 7.389 / "
                                       "11.75 ≈ 0.629, as on screen.", "key": {'parts': [{'label': None, 'value': 0.629, 'tol': 0.0005, 'unit': None}]}},
        ],
    },
    {
        "title": "Why −∞ and not 0",
        "segment": (45, 59),
        "figures": [{"t": 58.2, "caption": "With no mask, token 1 would give 63% of its attention to token 4, a word that "
                                           "hasn't happened yet."}],
        "body": [
            """<p>Why minus infinity, and not just zero? A score of zero still counts: e<sup>0</sup> = 1. Without any
mask, the first token's row (scores 2.0, 1.0, 0.5, 3.0) would put <b>63%</b> of its attention (0.631) on token 4, a word
that hasn't happened yet.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>“Zero score” is not “no attention”: only −∞ (or a very large
negative number) removes a token from the softmax.</div>""",
        ],
        "exercises": [
            {"kind": "code", "q": "<b>Try it yourself.</b> Fill the future with 0 instead of −∞. How much attention does "
                                  "token 1 now give to the three future tokens together?",
             "code": """masked0 = scores.masked_fill(mask, 0.0)
print(masked0.softmax(dim=-1).round(decimals=3))""",
             "answer": "About 29%: row 1 becomes 0.711, 0.096, 0.096, 0.096.",
             "why": "Each zero score still contributes e<sup>0</sup> = 1 to the softmax; only the last row, which has no "
                    "future, is unchanged."},
            {"kind": "tf", "q": "“Setting future scores to 0 is enough to hide the future.”",
             "answer": "False.", "why": "Softmax turns a score of 0 into a positive weight; you need −∞.", "key": {'value': False}},
        ],
    },
    {
        "title": "Checked on GPT-2",
        "segment": (59, 83),
        "figures": [{"t": 81.5, "caption": "Change “mat” to “moon”: earlier positions differ by exactly 0.0. Layer 1, head "
                                           "1: zeros above the diagonal."}],
        "body": [
            """<p>With GPT-2's real weights: <i>The cat sat on the mat</i>, then the same sentence ending in <i>moon</i>.
At every earlier position, the final-layer outputs are identical: a largest difference of exactly <b>0.0</b>. Only the
last position changes (by up to 14.7). In a real attention pattern from the first layer, everything above the diagonal
is zero, and the first token can only attend to itself (weight 1.00).</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>In a causal model the future cannot change the past: each
position's output depends only on the tokens up to it.</div>""",
        ],
        "exercises": [
            {"kind": "code", "q": "<b>Try it yourself.</b> Now change the <i>second</i> word: compare “The cat sat on the "
                                  "mat” with “The dog sat on the mat”. Which positions change?",
             "code": """b = tok("The dog sat on the mat", return_tensors="pt").input_ids
with torch.no_grad():
    hb = gpt2(b, output_hidden_states=True).hidden_states[-1][0]
print([(ha[i] - hb[i]).abs().max().item() for i in range(6)])""",
             "answer": "Position 0 (“The”) is unchanged (0.0); positions 1–5 all change (13.1, 1.8, 3.9, 1.9, 5.3).",
             "why": "Every later token can see “dog”, so all of them are affected; “The” cannot."},
            {"kind": "mc", "q": "In a causal model, the first token's attention weights in every head are…",
             "options": ["uniform over the sentence", "1.0 on itself", "0 everywhere", "1.0 on the last token"],
             "answer": "B.", "why": "It may only see itself, and softmax over one score gives 1.0.", "key": {'choice': 1}},
        ],
    },
    {
        "title": "Training without the mask",
        "segment": (83, 134),
        "figures": [{"t": 117.4, "caption": "Without the mask: training loss 0.04, but honest loss 7.36, worse than random "
                                            "guessing (4.17)."},
                    {"t": 133.3, "caption": "The masked model writes Shakespeare-like text; the unmasked one outputs 65 "
                                            "empty lines, then gibberish."}],
        "body": [
            """<p>The same tiny GPT (818,241 parameters) is trained twice for 1,500 steps on Shakespeare; the only
difference is one flag, the mask on or off. Without the mask, the training loss falls to <b>0.04</b>: the model simply
reads the next character. Tested honestly, given only the past (each prefix fed separately), its loss is <b>7.36</b>,
worse than guessing uniformly among 65 characters (ln 65 = 4.17). With the mask: <b>1.52</b> in training and <b>1.73</b>
on the honest test.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Asked to write, the masked model produces names, line breaks and almost-words. The unmasked one learned
to copy, not to predict: with nothing to copy, it outputs 65 empty lines, then gibberish.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A training loss is only meaningful if the model sees exactly
what it will see when used. Without the mask, the task becomes copying.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What loss does a model get by guessing uniformly among 65 characters?",
             "answer": "About 4.17.", "why": "−ln(1/65) = ln 65 ≈ 4.17; the unmasked model's 7.36 is worse because it is "
                                             "confidently wrong.", "key": {'parts': [{'label': None, 'value': 4.17, 'tol': 0.0834, 'unit': None}]}},
            {"kind": "short", "q": "Why is a very low training loss a warning sign here, not a success?",
             "answer": "The model could see the answer, so the loss measured copying, not prediction; the honest test shows "
                       "it cannot predict at all.",
             "why": "This is a form of data leakage: the target was part of the input."},
        ],
    },
    {
        "title": "Efficiency, inference, and the code",
        "segment": (134, 177),
        "figures": [{"t": 147.0, "caption": "Every position predicts the next character from the past only: one text gives "
                                            "as many examples as characters."},
                    {"t": 176.0, "caption": "The mask in one line, or one flag in PyTorch's attention function."}],
        "body": [
            """<p>The mask also makes training efficient. Each position gets its own prediction, using only what came
before, so one 64-character text yields <b>64 training examples</b> in a single pass.</p>""",
            """<p>At inference, the new token is always the last, so it may see everything before it: with a KV cache, a
single query against all cached keys needs no mask at all. Models like <b>BERT</b> drop the mask on purpose: they see
both sides, but cannot write left to right.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In code, the mask is one line, <code>masked_fill</code> of the upper triangle with −∞ before the softmax,
or a single flag: <code>F.scaled_dot_product_attention(q, k, v, is_causal=True)</code>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The mask lets a model learn every next-token prediction of a
text in parallel, while keeping each one honest.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A batch of 32 texts of 64 characters: how many next-character predictions are "
                                    "trained in one step?",
             "answer": "2,048.", "why": "32 × 64; every position of every text is a training example.", "key": {'parts': [{'label': None, 'value': 2048, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“When generating one new token with a KV cache, the causal mask must still be "
                                "applied.”",
             "answer": "False.", "why": "The single new query is the last position; every cached key is in its past, so "
                                       "nothing needs masking.", "key": {'value': False}},
            {"kind": "mc", "q": "Which model is trained without a causal mask?",
             "options": ["GPT-2", "Qwen2.5", "BERT", "The episode's masked tiny GPT"],
             "answer": "C.", "why": "BERT sees both sides of each token; it fills in blanks rather than writing left to "
                                   "right.", "key": {'choice': 2}},
        ],
    },
]
