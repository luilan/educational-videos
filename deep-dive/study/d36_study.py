"""Study guide content for How LLMs Work: Deep Dive, episode 36: Supervised Fine-Tuning.

Build:  python framework/study_guide.py deep-dive d36 --video deep-dive/media/videos/d36_scene/1080p60/SFTVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d36_sft/sft.py (Qwen2.5-0.5B base, float32, full fine-tuning with AdamW lr 1e-5, 75 steps of 8
examples, CPU; two runs gave identical results) or from the arithmetic shown.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 36",
    "title": "Supervised Fine-Tuning",
    "tagline": "From text predictor to assistant",
    "duration": "2:08",
    "intro": """<p>This lesson answers one question: how does a text predictor become an assistant? Asked questions in
chat format, the base Qwen2.5-0.5B never answers in the right form and never stops, although with a plain prompt it
knows 75% of the answers. <b>Supervised fine-tuning</b> on 280 example conversations, with the loss only on the
assistant's tokens, takes it to 95% exact answers on held-out questions, and it stops every time. Its three mistakes show
what fine-tuning does not add: knowledge and skills.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 17 (cross-entropy) and 19 (Adam/AdamW);
LLMs in Practice episodes 1 (prompts are just context) and 9 (fine-tuning and LoRA). Code: <code>code/d36_sft</code> (Qwen2.5-0.5B, about 8 GB of RAM to
train).</div>""",
}

CONCEPTS = [
    {
        "title": "What the base model lacks",
        "segment": (8, 42),
        "figures": [{"t": 41.5, "caption": "60 questions in chat format: right form 0%, stops 0%. With a plain “Question: … "
                                           "Answer:” prompt: the right answer in 75%."}],
        "body": [
            """<p>A base model only continues text. Asked 60 questions in the chat format (special tokens marking user
and assistant), the base Qwen2.5-0.5B gives <b>0%</b> answers in the right form and <b>never stops</b> within 40 tokens:
it repeats the question and emits junk tokens. With a plain prompt, “Question: What is 63 + 34?\\nAnswer:”, the same
model has the right answer in its first line <b>75%</b> of the time (40% in exactly the expected form).</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>The knowledge is learned in pre-training; what's missing is
the behaviour: answer in the expected form, then stop.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“The base model fails in chat format mainly because it does not know the capitals.”",
             "answer": "False.", "why": "With a plain prompt it gets 75% right; the chat tokens are simply unfamiliar "
                                       "territory for it.", "key": {'value': False}},
        ],
    },
    {
        "title": "Data and the loss mask",
        "segment": (42, 84),
        "figures": [{"t": 56.0, "caption": "280 conversations of 3 checkable tasks, in Qwen's chat template."},
                    {"t": 73.0, "caption": "Prompt tokens get label −100 (no loss); the answer and its end token are trained: 12 of "
                                           "30 tokens here, 31% of a batch."},
                    {"t": 82.6, "caption": "Answer loss over 75 steps of 8: 4.24 → 0.14."}],
        "body": [
            """<p>The training data are examples of the wanted behaviour: 280 short conversations (100 additions, 30
capitals × 3, 30 words in capitals × 3) in the chat template:
<code>&lt;|im_start|&gt;user … &lt;|im_end|&gt; &lt;|im_start|&gt;assistant … &lt;|im_end|&gt;</code>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Training is ordinary next-token prediction with one twist: the loss counts only the assistant's tokens.
Every other label is set to −100, which cross-entropy ignores. In one example that leaves 12 of 30 tokens; in a batch,
31%. The model is not taught to write questions, only to answer them, and because the end token belongs to the answer, it
learns to stop. Full fine-tuning, 75 steps of 8 examples, took the answer loss from 4.24 to 0.14 in a few minutes on a
CPU.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>SFT = next-token prediction on curated conversations, with the
loss masked to the assistant's part, end token included.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A conversation has 18 prompt tokens and a 12-token answer (end token included). What "
                                    "share of the tokens contribute to the loss?",
             "answer": "40%.", "why": "12 / 30.", "key": {'parts': [{'label': None, 'value': 40, 'tol': 0.5, 'unit': '%'}]}},
            {"kind": "short", "q": "What would go wrong if the end token were left out of the trained labels?",
             "answer": "The model would never learn to emit it, so it would keep generating after the answer.",
             "why": "Stopping is a learned behaviour like any other token."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Build masked labels and check which positions count.",
             "code": """import torch, torch.nn.functional as F
prompt, answer = [5, 6, 7, 8], [9, 10, 2]
labels = torch.tensor([-100] * len(prompt) + answer)
logits = torch.randn(len(labels), 12)
print(F.cross_entropy(logits, labels, ignore_index=-100))
print((labels != -100).sum())""",
             "answer": "A loss averaged over 3 positions.", "why": "Positions labelled −100 are skipped in both the sum and "
                                                                  "the average."},
        ],
    },
    {
        "title": "Results, and what SFT doesn't add",
        "segment": (84, 119),
        "figures": [{"t": 95.2, "caption": "60 held-out questions: exact answers 0% → 95%; stops 0% → 100%; length 40 → 7 "
                                           "tokens."},
                    {"t": 109.8, "caption": "The 3 misses: 48 + 45 = 113, “Kiev”, FLOWER → “FLORAL”."}],
        "body": [
            """<p>On 60 held-out questions, with numbers, countries and words never seen in training: <b>95%</b> exact
answers, stopping <b>every time</b> after 7 tokens on average. The three misses: “48 + 45 = 113”, “The capital of Ukraine
is Kiev” (the older spelling, so the expected answer “Kyiv” is missed), and “flower” in capitals becoming “FLORAL”.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Fine-tuning teaches format and behaviour quickly; facts and
skills still come from pre-training, and its errors show through.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "The held-out countries never appeared in the fine-tuning data, yet most answers are right. "
                                "Why?",
             "options": ["The test leaked into training", "The model already knew them from pre-training; SFT taught it how "
                                                         "to answer", "Fine-tuning taught it geography",
                         "It guessed"],
             "answer": "B.", "why": "The plain-prompt test showed the knowledge was there before fine-tuning.", "key": {'choice': 1}},
        ],
    },
]
