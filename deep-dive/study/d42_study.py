"""Study guide content for How LLMs Work: Deep Dive, episode 42: Induction Heads.

Build:  python framework/study_guide.py deep-dive d42 --video deep-dive/media/videos/d42_scene/1080p60/InductionVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d42_induction_heads/induction.py (GPT-2 small, eager attention, 8 seeded sequences of 50
random tokens repeated twice; heads removed with head_mask) or from the arithmetic shown.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 42",
    "title": "Induction Heads",
    "tagline": "The circuit that copies from context",
    "duration": "2:11",
    "intro": """<p>This lesson answers one question: how does a transformer copy from its context? GPT-2 can't predict 50
random tokens (loss 12.8), but when they repeat, it predicts the second copy almost perfectly (loss 0.25). The work is done
by <b>induction heads</b>: attention heads that find the earlier occurrence of the current token and look at what came
next. Scoring all 144 heads finds five strong ones in layers 5–7; removing them destroys the copying, while removing
random heads barely matters.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 8 (the attention matrix) and 13 (the
residual stream). Code: <code>code/d42_induction_heads</code> (GPT-2 small).</div>""",
}

CONCEPTS = [
    {
        "title": "Copying from context",
        "segment": (8, 47),
        "figures": [{"t": 37.0, "caption": "50 random tokens, then the same 50 again: loss 12.8 on the first copy, 0.25 on the "
                                           "second."},
                    {"t": 46.0, "caption": "The induction rule: the current token is A; find the earlier A; predict the token "
                                           "that followed it, B."}],
        "body": [
            """<p>Random tokens are unpredictable: on 50 of them GPT-2's loss is <b>12.8</b>. Repeat the same 50 tokens,
and on the second copy the loss is <b>0.25</b>. The model copies almost perfectly from its context. The rule it applies:
the current token is A; find where A appeared before; look at the token right after it, B; predict B.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Repeated random tokens isolate copying: nothing but the context
can predict them.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A loss of 0.25 means the model gives the right token what probability, on average "
                                    "(geometric mean)? And a loss of 12.8?",
             "answer": "About 0.78; about 0.000003.", "why": "e^−0.25 ≈ 0.78; e^−12.8 ≈ 2.8 × 10⁻⁶."},
        ],
    },
    {
        "title": "Finding the heads",
        "segment": (47, 99),
        "figures": [{"t": 62.4, "caption": "Induction score of all 144 heads: five above 0.8, all in layers 5–7 (5.5, 7.10, 6.9, "
                                           "5.1, 7.2)."},
                    {"t": 78.5, "caption": "The circuit: an earlier head records each token's predecessor; the induction head "
                                           "searches for it (Olsson et al., 2022)."},
                    {"t": 97.5, "caption": "Second-copy loss with heads removed: top 2/4/6 induction heads 0.35/1.78/5.81; "
                                           "random heads 0.29/0.38/0.66."}],
        "body": [
            """<p>An induction head is an attention head that performs that lookup. Its <b>induction score</b>: in the
second copy, the average attention from each token to the token right after its earlier occurrence. Of GPT-2's 144 heads
most score near zero, 15 score above 0.3, and five score above 0.8: 5.5 (0.94), 7.10 (0.92), 6.9 (0.91), 5.1 (0.91) and
7.2 (0.85).</p>""",
            "{fig0}",
            """<p>Why not in layer 0? To find “the token after A”, each position must first know what its previous token
was. In the original analysis, an earlier <b>previous-token head</b> writes that into the residual stream and the
induction head reads it: a circuit of two heads in different layers. (On these random sequences no single early GPT-2 head
scores strongly for attending to the previous token, so that half is spread out or done differently here.)</p>""",
            "{fig1}",
            """<p>Removing heads (head_mask = 0) tests their role. The top 2 induction heads: second-copy loss 0.25 →
0.35; the top 4: 1.78; the top 6: 5.81. Removing 6 random heads instead: 0.66 on average.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Score heads by a behaviour, then ablate them: a high score plus
a large ablation effect identifies a mechanism.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In the second copy, position t (counting from 0, with T = 50 tokens per copy) holds the "
                                    "same token as position t − 50. Which position should an induction head attend to?",
             "answer": "t − 49.", "why": "The token after the earlier occurrence: (t − 50) + 1."},
            {"kind": "mc", "q": "Why compare removing the top induction heads with removing random heads?",
             "options": ["To speed up the experiment", "To show the effect comes from those heads, not from removing any "
                                                         "6 heads", "Because random heads are induction heads",
                         "To measure the first copy"],
             "answer": "B.", "why": "Any ablation hurts a little; the control shows how much is specific."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Compute one head's induction score.",
             "code": """import torch
from transformers import GPT2LMHeadModel
m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
seq = torch.randint(1000, 30000, (4, 50)); ids = torch.cat([seq, seq], 1)
a = m(ids, output_attentions=True).attentions[5][:, 5]
print(sum(a[:, t, t - 49].mean() for t in range(50, 100)) / 50)""",
             "answer": "About 0.9 (head 5.5).", "why": "Head 5.5 scored 0.94 in the episode's run."},
        ],
    },
    {
        "title": "Why it matters",
        "segment": (99, 122),
        "figures": [{"t": 111.5, "caption": "Induction heads: a building block of in-context learning."}],
        "body": [
            """<p>Induction heads are a building block of <b>in-context learning</b>: using patterns from earlier in the
prompt. Research on small models (Olsson et al., 2022) found that they form suddenly during training, at the same moment
in-context learning improves. In code, the score is a single average over attention weights.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Some model behaviours are carried by small, findable
circuits.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“An induction head can copy a token sequence it never saw during training.”",
             "answer": "True.", "why": "The sequences here are random; the head copies whatever the context contains."},
        ],
    },
]
