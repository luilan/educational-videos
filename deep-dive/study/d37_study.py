"""Study guide content for How LLMs Work: Deep Dive, episode 37: Reward Models.

Build:  python framework/study_guide.py deep-dive d37 --video deep-dive/media/videos/d37_scene/1080p60/RewardModelVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d37_reward_model/reward_model.py (frozen Qwen2.5-0.5B, layer-12 hidden state of the last
token, linear head trained with Adam on the Bradley-Terry loss; held-out countries and sums) or from the arithmetic shown.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 37",
    "title": "Reward Models",
    "tagline": "Turning preferences into a number",
    "duration": "2:21",
    "intro": """<p>This lesson answers one question: how do we turn “answer A is better than B” into something a model
can be trained against? A <b>reward model</b> reads a conversation and outputs one number; it is trained on comparisons
with the <b>Bradley–Terry</b> loss. Built on a frozen Qwen2.5-0.5B, it picks the right capital for 98% of unseen
countries, but judges sums barely better than a coin, and when the preferred answers in training always end with “I hope
this helps!”, it learns the phrase instead of the facts: the seed of <b>reward hacking</b>.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 17 (cross-entropy) and 36 (supervised
fine-tuning). Code: <code>code/d37_reward_model</code> (Qwen2.5-0.5B, about 1 GB).</div>""",
}

CONCEPTS = [
    {
        "title": "From comparisons to a score",
        "segment": (8, 40),
        "figures": [{"t": 39.0, "caption": "A language model reads the conversation; a linear layer turns its hidden state into "
                                           "the reward r. P(A preferred) = σ(r_A − r_B)."}],
        "body": [
            """<p>For most questions people can't easily write the perfect answer, but they can say which of two answers
is better. A <b>reward model</b> turns those comparisons into a score: a language model reads the whole conversation,
and one linear layer maps its hidden state (here the last token's, at layer 12) to a single number r.</p>""",
            "{fig0}",
            """<p>Training uses pairs where answer A was preferred over answer B. The Bradley–Terry model says
P(A preferred) = σ(r_A − r_B), and the loss is −log σ(r_A − r_B): only the <i>difference</i> of rewards matters, so
rewards have no absolute scale.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A reward model is a classifier of comparisons whose hidden
output, the reward, can score any single answer.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What is the Bradley–Terry loss when r_chosen − r_rejected is 0? 1? −1?",
             "answer": "0.693; 0.313; 1.313.", "why": "−log σ(0) = ln 2; −log σ(1) = 0.313; −log σ(−1) = 1.313.", "key": {'parts': [{'label': 'difference 0', 'value': 0.693, 'tol': 0.0005, 'unit': None}, {'label': 'difference 1', 'value': 0.313, 'tol': 0.0005, 'unit': None}, {'label': 'difference −1', 'value': 1.313, 'tol': 0.0005, 'unit': None}]}},
            {"kind": "tf", "q": "“Adding 10 to every reward changes the reward model's predictions.”",
             "answer": "False.", "why": "Only differences enter σ(r_A − r_B).", "key": {'value': False}},
        ],
    },
    {
        "title": "What it can judge",
        "segment": (40, 72),
        "figures": [{"t": 55.5, "caption": "Capitals: 90 training pairs from 30 countries; 98% right on 20 new countries."},
                    {"t": 71.0, "caption": "Sums: 79% on training pairs, 55% on new sums: barely better than a coin."}],
        "body": [
            """<p>On frozen Qwen2.5-0.5B features, 90 pairs from 30 countries (the right capital preferred over a wrong
one) give a reward model that prefers the right capital for <b>98%</b> of 60 pairs about 20 countries it never saw. It
uses what the language model already knows. With 300 pairs of sums (the right total over one off by up to 10), it reaches
79% on its training pairs but only <b>55%</b> on new sums: it memorised instead of judging, because the frozen model's
features don't encode whether a sum is correct.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A reward model can only judge what its underlying model
understands.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The unbiased reward model gives “The capital of Hungary is Budapest.” 2.81 and “… is "
                                    "Vienna.” −1.07. What probability does it give to Budapest being preferred?",
             "answer": "0.98.", "why": "σ(2.81 − (−1.07)) = σ(3.88) ≈ 0.980.", "key": {'parts': [{'label': None, 'value': 0.98, 'tol': 0.005, 'unit': None}]}},
            {"kind": "short", "q": "The sums reward model scores 79% on training pairs but 55% on new ones. What does that "
                                   "gap mean?",
             "answer": "It memorised features of the specific training pairs instead of learning a rule that generalises.",
             "why": "Training accuracy far above held-out accuracy is the signature of overfitting."},
        ],
    },
    {
        "title": "Shortcuts and reward hacking",
        "segment": (72, 132),
        "figures": [{"t": 91.3, "caption": "Biased training (preferred answers end with “I hope this helps!”): 75% right without "
                                           "the phrase, 0% with the phrase on the wrong answer."},
                    {"t": 108.1, "caption": "Biased rewards for Hungary: Budapest −3.46, Vienna + phrase +4.11. Unbiased: still "
                                            "82% right with the phrase on the wrong answer."},
                    {"t": 123.0, "caption": "Optimizing against a reward model turns its shortcuts into targets: reward "
                                            "hacking."}],
        "body": [
            """<p>Real preference data has quirks. Suppose the preferred answer in training always ends with “I hope this
helps!”. On new countries without the phrase, the reward model is right only <b>75%</b> of the time; put the phrase on
the wrong answer and it prefers the wrong answer <b>every time</b> (0%). Its scores show why: “Budapest.” −3.46,
“Vienna. I hope this helps!” +4.11, “Budapest. I hope this helps!” +4.17. It learned the phrase, not the capital. A
reward model trained with the phrase on both answers or neither still gets 82% right when the phrase is on the wrong
answer, though even it rewards the phrase.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>This matters because the next step (episode 38) optimizes a model against the reward. Every shortcut the
reward model learned becomes a target: <b>reward hacking</b>. Answers grow longer, more flattering and more confident
without getting better.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>The reward model is a proxy for human judgement; any
correlation in the data that isn't quality can be exploited.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With the biased scores, what probability does the model give to “Budapest.” being "
                                    "preferred over “Vienna. I hope this helps!”?",
             "answer": "About 0.0005.", "why": "σ(−3.46 − 4.11) = σ(−7.57).", "key": {'parts': [{'label': None, 'value': 0.0005, 'tol': 5e-05, 'unit': None}]}},
            {"kind": "mc", "q": "Which change to the data would best remove this shortcut?",
             "options": ["More pairs with the same bias", "Pairs where the phrase appears on preferred and rejected answers "
                                                          "alike", "A bigger linear head", "Training for more steps"],
             "answer": "B.", "why": "If the phrase no longer predicts the preference, there is nothing to learn from it.", "key": {'choice': 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Compute the loss for a batch of reward pairs.",
             "code": """import torch, torch.nn.functional as F
r_chosen = torch.tensor([2.81, 0.5, -1.0])
r_rejected = torch.tensor([-1.07, 0.5, 1.0])
print(-F.logsigmoid(r_chosen - r_rejected))""",
             "answer": "About [0.020, 0.693, 2.127].", "why": "−log σ(3.88), −log σ(0), −log σ(−2)."},
        ],
    },
]
