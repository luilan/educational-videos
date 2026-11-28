"""Study guide content for Reinforcement Learning for LLMs, episode 7: Rewards From Preferences.

Build:  python framework/study_guide.py rl-llms r07
Every number comes from code/r07_rewards_from_preferences (make_data.py, reward_model.py, 2026-11-28).
"""

LESSON = {
    "series": "Reinforcement Learning for LLMs",
    "label": "Episode 7",
    "title": "Rewards From Preferences",
    "tagline": "When nothing can grade the answer, learn what people prefer",
    "duration": "2:19",
    "intro": """<p>Episodes 1–6 had a program that could grade every answer. For most of what people ask a chatbot,
no such program exists. RLHF (reinforcement learning from human feedback) replaces it with comparisons: people pick
the better of two answers, and a <b>reward model</b> learns to predict their picks. This lesson builds one from real
answers by a small language model, with a scripted labeller whose rule we know. We can then measure exactly what the
reward model learned, and what it learned instead: a shortcut that the next episode's RL will exploit.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–3 of this series (rewards and baselines);
Foundations on the sigmoid and logistic regression. Code: <code>code/r07_rewards_from_preferences</code> (Python with
PyTorch and transformers; about 35 minutes to sample the answers, 10 to train).</div>""",
}

CONCEPTS = [
    {
        "title": "From grades to preferences",
        "segment": (7.7, 50.1),
        "figures": [{"t": 23.1, "caption": "No program can grade most chatbot answers; RLHF asks people which of two "
                                           "answers is better, and a reward model learns to predict their choices."},
                    {"t": 36.7, "caption": "Qwen2.5-0.5B-Instruct answered 400 word problems 4 times each: full "
                                           "explanations (91% right, 113 words) and one sentence (16% right, 18 words)."},
                    {"t": 49.9, "caption": "The scripted labeller: right beats wrong; if equally right, shorter wins; "
                                           "10% of labels flipped."}],
        "body": [
            """<p>A program can check “47 + 38 = 85”. Nothing can check “explain why the sky is blue to a child”.
RLHF asks people an easier question instead: <b>which of these two answers is better?</b> A <b>reward model</b> is
trained on many such comparisons and then gives every new answer a score that RL can use.</p>""",
            "{fig0}",
            """<p>Our answers come from Qwen2.5-0.5B-Instruct on 400 two-step word problems, 4 answers each at
temperature 0.7: two with the question alone (it explains step by step) and two asked for one sentence. Of the 1,600
answers, the full explanations were 90.8% right (113 words on average) and the one-sentence ones 16.5% right (18
words). Note what this means: in this data, length and correctness go together.</p>""",
            "{fig1}",
            """<p>Our “people” are a script, so we know exactly what they want: (1) a right answer beats a wrong one;
(2) if both are right or both wrong, the shorter wins; (3) 10% of the labels are flipped, because real labellers
disagree with each other and with themselves. Pairs of answers to the same question give 2,342 labelled comparisons.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>When answers cannot be graded, people compare pairs; the
reward model turns those comparisons into a score.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "4 answers to one question give how many pairs?",
             "answer": "6.", "why": "4 × 3 / 2.",
             "key": {"parts": [{"label": None, "value": 6, "tol": 0, "unit": None}]}},
            {"kind": "tf", "q": "“In our data, a long answer was usually a right answer.”",
             "answer": "True.", "why": "90.8% of the full explanations were right, against 16.5% of the one-sentence "
                                       "answers.",
             "key": {"value": True}},
        ],
    },
    {
        "title": "The Bradley–Terry reward model",
        "segment": (50.6, 75.0),
        "figures": [{"t": 61.9, "caption": "Bradley–Terry: P(A preferred over B) = sigmoid(r(A) − r(B)); only "
                                           "differences between scores matter."},
                    {"t": 74.8, "caption": "Our reward model: Qwen2.5-0.5B frozen plus one linear layer (897 numbers), "
                                           "trained on 1,752 pairs from 300 problems, tested on 590 pairs from 100 more."}],
        "body": [
            """<p>The reward model gives every answer a score r. The <b>Bradley–Terry</b> model (1952) links scores to
choices: the probability that A is preferred over B is sigmoid(r(A) − r(B)). Equal scores give 50%; a difference of 2
gives 88%. Training minimizes −log sigmoid(r(preferred) − r(other)) over all labelled pairs. Only <b>differences</b>
matter: adding the same constant to every score changes nothing, the same idea as a baseline in episode 3.</p>""",
            "{fig0}",
            """<p>Real reward models are a language model with its final layer replaced by a single score, fine-tuned as
a whole. Ours is cheaper: Qwen2.5-0.5B-Instruct kept frozen, reading the question and the answer, plus one trained
linear layer on its last hidden state (896 weights and a bias = 897 numbers). It was trained on the 1,752 pairs of
300 problems and tested on the 590 pairs of the other 100.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>A reward model is trained so that score differences predict
which answer people prefer: P = sigmoid(r(A) − r(B)).</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "r(A) = 1.5 and r(B) = 0.5. What is P(A preferred), to 2 decimals?",
             "answer": "0.73.", "why": "sigmoid(1) = 1 / (1 + e^−1) ≈ 0.731.",
             "key": {"parts": [{"label": None, "value": 0.73, "tol": 0.01, "unit": None}]}},
            {"kind": "tf", "q": "“Adding 10 to every score changes the predicted preferences.”",
             "answer": "False.", "why": "Only differences enter the sigmoid.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "What it learned, and what it learned instead",
        "segment": (75.5, 112.5),
        "figures": [{"t": 89.4, "caption": "Held-out pairs: 73% agreement with the labels (≈90% possible); 90% when one "
                                           "answer is right and one wrong; 55% when length should decide."},
                    {"t": 102.6, "caption": "Mean scores: right beats wrong, but among wrong answers long ones score "
                                            "higher: in the data, long answers were 91% right."},
                    {"t": 112.3, "caption": "A blind spot: a wrong one-sentence answer (61 books; right: 63) scores "
                                            "+1.13, above a correct full solution (+0.95)."}],
        "body": [
            """<p>On the 590 held-out pairs the reward model agrees with the labels <b>73.1%</b> of the time. Because 10%
of labels are flipped, even a perfect model would agree only about 90% of the time. Split by what the rule decides
on: when one answer is right and the other wrong it agrees with the rule <b>90.1%</b> of the time; when both are
equally right or wrong, so that length should decide, only <b>55.5%</b>: close to a coin flip.</p>""",
            "{fig0}",
            """<p>Its average scores show what it learned. Right and short +1.33, right and long +0.64, wrong and long
−0.46, wrong and short −0.88. Right beats wrong, and among right answers shorter wins, as the rule says. But among
<b>wrong</b> answers it prefers the long ones, the opposite of the rule. In our data long answers were nearly always
right, so “long” became a cue for “right”: a <b>shortcut</b> learned from how the data happened to be made.</p>""",
            "{fig1}",
            """<p>A concrete blind spot: for “A shelf holds 51 books. Yuki adds 4 boxes of 3 books”, the one-sentence
answer “… So now there are 61 books on the shelf.” (wrong: 63) scores +1.13, higher than a correct 125-word solution
(+0.95).</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>A reward model learns the labellers' preferences only as
far as the data separates them from easier cues; shortcuts like “long looks right” come along.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why can the reward model not reach 100% agreement with the labels?",
             "answer": "10% of the labels were flipped at random, so about 1 pair in 10 contradicts the rule; the best "
                       "possible is about 90%.",
             "why": "Label noise sets a ceiling."},
            {"kind": "mc", "q": "Why did it learn that long wrong answers beat short wrong ones?",
             "options": ["The labellers preferred long answers", "Long answers were almost always right in the data, so "
                         "length became a cue for correctness", "The linear layer cannot see length", "The noise flipped "
                         "those pairs"],
             "answer": "B.", "why": "It picked up a correlation in the data, not the labellers' rule.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "A learned guess",
        "segment": (113.0, 130.5),
        "figures": [{"t": 123.9, "caption": "A reward model is a learned guess of what people prefer, with their noise and "
                                            "the shortcuts in the data; RL will push against it."},
                    {"t": 130.3, "caption": "In code: the Bradley–Terry loss, −log sigmoid(r(preferred) − r(other))."}],
        "body": [
            """<p>A reward model is a <b>learned guess</b> of human preferences: it inherits the labellers' noise and the
shortcuts of the data. For ranking a few answers that is often good enough. But RL does not just rank: it searches
for whatever answers score highest, and will find exactly the places where the guess is wrong. That is the next
episode, where we run RL against this reward model.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Good enough to rank answers is not the same as safe to
optimize against.</div>""",
        ],
        "exercises": [
            {"kind": "code", "q": "<b>Try it yourself.</b> The Bradley–Terry loss.",
             "code": """r_good = reward_model(preferred)
r_bad = reward_model(other)
loss = -F.logsigmoid(r_good - r_bad).mean()     # Bradley-Terry""",
             "answer": "reward_model.py trains the linear head on cached features; try 30% label noise.",
             "why": "One line turns pairwise choices into a training signal for a scorer."},
        ],
    },
]
