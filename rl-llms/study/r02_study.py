"""Study guide content for Reinforcement Learning for LLMs, episode 2: REINFORCE.

Build:  python framework/study_guide.py rl-llms r02
Every number comes from code/r02_reinforce (reinforce.py, 2026-11-27): REINFORCE without a baseline from episode 1's
model, 2,000 steps of 32 questions × 8 tries, learning rate 0.0003.
"""

LESSON = {
    "series": "Reinforcement Learning for LLMs",
    "label": "Episode 2",
    "title": "REINFORCE",
    "tagline": "Make rewarded answers more likely: reward times log-probability",
    "duration": "2:27",
    "intro": """<p><b>REINFORCE</b> (Williams, 1992) is the simplest way to turn grades into learning: for every
sampled answer, push up its log-probability in proportion to its reward. This lesson explains the rule and why it uses
the log-probability, then trains episode 1's tiny model with it. With no answer key, accuracy on new questions climbs
from 14% to 73% in about a minute. The model even learns answers it never sampled before. The lesson ends with the
rule's weakness, which the next episode fixes.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 1 of this series; Foundations episodes 5
(exponentials and logs), 6 (probability and sampling) and 10 (gradients). Code: <code>code/r02_reinforce</code>
(Python with PyTorch; about a minute on a CPU after episode 1's <code>grade.py</code> has written
<code>base.pt</code>).</div>""",
}

CONCEPTS = [
    {
        "title": "The REINFORCE rule",
        "segment": (7.5, 37.4),
        "figures": [{"t": 21.3, "caption": "Episode 1's model gets 14% of new questions right and its tries are graded "
                                           "1 or 0. REINFORCE (Williams, 1992) turns those grades into learning."},
                    {"t": 37.1, "caption": "For every answer, push up reward × log-probability: a rewarded answer "
                                           "becomes more likely, a 0-reward answer gets no push."}],
        "body": [
            """<p>REINFORCE needs only three things: a way to <b>sample</b> answers, a <b>reward</b> for each, and the
model's <b>log-probability</b> of each answer it sampled. The rule: for every sampled answer, nudge the weights to
increase</p>
<p style="text-align:center"><code>reward × log p(answer)</code></p>
<p>For an answer with reward 1 this is exactly a supervised training step on the model's own answer: it becomes more
likely. For an answer with reward 0 the product is 0, so that answer gets <b>no push</b> at all. The log-probability of
a whole answer is the sum of the log-probabilities of its tokens, so every token of a rewarded answer is reinforced.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>REINFORCE is supervised training on the model's own
answers, weighted by their rewards.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "An answer has reward 0 and log-probability −2.3. What is its term reward × log p?",
             "answer": "0.", "why": "0 × anything is 0: the answer gets no direct push.",
             "key": {"parts": [{"label": None, "value": 0, "tol": 0, "unit": None}]}},
            {"kind": "tf", "q": "“For an answer with reward 1, a REINFORCE step is the same as a supervised step that "
                                "trains the model on that answer.”",
             "answer": "True.", "why": "Maximizing 1 × log p(answer) is exactly the supervised objective for that "
                                       "text.",
             "key": {"value": True}},
            {"kind": "mc", "q": "Which of these does REINFORCE need?",
             "options": ["The right answer to every question", "A differentiable checker",
                         "The log-probability of each sampled answer", "A second model"],
             "answer": "C.", "why": "It needs samples, their rewards and their log-probabilities; the checker can be "
                                    "any program.",
             "key": {"choice": 2}},
        ],
    },
    {
        "title": "Why reward times log-probability",
        "segment": (37.9, 55.8),
        "figures": [{"t": 55.5, "caption": "The goal is a higher average reward. Its gradient equals the average of "
                                           "reward × ∇ log p(answer), which samples can estimate."}],
        "body": [
            """<p>The goal is a higher <b>average reward</b>: the sum over all possible answers of p(answer) ×
reward(answer). We cannot backpropagate through the checker (it is a program, not a formula) or through the random
choice of tokens. The <b>log-derivative trick</b> avoids both. Since ∇p = p · ∇ log p,</p>
<p style="text-align:center"><code>∇ average reward = Σ p(answer) · reward · ∇ log p(answer)</code></p>
<p>That is itself an average over answers drawn from p, so it can be <b>estimated from samples</b>: draw answers,
and average reward × ∇ log p(sample). The reward is just a number multiplying the gradient; it never needs a gradient
of its own. Training with the loss <code>−mean(reward × log p)</code> follows exactly this estimate.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>The gradient of the expected reward is the expected value of
reward × ∇ log p, so sampled answers give an unbiased estimate without differentiating the reward.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why can't we just backpropagate the reward to the weights directly?",
             "answer": "The reward comes from a checker program applied to sampled text; neither the program nor the "
                       "random sampling of tokens has a gradient.",
             "why": "REINFORCE only needs gradients of the model's own log-probabilities."},
            {"kind": "number", "q": "If p(answer) = 0.25, what is ∇p in terms of ∇ log p? Give the factor.",
             "answer": "0.25.", "why": "∇p = p · ∇ log p = 0.25 · ∇ log p.",
             "key": {"parts": [{"label": None, "value": 0.25, "tol": 0, "unit": None}]}},
            {"kind": "tf", "q": "“The REINFORCE estimate is exact for a single sample.”",
             "answer": "False.", "why": "One sample gives an unbiased but noisy estimate; averaging many samples "
                                        "reduces the noise.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "One step, and two thousand",
        "segment": (56.3, 88.5),
        "figures": [{"t": 72.5, "caption": "One step: 32 questions × 8 tries = 256 answers; 53 are right and only those "
                                           "are pushed up; other answers to the same questions lose a little."},
                    {"t": 88.2, "caption": "2,000 steps, about a minute on a CPU: 14% → 73% on 1,000 questions the "
                                           "model never trains on. No answer was shown, only grades."}],
        "body": [
            """<p>Each training step takes 32 random training questions and samples 8 answers to each: 256 answers. In
the first step 53 were right. Only those 53 get a push. Yet the others are not untouched: the model's probabilities
for the next token must add up to 1, so making a right answer more likely makes the other answers to the same
question <b>less likely</b>, a little.</p>""",
            "{fig0}",
            """<p>After 2,000 such steps (about a minute on a CPU, learning rate 0.0003), the model gets <b>73%</b> of
the 1,000 test questions right on its first choice, up from 14%. The test questions were never used in training, and
no right answer was shown during this training: only questions and grades. The curve rises slowly at first (few
right answers to reinforce), fastest between steps 500 and 1,250, and then flattens.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Reinforcing only the right answers is enough to move a
model from 14% to 73%, because raising them lowers the alternatives.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In the first step, what fraction of the 256 answers was right? (Round to a whole "
                                    "percent.)",
             "answer": "21%.", "why": "53 / 256 = 0.207.",
             "key": {"parts": [{"label": None, "value": 21, "tol": 1, "unit": "%"}]}},
            {"kind": "number", "q": "How many answers does the whole run sample (2,000 steps)?",
             "answer": "512,000.", "why": "2,000 × 256.",
             "key": {"parts": [{"label": None, "value": 512000, "tol": 0, "unit": None}]}},
            {"kind": "short", "q": "Why does the curve rise slowly in the first few hundred steps?",
             "answer": "Only about one answer in five is right at first, so few answers get pushed up; as accuracy "
                       "rises there are more right answers to reinforce.",
             "why": "REINFORCE learns from its own successes; at the start they are rare."},
        ],
    },
    {
        "title": "What it learned, and where it stalls",
        "segment": (89.0, 139.8),
        "figures": [{"t": 98.1, "caption": "27+15: before, 48 forty times and 42 twenty-three times; after, 42 in all "
                                           "64 tries."},
                    {"t": 118.8, "caption": "47+38: 85 never appeared in 64 tries before training; after, 58 times. "
                                            "Of the 644 questions never right before, 445 are now right."},
                    {"t": 131.5, "caption": "The curve flattens near 73%: reward-0 answers teach nothing directly, and "
                                            "once most answers are right nearly everything is pushed up."},
                    {"t": 139.5, "caption": "REINFORCE in code: the loss is minus reward times log-probability, "
                                            "averaged over the answers."}],
        "body": [
            """<p>For <code>27+15=</code>, 64 tries before training gave 48 forty times and the right 42 twenty-three
times; after training, 42 all 64 times. More surprising is <code>47+38=</code>: before training the right answer 85
did <b>not appear once</b> in 64 tries (75 thirty-five times, 74 twenty times), so this question never produced a
reward. After training, 85 comes out 58 times. The model learned to carry the one on other questions, and that skill
<b>carried over</b>. Of the 644 test questions that were never right in 64 tries before training, 445 are now right on
the first choice.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The weakness shows at the end of the curve, which flattens near 73%. With rewards of 0 or 1, wrong
answers teach nothing directly, and once most answers are right almost every answer gets pushed up, the good and the
lucky alike, so the gradient becomes noisy. The fix, in the next episode, is to compare each reward with what was
expected: a <b>baseline</b>.</p>""",
            '<div class="figrow">{fig2}{fig3}</div>',
            """<div class="box key"><b class="t">Key idea</b>What is reinforced on some questions generalizes to others,
even ones that never earned a reward; plain REINFORCE slows down once most answers are right.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What fraction of the 644 never-right questions became right? (Round to a whole "
                                    "percent.)",
             "answer": "69%.", "why": "445 / 644 = 0.691.",
             "key": {"parts": [{"label": None, "value": 69, "tol": 1, "unit": "%"}]}},
            {"kind": "tf", "q": "“A question that never produced a right sample cannot improve under REINFORCE.”",
             "answer": "False.", "why": "Its own reward was always 0, but learning on other questions (carrying) "
                                        "transfers: 47+38 went from 0 to 58 right in 64 tries.",
             "key": {"value": False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> One REINFORCE step.",
             "code": """idx, answers, lengths, qs = sample(model, questions, n=8)
r = torch.tensor([reward(a, b, x) for (a, b), x in zip(qs, answers)])
logp = answer_logprob(model, idx, lengths)
loss = -(r * logp).mean()
opt.zero_grad(); loss.backward(); opt.step()""",
             "answer": "reinforce.py runs 2,000 of these and logs the accuracy every 50 steps.",
             "why": "Minimizing −reward × log p raises the probability of rewarded answers."},
        ],
    },
]
