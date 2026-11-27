"""Study guide content for Reinforcement Learning for LLMs, episode 3: Baselines and Advantages.

Build:  python framework/study_guide.py rl-llms r03
Every number comes from code/r03_baselines (baselines.py and baselines.py offset, 2026-11-27).
"""

LESSON = {
    "series": "Reinforcement Learning for LLMs",
    "label": "Episode 3",
    "title": "Baselines and Advantages",
    "tagline": "Don't ask whether it was right. Ask whether it was better than usual",
    "duration": "2:25",
    "intro": """<p>A <b>baseline</b> is a reward you expected anyway; subtracting it leaves the <b>advantage</b>,
how much better or worse an answer did than usual. This lesson defines two simple baselines and shows why they leave
the average update unchanged. Then it tests them honestly. With rewards of 0 and 1, they made no real difference in our
runs. Once wrong answers also earn points, REINFORCE without a baseline wrecks the model, and with a baseline nothing
changes at all. That is what baselines are really for.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1 and 2 of this series; Foundations
episode 8 (averages and spread). Code: <code>code/r03_baselines</code> (Python with PyTorch; about 12 minutes,
then about 6 for <code>python baselines.py offset</code>, which needs episode 2's <code>reinforced.pt</code>).</div>""",
}

CONCEPTS = [
    {
        "title": "Advantage = reward − baseline",
        "segment": (7.5, 45.2),
        "figures": [{"t": 22.4, "caption": "A baseline is the reward you expected; what is left after subtracting it is "
                                           "the advantage. REINFORCE then pushes up advantage × log p."},
                    {"t": 44.9, "caption": "Batch average (21% right): right +0.79, wrong −0.21. Per question (3 of 8 "
                                           "right): right +0.625, wrong −0.375. Wrong answers are now pushed down."}],
        "body": [
            """<p>Plain REINFORCE pushes up every sampled answer in proportion to its reward. With a <b>baseline</b>
<i>b</i>, it pushes up in proportion to the <b>advantage</b>, reward − <i>b</i>: positive if the answer did better than
expected, negative if worse.</p>""",
            "{fig0}",
            """<p>Two simple choices of <i>b</i>:</p>
<ul><li><b>Batch average.</b> If 21% of the 256 answers in a batch are right, the mean reward is 0.21, so a right
answer gets +0.79 and a wrong one −0.21.</li>
<li><b>Per question.</b> Use the mean reward of the 8 tries at the same question. With 3 of 8 right, right answers get
+0.625 and wrong ones −0.375. If all 8 are right, or all wrong, every advantage is 0 and the question teaches
nothing.</li></ul>
<p>Either way, wrong answers now get a <b>negative</b> weight: they are pushed down directly, not only because the
probabilities must add up to 1.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>The advantage measures better or worse than expected, so
good answers go up and bad answers go down.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A batch has 40% right answers (rewards 0 or 1). What is the advantage of a wrong "
                                    "answer with the batch-average baseline?",
             "answer": "−0.4.", "why": "0 − 0.4.",
             "key": {"parts": [{"label": None, "value": -0.4, "tol": 0, "unit": None}]}},
            {"kind": "number", "q": "With the per-question baseline, 5 of 8 tries are right. What is the advantage of a "
                                    "right answer?",
             "answer": "+0.375.", "why": "1 − 5/8 = 0.375.",
             "key": {"parts": [{"label": None, "value": 0.375, "tol": 0, "unit": None}]}},
            {"kind": "tf", "q": "“With the per-question baseline, a question where all 8 tries are wrong still pushes "
                                "the wrong answers down.”",
             "answer": "False.", "why": "The mean is 0, so every advantage is 0 − 0 = 0: no push at all.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "Same direction, less noise (in theory)",
        "segment": (45.7, 57.7),
        "figures": [{"t": 57.4, "caption": "Subtracting a constant b changes nothing on average, because the mean of ∇ "
                                           "log p over the model's own samples is 0; it only changes the noise."}],
        "body": [
            """<p>Why is it allowed to subtract <i>b</i>? The update is the average of (reward − <i>b</i>) · ∇ log p,
which is the average of reward · ∇ log p minus <i>b</i> times the average of ∇ log p. Over the model's own samples
the average of ∇ log p is <b>zero</b>: the probabilities always add up to 1, so if some go up, others must go down, and
their gradients cancel on average. So a baseline that does not depend on which answer was sampled leaves the
<b>expected</b> update unchanged.</p>
<p>What it can change is the <b>noise</b> around that average, the difference between one batch's estimate and
another's. Less noise usually means faster, steadier learning. That is the textbook argument for baselines.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>A baseline that does not depend on the sampled answer keeps
the gradient unbiased; it can only reduce (or increase) its noise.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why must the baseline not depend on which answer was sampled?",
             "answer": "Otherwise b · ∇ log p no longer averages to zero, and the expected update changes (it becomes "
                       "biased).",
             "why": "The cancellation uses the fact that b is the same whatever was sampled."},
            {"kind": "mc", "q": "What does a good baseline reduce?",
             "options": ["The expected gradient", "The variance (noise) of the gradient estimate",
                         "The number of samples", "The reward"],
             "answer": "B.", "why": "The expected gradient stays the same; only the spread between batches can shrink.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "What we measured",
        "segment": (58.2, 107.7),
        "figures": [{"t": 77.6, "caption": "Rewards 0 and 1, 3 runs each: no baseline 76%, batch 74%, per question "
                                           "74%. Signal-to-noise at the start: 0.25, 0.23, 0.25."},
                    {"t": 98.1, "caption": "A point for trying (wrong 1, right 2): without a baseline, accuracy falls "
                                           "from 14% to 2–6% in all three runs."},
                    {"t": 107.4, "caption": "With a baseline the shifted run (dashed) is identical, step for step, to "
                                            "the run with rewards 0 and 1."}],
        "body": [
            """<p><b>Experiment 1</b> keeps rewards of 0 and 1 and runs each method 3 times for 2,000 steps from
episode 1's model. Final accuracy on the 1,000 test questions: no baseline 75.5 / 78.8 / 73.2% (mean 76%), batch average
76.8 / 71.5 / 73.6% (74%), per question 72.4 / 76.9 / 73.1% (74%). The spread between seeds is bigger than the
difference between methods. The gradient's signal-to-noise ratio at the start is also about the same (0.25, 0.23, 0.25),
and late in training (episode 2's 73% model) too (0.65, 0.62, 0.67). Here, with rare successes and the Adam optimizer
(which rescales every step), the baseline <b>barely matters</b>. That is a real result, not a failure: textbook
improvements do not always show up on a given problem.</p>""",
            "{fig0}",
            """<p><b>Experiment 2</b> changes only the rewards: 1 point for a wrong answer, 2 for a right one. The right
answer is still better. Without a baseline, every sampled answer now gets a positive weight, so every answer is pushed
up, the wrong ones nearly as hard as the right ones. The model reinforces whatever it already does most, which is
usually wrong: accuracy falls from 14% to <b>2.4%, 2.4% and 5.5%</b> in the three runs, and it halves within 200 steps.</p>""",
            "{fig1}",
            """<p>With either baseline, the run with rewards 1 and 2 is <b>identical, step for step</b>, to the run with
0 and 1 (same seed): subtracting the average removes the extra point exactly, so the advantages are the same numbers.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>With 0/1 rewards baselines made little difference here;
with an offset in the rewards, no baseline is a disaster and a baseline makes the offset irrelevant.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In experiment 2 without a baseline, what is the weight of a wrong answer relative "
                                    "to a right one?",
             "answer": "0.5 (1 versus 2).", "why": "Wrong answers get reward 1, right ones 2: half the push, but still "
                                                   "a push up.",
             "key": {"parts": [{"label": None, "value": 0.5, "tol": 0, "unit": None}]}},
            {"kind": "number", "q": "In experiment 2 with the batch baseline, if 20% of answers are right, what is the "
                                    "advantage of a right answer?",
             "answer": "+0.8.", "why": "Mean reward 1.2; 2 − 1.2 = 0.8, the same as 1 − 0.2 with rewards 0 and 1.",
             "key": {"parts": [{"label": None, "value": 0.8, "tol": 0, "unit": None}]}},
            {"kind": "tf", "q": "“Experiment 1 shows that baselines never help.”",
             "answer": "False.", "why": "It shows they made little difference for 0/1 rewards on this task with Adam; "
                                        "experiment 2 shows a case where they are essential.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "The real job of a baseline",
        "segment": (108.2, 138.6),
        "figures": [{"t": 127.0, "caption": "Learning should depend on better or worse than usual, not on where zero is. "
                                            "PPO learns its baseline (a value head); GRPO uses the group average."},
                    {"t": 138.3, "caption": "In code: subtract the batch mean, or each question's mean over its tries, "
                                            "before multiplying by the log-probability."}],
        "body": [
            """<p>Experiment 2 shows the real job of a baseline: it makes learning depend on <b>better or worse than
usual</b>, not on where zero happens to be. Our checker's 0 and 1 had a natural zero. Reward models (episode 7) output
scores on an arbitrary scale, often all positive or all negative, and a verifier might give partial credit. Without a
baseline the zero point of that scale would decide whether answers are pushed up or down.</p>
<p>So the standard RL methods for language models all use one. <b>PPO</b> (episode 5) <i>learns</i> its baseline: a
value head that predicts the expected reward. <b>GRPO</b> (episode 6) uses exactly our per-question baseline, the
average reward of a group of answers to the same question, which removes the need for a value model.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A baseline makes the update invariant to shifting all
rewards by a constant; that is why every practical method has one.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "A reward model scores answers between 5.0 and 7.0. Without a baseline, what happens?",
             "options": ["Only the best answers are pushed up", "Every sampled answer is pushed up",
                         "Every answer is pushed down", "Nothing; the scale doesn't matter"],
             "answer": "B.", "why": "All rewards are positive, so every answer gets a positive weight, as in "
                                    "experiment 2.",
             "key": {"choice": 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> The two baselines.",
             "code": """adv = r - r.mean()                                              # batch average
adv = (r.view(Q, G) - r.view(Q, G).mean(1, keepdim=True)).view(-1)   # per question
loss = -(adv * logp).mean()""",
             "answer": "baselines.py trains with each; add 1 to r and compare (baselines.py offset).",
             "why": "One subtraction makes the update ignore a constant shift in the rewards."},
        ],
    },
]
