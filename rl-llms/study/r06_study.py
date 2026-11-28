"""Study guide content for Reinforcement Learning for LLMs, episode 6: GRPO: Grading on a Curve.

Build:  python framework/study_guide.py rl-llms r06
Every number comes from code/r06_grpo (grpo.py and grpo.py extra, 2026-11-28).
"""

LESSON = {
    "series": "Reinforcement Learning for LLMs",
    "label": "Episode 6",
    "title": "GRPO: Grading on a Curve",
    "tagline": "No value model: judge each answer against its siblings",
    "duration": "2:12",
    "intro": """<p><b>GRPO</b> (Group Relative Policy Optimization) is the RL method DeepSeek introduced in 2024 and
used to train its reasoning models. It keeps PPO's ratio and clip but replaces the value model with a simple rule:
compare each answer with the other answers to the same question. This lesson builds it, then tests it honestly on our
addition model. GRPO learned less than PPO here. Two reasons explain the gap, groups that teach nothing and the
strength of the KL leash, and the fixes for both have known costs. GRPO's real advantage is cost, not learning per
answer.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 3–5 of this series (baselines, the KL
leash, PPO's ratio, clip and value head). Code: <code>code/r06_grpo</code> (Python with PyTorch; about 11 + 8
minutes).</div>""",
}

CONCEPTS = [
    {
        "title": "Grading on a curve",
        "segment": (7.7, 37.0),
        "figures": [{"t": 20.9, "caption": "PPO needs a value model, for a language model usually as big as the "
                                           "model itself. GRPO (DeepSeek, 2024) drops it."},
                    {"t": 36.8, "caption": "A group of 8 answers with 3 right: advantage = (reward − group average) ÷ "
                                           "group spread, so +1.21 for a right answer and −0.72 for a wrong one."}],
        "body": [
            """<p>PPO's baseline is a <b>value model</b> that predicts the expected reward. For a language model it is
usually a second network the size of the policy, with its own memory and training. GRPO (DeepSeekMath, arXiv
2402.03300, February 2024) removes it.</p>""",
            "{fig0}",
            """<p>Instead, GRPO asks each question several times; the answers to one question form a <b>group</b> of
size G. Each answer's advantage compares it with its own group:</p>
<p style="text-align:center"><code>advantage = (reward − group mean) / (group standard deviation + 0.0001)</code></p>
<p>This is episode 3's per-question baseline, divided by the spread. With 3 right answers out of 8 the mean is 0.375
and the standard deviation 0.518, so right answers get +1.21 and wrong ones −0.72.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>GRPO's baseline is the average of the other answers to the
same question, so no value model is needed.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A group of 4 answers has rewards 1, 0, 0, 0. What is the advantage of the right "
                                    "answer before dividing by the spread?",
             "answer": "0.75.", "why": "The mean is 0.25; 1 − 0.25 = 0.75.",
             "key": {"parts": [{"label": None, "value": 0.75, "tol": 0, "unit": None}]}},
            {"kind": "tf", "q": "“GRPO needs a trained model to compute advantages.”",
             "answer": "False.", "why": "The advantage uses only the rewards of the group: an average and a spread.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "What stays, and what it saves",
        "segment": (37.5, 47.3),
        "figures": [{"t": 47.1, "caption": "PPO in RLHF keeps four networks (policy, reference, reward model, value "
                                           "model); GRPO with a checker program keeps two."}],
        "body": [
            """<p>The rest of GRPO is PPO: the per-token ratio and the clip, with several updates per batch. GRPO also
puts a <b>KL leash</b> to the frozen starting model directly in the loss (DeepSeekMath used β = 0.04), rather than in
the reward as in episode 4.</p>
<p>The saving is memory and computing. Classic RLHF with PPO keeps four networks: the policy, the frozen reference,
the reward model and the value model. GRPO drops the value model; with a program as the grader, as in maths or
code with checkable answers, only the policy and the reference remain.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>GRPO = PPO's clipped update + group-relative advantages + a
KL term in the loss, without a value model.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In RLHF with PPO, how many networks are in memory? And in GRPO with a checker "
                                    "program?",
             "answer": "4 and 2.", "why": "Policy, reference, reward model, value model; then policy and reference.",
             "key": {"parts": [{"label": "PPO", "value": 4, "tol": 0, "unit": None},
                               {"label": "GRPO", "value": 2, "tol": 0, "unit": None}]}},
        ],
    },
    {
        "title": "What we measured, and why",
        "segment": (47.8, 88.5),
        "figures": [{"t": 59.3, "caption": "Same budget as episode 5: PPO 28%; GRPO with groups of 2, 4, 8 or 16 "
                                           "answers 16–18%."},
                    {"t": 76.4, "caption": "Groups whose answers all get the same reward teach nothing: 92%, 84%, 75% "
                                           "and 70% of groups for G = 2, 4, 8, 16."},
                    {"t": 88.3, "caption": "Without the KL term GRPO reaches 26%, close to PPO's 28%: most of the gap "
                                           "was the leash."}],
        "body": [
            """<p>Same task, starting model and budget as episode 5 (400 batches of 256 answers, 16 updates per batch,
step 0.0001, 3 runs). Final accuracy: PPO <b>28.2%</b> (27.6, 27.8, 29.1). GRPO with groups of 2, 4, 8 and 16
answers: <b>15.8, 18.0, 17.0 and 17.2%</b>. On our task GRPO was clearly worse.</p>""",
            "{fig0}",
            """<p><b>Reason 1: empty groups.</b> If every answer in a group gets the same reward, every advantage is
zero and the group contributes nothing. Our model is very consistent: for most questions it is almost always right
or almost always wrong. With 8 answers per question, 75% of groups were all right or all wrong; with 2 answers, 92%;
even with 16, 70%. PPO's value head still gives a wrong answer a negative advantage (reward 0 minus a predicted
0.2), so it learns from questions where every try failed.</p>""",
            "{fig1}",
            """<p><b>Reason 2: the leash.</b> We used DeepSeekMath's KL strength, β = 0.04. Without the KL term, GRPO
with groups of 8 reached <b>25.8%</b> (26.8, 23.2, 27.3), close to PPO. In this short run on a tiny model, most of
the gap was the leash. (Dividing by the spread mattered too: without it, 13.5%.)</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Groups where all answers agree give GRPO no signal, and a
KL leash can hold it back; both explain why it trailed PPO here.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "A question is so hard that all 8 answers in its group are wrong. What does GRPO "
                                   "learn from it, and what does PPO learn?",
             "answer": "GRPO: nothing (all advantages are 0). PPO: each wrong answer gets 0 minus the predicted reward, "
                       "a negative advantage, so it is pushed down.",
             "why": "GRPO's baseline is the group itself; PPO's is a separate prediction."},
            {"kind": "number", "q": "With groups of 8, 75% of groups were empty. Out of 32 groups in a batch, about "
                                    "how many taught anything?",
             "answer": "8.", "why": "25% of 32.",
             "key": {"parts": [{"label": None, "value": 8, "tol": 0, "unit": None}]}},
        ],
    },
    {
        "title": "Dynamic sampling, and what GRPO is for",
        "segment": (89.0, 123.8),
        "figures": [{"t": 100.8, "caption": "Dynamic sampling (DAPO): drop groups that all agree and sample more until "
                                            "the batch is full: 27%, but 4.4× as many answers generated."},
                    {"t": 117.5, "caption": "GRPO's win is cost, not learning per answer; its weak spots are empty "
                                            "groups and the strength of the leash."},
                    {"t": 123.6, "caption": "In code: reshape the rewards into groups, subtract each group's mean, "
                                            "divide by its spread."}],
        "body": [
            """<p>A later recipe, <b>DAPO</b> (2025), fixes empty groups with <b>dynamic sampling</b>: throw away groups
whose rewards all agree and sample more questions until the batch is full of useful groups. With groups of 8 it
reached <b>27.4%</b> (28.5, 32.0, 21.7), but it generated 433,664–465,152 answers instead of 102,400, <b>4.4
times</b> as many. Since generating is the expensive part for large models, this is a real cost.</p>""",
            "{fig0}",
            """<p>So GRPO's advantage is <b>cost</b>: with no value network, RL on very large models becomes much cheaper,
which is why it spread so fast after DeepSeek's reasoning models. On learning per answer it was not better than PPO
here. Its weak spots, empty groups and the strength of the leash, are exactly what later recipes adjust; DAPO, for
example, drops the KL term altogether.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>GRPO trades a little learning per answer for a much
cheaper setup; fixes like dynamic sampling buy the learning back with more generation.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Dynamic sampling generated about 450,000 answers instead of 102,400. About how many "
                                    "times as many is that (1 decimal)?",
             "answer": "About 4.4.", "why": "450,000 / 102,400 ≈ 4.4.",
             "key": {"parts": [{"label": None, "value": 4.4, "tol": 0.1, "unit": None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Group-relative advantages.",
             "code": """g = r.view(n_questions, G)                  # one row per group
adv = g - g.mean(1, keepdim=True)           # minus the group average
adv = adv / (g.std(1, keepdim=True) + 1e-4) # divided by its spread
loss = clipped_ppo_loss(ratio, adv.view(-1)) + beta * kl""",
             "answer": "grpo.py runs PPO and GRPO with several group sizes; grpo.py extra adds no-KL and dynamic "
                       "sampling. Try β = 0.01.",
             "why": "Three lines replace PPO's value model."},
        ],
    },
]
