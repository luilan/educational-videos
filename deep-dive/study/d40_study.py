"""Study guide content for How LLMs Work: Deep Dive, episode 40: Reinforcement Learning for Reasoning.

Build:  python framework/study_guide.py deep-dive d40 --video deep-dive/media/videos/d40_scene/1080p60/RLReasoningVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d40_rl_reasoning/grpo.py (Qwen2.5-0.5B-Instruct, GRPO: 30 steps × 4 questions × 8
samples, lr 2e-6, KL beta 0.04, seeded; 100 held-out questions, greedy) or from the arithmetic shown.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 40",
    "title": "RL for Reasoning",
    "tagline": "Training on answers a program can check",
    "duration": "2:27",
    "intro": """<p>This lesson answers one question: how can a model improve with no human judgements at all? When a
program can check the answer (math, code, puzzles), the check itself is the reward. <b>GRPO</b> samples a group of
answers per question and pushes up the ones that beat their group's average. On Qwen2.5-0.5B, 30 steps take the reward
from 0 to 94 out of 100, but the arithmetic was already there: what RL taught was the answer format and when to stop.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 37 (reward models), 38 (RLHF with PPO)
and 39 (DPO). Code: <code>code/d40_rl_reasoning</code> (PyTorch + transformers; about 55 minutes and 10 GB of memory on
a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Verifiable rewards",
        "segment": (8, 39),
        "figures": [{"t": 20.0, "caption": "Preferences need people; checkable answers need only a program."},
                    {"t": 38.0, "caption": "The task and the reward: 1 for the right “Answer: N”, else 0."}],
        "body": [
            """<p>RLHF and DPO learn from human preferences. For tasks whose answer a program can check (an arithmetic
result, code that passes tests, a solved puzzle), there is a cheaper signal: the check itself. Training with such a
<b>verifiable reward</b> needs no reward model and no labels, and the model cannot fool a learned judge, because there
is none. This is how reasoning models such as DeepSeek-R1 were trained.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Our version: Qwen2.5-0.5B-Instruct answers questions like “What is 37 × 2 + 48?”, asked to think step by
step and end with “Answer: &lt;number&gt;”. The reward is 1 if that final number is right, 0 otherwise: strict about
both correctness and format.</p>""",
            """<div class="box key"><b class="t">Key idea</b>When answers can be checked automatically, the checker is the
reward: no reward model, no human labels.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which task is the best fit for a verifiable reward?",
             "options": ["Writing a function that must pass unit tests", "Writing a friendly email",
                         "Summarising a news article"],
             "answer": "Writing a function that must pass unit tests.",
             "why": "Running the tests checks the answer automatically; the other two need a judge.", "key": {'choice': 0}},
            {"kind": "short", "q": "Our reward also requires the exact text “Answer: N”. What does that strictness "
                                   "reward, besides correct arithmetic?",
             "answer": "Following the requested output format.",
             "why": "Before training the model was usually right but used a different format, so it scored 0."},
        ],
    },
    {
        "title": "GRPO: the group is the baseline",
        "segment": (39, 73),
        "figures": [{"t": 58.5, "caption": "Eight answers to one question: advantage = (reward − group mean) / group "
                                           "spread."},
                    {"t": 72.5, "caption": "All right or all wrong: every advantage is 0, no signal."},
                    {"t": 138.0, "caption": "The core of GRPO in code."}],
        "body": [
            """<p>For each question, sample a group of G = 8 answers and score them. Each answer's <b>advantage</b> is
its reward minus the group's mean, divided by the group's standard deviation. The loss raises the log-probability of
every token in answers with positive advantage and lowers it for negative ones, plus a KL penalty (β = 0.04) that keeps
the model near where it started.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>PPO (episode 38) needs a value head to estimate how good a situation is; GRPO uses the other answers to
the same question instead. The price: if all eight answers get the same reward, every advantage is 0 and that question
teaches nothing. Learning needs questions the model sometimes gets right and sometimes wrong.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Each answer is judged against its siblings: better than the
group average → more likely; worse → less likely.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A group's rewards are 1, 0, 1, 1, 0, 0, 1, 0. What is the advantage of each right "
                                    "answer? (Use the sample standard deviation.)",
             "answer": "About +0.94.", "why": "Mean 0.5; standard deviation 0.53; (1 − 0.5) / 0.53 ≈ 0.94.", "key": {'parts': [{'label': None, 'value': 0.94, 'tol': 0.0188, 'unit': None}]}},
            {"kind": "number", "q": "Seven answers right, one wrong. What are the advantages?",
             "answer": "About +0.35 for each right answer, −2.47 for the wrong one.",
             "why": "Mean 0.875, standard deviation 0.354: a rare failure gets a large push down.", "key": {'parts': [{'label': 'each right answer', 'value': 0.35, 'tol': 0.007, 'unit': None}, {'label': 'wrong answer', 'value': -2.47, 'tol': 0.0494, 'unit': None}]}},
            {"kind": "tf", "q": "“A question the model always gets right still improves it under GRPO.”",
             "answer": "False.", "why": "All rewards equal → all advantages 0; only the KL term acts.", "key": {'value': False}},
        ],
    },
    {
        "title": "What the training actually changed",
        "segment": (73, 127),
        "figures": [{"t": 88.0, "caption": "Before: reward 0/100, but the last number is right 95/100."},
                    {"t": 102.5, "caption": "Training reward per step: 0.25 → 0.94 by step 5, then 0.66–1.00."},
                    {"t": 126.0, "caption": "After: 94/100; every answer finishes, 29% shorter; arithmetic unchanged."}],
        "body": [
            """<p>Before training, on 100 held-out questions (greedy): reward 0. Yet the last number in the text is
right 95 times: the model computes correctly, then ends with <code>\\boxed{…}</code> instead of “Answer: N”, and 16
answers run past the 160-token limit.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<table><tr><th></th><th>before</th><th>after GRPO</th></tr>
<tr><td>reward (“Answer: N” right)</td><td>0</td><td>94</td></tr>
<tr><td>finished within 160 tokens</td><td>84</td><td>100</td></tr>
<tr><td>mean length (tokens)</td><td>148</td><td>105</td></tr>
<tr><td>last number right</td><td>95</td><td>94</td></tr></table>""",
            "{fig2}",
            """<p>So RL did not teach arithmetic here: it taught the model to use what it already knew in the form the
reward checks, and to stop. That is typical: RL mostly <i>amplifies</i> behaviour the model already produces sometimes
(sampled answers already scored 0.25 at step 1). In large models, the same pressure has grown longer chains of thought
that check their own work.</p>""",
            """<div class="box key"><b class="t">Key idea</b>RL with a verifiable reward strengthens what the model can
already do some of the time; check which behaviour actually changed.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Mean length fell from 148 to 105 tokens. By what percentage?",
             "answer": "About 29%.", "why": "1 − 105 / 148 ≈ 0.29.", "key": {'parts': [{'label': None, 'value': 29, 'tol': 0.58, 'unit': '%'}]}},
            {"kind": "short", "q": "Why could GRPO make progress from the very first step, even though greedy decoding "
                                   "scored 0?",
             "answer": "Sampled answers sometimes used the right format (training reward 0.25 at step 1), so groups had "
                       "mixed results and non-zero advantages.",
             "why": "GRPO can only amplify behaviour that appears in the samples."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Compute the GRPO advantages for 4 groups of 8 rewards.",
             "code": """import torch
r = torch.tensor([[1, 0, 1, 1, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float)
adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
print(adv.round(decimals=2))""",
             "answer": "Rows: ±0.94; +0.35 and −2.47; all 0; all 0.",
             "why": "The last two groups give no learning signal."},
        ],
    },
]
