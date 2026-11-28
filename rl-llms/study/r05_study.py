"""Study guide content for Reinforcement Learning for LLMs, episode 5: PPO: Small, Safe Steps.

Build:  python framework/study_guide.py rl-llms r05
Every number comes from code/r05_ppo (ppo.py, 2026-11-28).
"""

LESSON = {
    "series": "Reinforcement Learning for LLMs",
    "label": "Episode 5",
    "title": "PPO: Small, Safe Steps",
    "tagline": "Squeeze more learning out of every answer, without breaking the model",
    "duration": "2:11",
    "intro": """<p><b>PPO</b> (proximal policy optimization) is the algorithm behind classic RLHF. Its idea is to
reuse each batch of sampled answers for several updates, because for a large model sampling is the expensive part.
This lesson builds its three pieces (the probability ratio, the clip and a learned value baseline) and measures them on
our addition model. PPO learned more from the same 102,400 answers than REINFORCE, and the clip made it reliable.
But it still needed a smaller step size, and it costs more computing per batch.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 2–4 of this series (REINFORCE, baselines,
the KL leash). Code: <code>code/r05_ppo</code> (Python with PyTorch; about 8 minutes on 16 threads, needs episode 1's
<code>base.pt</code>).</div>""",
}

CONCEPTS = [
    {
        "title": "Reusing a batch, and why it is risky",
        "segment": (7.7, 35.9),
        "figures": [{"t": 24.2, "caption": "Generating answers is the expensive part. REINFORCE takes one step per "
                                           "batch and discards it; PPO reuses each batch for many steps."},
                    {"t": 35.7, "caption": "More steps on the same answers act like bigger steps: REINFORCE with 3× "
                                           "bigger steps ends at 3%, below the 14% it started from."}],
        "body": [
            """<p>Each RL batch costs two things: <b>generating</b> the answers (one forward pass per token, for every
answer) and <b>learning</b> from them (a backward pass). For a large language model, generation usually dominates.
REINFORCE uses each batch for a single update and throws it away. PPO squeezes more out of it by taking several
updates on the same answers; we used 4 passes over the batch in 4 minibatches, 16 updates per batch.</p>""",
            "{fig0}",
            """<p>Reusing a batch is risky for the same reason big steps are risky (episode 4): many updates on the same
answers move the model far. REINFORCE with a step size of 0.001 instead of 0.0003 fell to <b>3%</b> after 400
batches (from 14%), in all three runs.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>PPO reuses each batch for several updates to save the
expensive sampling; the danger is moving the model too far on old answers.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "We used 400 batches of 32 questions × 8 tries. How many answers did each method "
                                    "sample?",
             "answer": "102,400.", "why": "400 × 256.",
             "key": {"parts": [{"label": None, "value": 102400, "tol": 0, "unit": None}]}},
            {"kind": "number", "q": "With 4 epochs and 4 minibatches, how many updates does PPO make per batch?",
             "answer": "16.", "why": "Each of the 4 passes splits the batch into 4 minibatches: 4 × 4.",
             "key": {"parts": [{"label": None, "value": 16, "tol": 0, "unit": None}]}},
        ],
    },
    {
        "title": "The ratio and the clip",
        "segment": (36.4, 60.8),
        "figures": [{"t": 47.9, "caption": "After one update the batch comes from an older model; the ratio p now / p "
                                           "when sampled, per token, scales each push."},
                    {"t": 60.6, "caption": "The clipped objective: once the ratio leaves 0.8–1.2 in the direction the "
                                           "advantage wants, it goes flat and that token stops pushing."}],
        "body": [
            """<p>After the first update, the answers in the batch were sampled by an <b>older</b> version of the
model. PPO corrects for this token by token with the <b>ratio</b> r = p<sub>now</sub>(token) / p<sub>when
sampled</sub>(token) and pushes with r × advantage. At the first update r = 1 and this is plain REINFORCE.</p>
<p>Then the <b>clip</b>. PPO's objective is min(r × A, clip(r, 0.8, 1.2) × A). If the advantage A is positive, the
objective stops growing once r passes 1.2: a good token that is already 20% likelier than when it was sampled gets no
further push. If A is negative, it stops once r falls below 0.8. A flat objective has zero gradient, so no single answer
can drag the model far during the reuse. In our PPO runs about 12% of answer tokens were clipped.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>The ratio accounts for the batch being older; the clip
switches off each token's push once it has moved more than 20% in the wanted direction.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A token had probability 0.25 when sampled and has 0.35 now. What is the ratio?",
             "answer": "1.4.", "why": "0.35 / 0.25.",
             "key": {"parts": [{"label": None, "value": 1.4, "tol": 0.001, "unit": None}]}},
            {"kind": "tf", "q": "“With a positive advantage and ratio 1.4, the clipped objective still pushes this "
                                "token up.”",
             "answer": "False.", "why": "Above 1.2 the objective uses the clipped value 1.2 × A, which does not depend "
                                        "on the model: the gradient is zero.",
             "key": {"value": False}},
            {"kind": "mc", "q": "A token with a negative advantage has ratio 1.5 (the model made it likelier by "
                                "mistake). Does the clip stop the correction?",
             "options": ["Yes, the ratio is outside 0.8–1.2", "No: for a negative advantage only ratios below 0.8 are "
                         "clipped, so the push down continues", "Only if the advantage is small",
                         "The token is dropped from the batch"],
             "answer": "B.", "why": "min(r·A, clip(r)·A) with A &lt; 0 picks r·A when r &gt; 1.2, so the gradient "
                                    "remains and pushes the mistake back.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "A learned baseline: the value head",
        "segment": (61.2, 73.0),
        "figures": [{"t": 72.8, "caption": "The value head, one linear layer on the hidden state at '=', predicts the "
                                           "expected reward; advantage = reward − prediction."}],
        "body": [
            """<p>PPO also needs a baseline (episode 3). Instead of averaging several tries at the same question, it
<b>learns</b> one: a <b>value head</b>, here a single linear layer reading tiny_gpt's hidden state at the “=” that
ends the question, trained to predict the reward to expect (squared error against the actual rewards). The advantage
is reward − predicted reward. By the end of our runs the head predicted an average of 0.23–0.30 while the batches
earned 0.25–0.32: it tracks the model's success rate as it improves.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>The value head is a learned baseline: a prediction of how
well the model usually does on this question.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The value head predicts 0.3 for a question and the answer is right (reward 1). "
                                    "What is the advantage?",
             "answer": "0.7.", "why": "1 − 0.3.",
             "key": {"parts": [{"label": None, "value": 0.7, "tol": 0, "unit": None}]}},
        ],
    },
    {
        "title": "What we measured",
        "segment": (73.5, 106.2),
        "figures": [{"t": 92.2, "caption": "Same 400 batches: REINFORCE 23%, PPO (16 updates per batch, step "
                                           "0.0001) 27%, no clip 13%, 18%, 28%."},
                    {"t": 106.0, "caption": "At REINFORCE's step size (0.0003), 16 updates per batch drop PPO to 9% and "
                                            "PPO without the clip to 1%."}],
        "body": [
            """<p>Every method got the same 400 batches, 3 runs each. Final accuracy on the 1,000 test questions
(start 14%): REINFORCE at its usual step size 0.0003: 23.4, 23.9 and 21.5% (mean 23%). <b>PPO</b> with 16 updates
per batch at a third of that step size (0.0001): <b>28.7, 25.5 and 27.7%</b> (27%). Every PPO run beat every
REINFORCE run. The same 16 updates <b>without the clip</b>: 18.5, 12.9 and 27.9%: sometimes as good, sometimes
much worse.</p>""",
            "{fig0}",
            """<p>The clip is not a free pass. At REINFORCE's step size of 0.0003, PPO's first batch moved the model
0.30–0.48 nats away from the model that sampled it (0.11–0.12 at 0.0001), and accuracy fell to 3–4% within 40
batches, ending at <b>9%</b> (no clip: <b>1%</b>). The clip limits each token's contribution, but every update still
changes all the weights, and 16 of them at a large step add up. Real PPO setups pair the clip with small step sizes
and often stop a batch's updates early when the KL grows too large.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>With a small enough step, PPO learned more per answer
than REINFORCE, and the clip made it reliable; with a large step, nothing saved it.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many percentage points did PPO gain over REINFORCE after 400 batches (means "
                                    "of 3 runs)?",
             "answer": "About 4 (27.3% vs 22.9%).", "why": "Read from the table in the README.",
             "key": {"parts": [{"label": None, "value": 4.4, "tol": 0.6, "unit": None}]}},
            {"kind": "tf", "q": "“The clip makes the learning rate irrelevant.”",
             "answer": "False.", "why": "At 0.0003, PPO with the clip ended at 9%, far below REINFORCE's 23%.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "The trade-off",
        "segment": (106.7, 124.9),
        "figures": [{"t": 119.6, "caption": "PPO buys more learning per answer with more computing per batch; for "
                                            "language models the value head is usually a second network as big as the "
                                            "model."},
                    {"t": 124.7, "caption": "In code: the per-token ratio, the clipped ratio, and the smaller of the two "
                                            "objectives."}],
        "body": [
            """<p>PPO buys more learning per sampled answer, and pays in computing: 16 backward passes per batch
instead of one (on our CPU a PPO batch took about 2.7× as long). For language models there is a bigger cost. The
value head is usually not one linear layer but a <b>second network</b> the size of the policy (often initialised
from the reward model), with its own memory and training. Removing it is exactly what the next method, GRPO, does.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>PPO trades computing and a value model for more learning
per answer; GRPO keeps the clip and drops the value model.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why does reusing a batch matter more for a 70-billion-parameter model than for "
                                   "tiny_gpt?",
             "answer": "Generating long answers with a huge model is very expensive, so getting more updates out of "
                       "each batch saves most of the cost; for tiny_gpt, sampling is cheap.",
             "why": "PPO saves samples, not computing."},
            {"kind": "code", "q": "<b>Try it yourself.</b> The clipped PPO objective.",
             "code": """ratio = torch.exp(logp_new - logp_old)          # per token
clipped = ratio.clamp(1 - 0.2, 1 + 0.2)
loss = -torch.min(ratio * adv, clipped * adv).mean()""",
             "answer": "ppo.py compares REINFORCE, PPO and PPO without the clip; try 8 epochs instead of 4.",
             "why": "Three lines turn REINFORCE into PPO's clipped surrogate."},
        ],
    },
]
