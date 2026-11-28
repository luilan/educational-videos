"""Study guide content for Reinforcement Learning for LLMs, episode 4: The KL Leash.

Build:  python framework/study_guide.py rl-llms r04
Every number comes from code/r04_kl_leash (kl_leash.py, 2026-11-28).
"""

LESSON = {
    "series": "Reinforcement Learning for LLMs",
    "label": "Episode 4",
    "title": "The KL Leash",
    "tagline": "Chase the reward, but stay close to where you started",
    "duration": "2:29",
    "intro": """<p>RL can wreck a model. With steps ten times bigger, our model collapsed within 150 updates into
answering 111 to every question, and it could never recover. The standard protection is a <b>KL penalty</b>: keep a
frozen copy of the starting model, the <b>reference</b>, and charge the model for every answer it makes more likely
than the reference did. This lesson builds that penalty, shows it rescuing the collapsed run, and measures what it
costs when the leash is too tight.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–3 of this series (REINFORCE, the
per-question baseline); Foundations episodes on logarithms and probability. Code: <code>code/r04_kl_leash</code>
(Python with PyTorch; about 17 minutes on a CPU, needs episode 1's <code>base.pt</code>).</div>""",
}

CONCEPTS = [
    {
        "title": "When RL steps are too big",
        "segment": (7.7, 38.3),
        "figures": [{"t": 22.8, "caption": "Learning rate 0.0003 takes the model from 14% to 73%. Ten times bigger "
                                           "steps: within 150 updates it answers 111 to every question, 1.5% right."},
                    {"t": 38.1, "caption": "All 8 tries give the same answer and the same reward, so every "
                                           "per-question advantage is 0 − 0 = 0: nothing left to learn from."}],
        "body": [
            """<p>In episode 2, REINFORCE with a learning rate of 0.0003 took our model from 14% to 73% right in
2,000 steps. We ran the same thing with a learning rate of 0.003, ten times bigger. Within <b>150 steps</b> the model
answered <b>“111.”</b> to every question, which is right only for the 1.5% of questions whose sum is 111.</p>""",
            "{fig0}",
            """<p>A few large updates pushed one answer up so hard that the model became <b>certain</b> of it. Its
sampled answers stopped varying: all 8 tries at a question are the same answer with the same reward. With the
per-question baseline, every advantage is then reward − average = 0, so the gradient is zero. RL only learns by
comparing different attempts; once the attempts are identical, there is nothing to compare, and the model is
<b>stuck for good</b>.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>RL can collapse a model into one answer; once all its
samples agree, it gets no learning signal and cannot climb back out.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A question's 8 tries are all “111.” and all wrong. With the per-question "
                                    "baseline, what is the advantage of each try?",
             "answer": "0.", "why": "The rewards are all 0, their mean is 0, and 0 − 0 = 0.",
             "key": {"parts": [{"label": None, "value": 0, "tol": 0, "unit": None}]}},
            {"kind": "tf", "q": "“If a collapsed model happened to answer correctly every time, it would also get "
                                "no learning signal from the per-question baseline.”",
             "answer": "True.", "why": "All rewards 1, mean 1, every advantage 0. That case is harmless, because the "
                                       "answers are already right.",
             "key": {"value": True}},
        ],
    },
    {
        "title": "The KL penalty",
        "segment": (38.8, 61.0),
        "figures": [{"t": 60.7, "caption": "The reference is a frozen copy from before training. Each answer's "
                                           "reward is reduced by β × (log p_model − log p_reference); the average of "
                                           "that difference is the KL divergence."}],
        "body": [
            """<p>Before training starts, make a <b>frozen copy</b> of the model: the <b>reference</b>. It is never
updated. For every answer the model samples, compute how much more likely the model makes it than the reference
does, as a difference of log-probabilities:</p>
<p style="text-align:center"><code>d = log p_model(answer) − log p_reference(answer)</code></p>
<p>and train on a <b>shaped reward</b> instead of the raw one:</p>
<p style="text-align:center"><code>shaped = reward − β · d</code></p>
<p>An answer the model has made far more likely than the reference did (large <i>d</i>) loses reward. The strength β
sets how short the leash is. Averaged over the model's own answers, <i>d</i> is the <b>KL divergence</b> from the
model to the reference, in nats per answer: 0 when the two models are identical, larger the further the model has
moved. So the penalty is a running charge for distance.</p>
<p>One useful fact: for a single question, the best possible model under this shaped reward makes each answer's
probability proportional to <code>p_reference(answer) · e<sup>reward/β</sup></code>. A small β lets the reward
dominate; a large β keeps the model close to the reference.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>The KL penalty subtracts β × (how much more likely the
model makes this answer than the frozen reference) from every reward, so moving away from the start has a price.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The model gives an answer probability 0.8; the reference gave it 0.1. Using "
                                    "natural logs, what is d = log p_model − log p_reference (2 decimals)?",
             "answer": "2.08.", "why": "ln 0.8 − ln 0.1 = ln 8 ≈ 2.08.",
             "key": {"parts": [{"label": None, "value": 2.08, "tol": 0.01, "unit": None}]}},
            {"kind": "number", "q": "With β = 0.1, a right answer (reward 1) has d = 2.08. What is its shaped "
                                    "reward (2 decimals)?",
             "answer": "0.79.", "why": "1 − 0.1 × 2.08 = 0.792.",
             "key": {"parts": [{"label": None, "value": 0.79, "tol": 0.01, "unit": None}]}},
            {"kind": "mc", "q": "Why must the reference be frozen?",
             "options": ["It saves memory", "If it moved with the model, d would stay near 0 and the leash would "
                         "hold nothing", "Frozen models sample faster", "The reward model requires it"],
             "answer": "B.", "why": "The penalty measures distance from a fixed point; a moving point measures "
                                    "nothing.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "What the leash buys, and what it costs",
        "segment": (61.5, 112.5),
        "figures": [{"t": 73.2, "caption": "Same 10× steps with β = 0.1: odd answers for a while (“14+.”, “.”), "
                                           "then recovery to 61%. KL 6.9 instead of 23."},
                    {"t": 92.1, "caption": "At the normal learning rate: β up to 0.03 keeps ~70% accuracy; β = 0.1 "
                                           "gives 23%, β = 1 barely moves from the start (15%)."},
                    {"t": 112.3, "caption": "Answers to 47+38: the reference mixes 74, 75, 81; no leash says 85 every "
                                            "time; β = 1 keeps the reference's mix."}],
        "body": [
            """<p><b>The rescue.</b> We repeated the 10× run with β = 0.1. For a few hundred steps the model
wobbled and even produced malformed answers such as “14+.” and “.”, but it never locked into one answer: the penalty
makes a collapsed answer expensive, because the model gives it a far higher probability than the reference does.
After 2,000 steps it was <b>61%</b> right, with a KL of 6.9 nats, against 1.5% and a KL of 23 without the leash.</p>""",
            "{fig0}",
            """<p><b>The cost.</b> At the normal learning rate (no collapse to prevent), we tried six values of β,
2 runs each. Final accuracy (KL): β = 0: 73% (7.5); 0.01: 74% (7.3); 0.03: 67% (6.3); 0.1: 23% (1.2); 0.3: 16%
(0.6); 1: 15% (0.3), against 14% at the start. A loose leash costs almost nothing; past about 0.1, the penalty for
moving outweighs the reward for getting answers right within 2,000 steps, and the model hardly learns.</p>""",
            "{fig1}",
            """<p><b>How the answers read.</b> For 47+38 the reference samples a mix of wrong answers (74, 75, 81).
With no leash the model says 85 every time; with β = 1 it keeps almost exactly the reference's mix. In arithmetic a
single right answer is the goal, so losing variety is fine. For a chat model, the reference's spread of answers is
its style, its phrasing and what it knows, and that is what the leash protects.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>β trades reward against distance: too loose and nothing
stops a collapse; too tight and the model cannot learn.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In our sweep, how many percentage points of accuracy did going from β = 0.03 to "
                                    "β = 0.1 cost?",
             "answer": "About 44 (67% → 23%).", "why": "Mean of 2 runs each: 67.3% and 22.8%.",
             "key": {"parts": [{"label": None, "value": 44, "tol": 1, "unit": None}]}},
            {"kind": "tf", "q": "“With β = 0.1 at the 10× learning rate, the model never produced a malformed "
                                "answer.”",
             "answer": "False.", "why": "Around step 500 it sampled answers like “14+.” and “.”; it recovered "
                                        "afterwards.",
             "key": {"value": False}},
            {"kind": "short", "q": "Why did the KL penalty stop the collapse into “111.”?",
             "answer": "The collapsed model gives “111.” a probability near 1 where the reference gave it very little, "
                       "so d is large and the shaped reward of that answer drops far below the alternatives.",
             "why": "The penalty grows with how far the model's probabilities move from the reference."},
        ],
    },
    {
        "title": "Why language models always keep a leash",
        "segment": (113.0, 140.7),
        "figures": [{"t": 130.9, "caption": "The reward model is imperfect and a drifting model finds text that "
                                            "scores well; the reference is the model before RL. PPO puts the penalty "
                                            "in the reward, GRPO in the loss."},
                    {"t": 140.5, "caption": "In code: the log-probability difference against the frozen reference, "
                                            "then β times it subtracted from the reward, before the baseline."}],
        "body": [
            """<p>Our checker cannot be fooled, so the danger here was collapse. With language models the reward
usually comes from a <b>reward model</b> (episode 7), which is imperfect: a model free to drift finds text the reward
model likes that people would not (episode 8, reward hacking). The leash limits how far it can drift. The reference
is the model as it was before RL, usually the instruction-tuned model.</p>
<p>The two methods ahead put the same penalty in different places. <b>PPO</b> (episode 5), as used in RLHF,
subtracts it from the reward, exactly as we did. <b>GRPO</b> (episode 6) adds a KL term to the loss instead.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>The reference model plus a KL penalty is the standard
guard in RL for language models: it keeps style and knowledge while the model chases an imperfect score.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "In RLHF, which model is usually the reference?",
             "options": ["The reward model", "The model as it was before RL (instruction-tuned)",
                         "A larger teacher model", "The model from the previous step"],
             "answer": "B.", "why": "The leash keeps the RL-trained model close to where RL started.",
             "key": {"choice": 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> The leash in REINFORCE.",
             "code": """kl = logp.detach() - ref_logp    # vs the frozen reference
shaped = r - beta * kl            # the leash
g = shaped.view(Q, G)
adv = (g - g.mean(1, keepdim=True)).view(-1)
loss = -(adv * logp).mean()""",
             "answer": "kl_leash.py runs the β sweep and the 10× learning-rate runs; try β = 0.05 at the 10× rate.",
             "why": "Two lines turn plain REINFORCE into the leashed version used for language models."},
        ],
    },
]
