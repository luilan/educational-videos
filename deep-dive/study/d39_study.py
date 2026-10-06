"""Study guide content for How LLMs Work: Deep Dive, episode 39: DPO.

Build:  python framework/study_guide.py deep-dive d39 --video deep-dive/media/videos/d39_scene/1080p60/DPOVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d39_dpo/dpo.py (GPT-2 small; 1,536 seeded samples → 215 pairs; DPO β 0.1, Adam lr 1e-5,
3 epochs, 51 steps; evaluation on 64 fresh seeded samples, KL = mean per-sequence sum of log p_policy − log p_reference
on those samples) and, for the comparison, code/d38_ppo/ppo.py, or from the arithmetic shown.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 39",
    "title": "DPO",
    "tagline": "Preferences without reinforcement learning",
    "duration": "2:14",
    "intro": """<p>This lesson answers one question: can we learn from preferences without the whole RL machinery? <b>Direct
Preference Optimization</b> uses a mathematical shortcut: under a KL leash, the reward can be written in terms of the
policy itself, so the model is its own reward model. Training becomes one loss over fixed preference pairs. On episode
38's task, DPO raises GPT-2's positivity from 0.14 to 1.08 in 51 steps (about two minutes), keeping fluent text; PPO
reached 0.92 in about 19 minutes.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 37 (reward models) and 38 (RLHF with
PPO). Code: <code>code/d39_dpo</code> (GPT-2 small).</div>""",
}

CONCEPTS = [
    {
        "title": "The model is its own reward model",
        "segment": (8, 54),
        "figures": [{"t": 38.0, "caption": "Under a KL leash, reward(x, y) = β · log(p_model(y|x) / p_ref(y|x)) plus a term that "
                                           "depends only on x."},
                    {"t": 53.4, "caption": "Substituting into the Bradley–Terry loss gives DPO."}],
        "body": [
            """<p>RLHF with PPO maximises reward minus β × KL from a reference model. That objective has a known best
policy: p*(y|x) ∝ p_ref(y|x) · exp(reward(x, y) / β). Turned around, the reward can be written through the policy:
<b>reward(x, y) = β · log(p(y|x) / p_ref(y|x))</b> + a term that depends only on the prompt x.</p>""",
            "{fig0}",
            """<p>Plug that into the reward model's loss from episode 37, −log σ(r_chosen − r_rejected): the prompt term
cancels, and you get the <b>DPO loss</b> −log σ(β · [Δ_chosen − Δ_rejected]) with Δ = log p(y|x) − log p_ref(y|x). It
raises the chosen answer's likelihood relative to the reference and lowers the rejected one's. No reward model, no value
head, no sampling during training; the reference model is still needed.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>DPO fits the policy so that its implicit reward explains the
preferences.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "β = 0.1, the chosen answer's log ratio Δ is +2 and the rejected one's −3. What is the DPO "
                                    "loss for this pair? And when both Δ are 0 (the start of training)?",
             "answer": "0.474; 0.693.", "why": "Margin 0.1 × 5 = 0.5, −log σ(0.5) = 0.474; −log σ(0) = ln 2."},
            {"kind": "tf", "q": "“DPO needs no reference model.”",
             "answer": "False.", "why": "Δ is measured against the reference; it plays the role of PPO's KL anchor."},
        ],
    },
    {
        "title": "DPO on GPT-2",
        "segment": (54, 98),
        "figures": [{"t": 68.0, "caption": "1,536 samples scored once → 215 pairs (every sample scoring above 0 vs a lower one)."},
                    {"t": 81.4, "caption": "DPO loss over 51 steps: 0.693 → under 0.3."},
                    {"t": 96.8, "caption": "Fresh samples: reward 0.14 → 1.08, fluent text, KL 3.8."}],
        "body": [
            """<p>The same goal as episode 38: positive continuations, scored by positive minus negative words. 1,536
continuations are sampled from GPT-2 once; every one that scored above zero is paired with a random lower-scoring one
of the same prompt: 215 preference pairs. The original GPT-2 is the reference.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>51 steps (3 epochs of batches of up to 16 pairs): the loss falls from 0.693 to under 0.3, and from the
second epoch the implicit reward ranks the chosen answer higher in every logged batch. On fresh samples the reward rises
from <b>0.14 to 1.08</b>, the text stays fluent (“Beautiful design with fantastic functions. And I love that…”), and the
KL from GPT-2 is 3.8.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>DPO's data is fixed in advance; all the learning signal is in
the pairs.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Only 215 of 1,536 samples became pairs. Why so few?",
             "answer": "Most samples score 0 (no positive or negative words), and a pair needs a sample scoring above 0 plus "
                       "a lower-scoring one.", "why": "Plain GPT-2 averages only 0.14 positive words per continuation."},
        ],
    },
    {
        "title": "PPO vs DPO",
        "segment": (98, 123),
        "figures": [{"t": 113.2, "caption": "PPO (β 0.5): 0.92, ≈ 19 min, reward model + value head + sampling. DPO: 1.08, ≈ 2 "
                                            "min."}],
        "body": [
            """<p>PPO at β = 0.5 reached 0.92 after about 19 minutes of sampling and scoring; DPO reached 1.08 in about
two minutes, with no reward model, value head or sampling in the loop. (The KLs aren't measured identically: PPO's ≈ 6
comes from its last training batch, DPO's 3.8 from fresh samples.) That simplicity is a large part of why many open
models are preference-tuned with DPO or its variants. PPO keeps one advantage: it learns from the model's own fresh
samples, so it can keep improving beyond the fixed data.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Same objective, two routes: PPO optimises a learned reward
online; DPO solves for the policy directly from offline pairs.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What does DPO need that ordinary supervised fine-tuning does not?",
             "options": ["A reward model", "A rejected answer for each example, and a reference model",
                         "Sampling during training", "A value head"],
             "answer": "B.", "why": "The loss compares chosen with rejected, both relative to the reference."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Compute DPO losses for a few pairs.",
             "code": """import torch, torch.nn.functional as F
beta = 0.1
d_c = torch.tensor([2.0, 0.0, -1.0])   # log p - log p_ref, chosen
d_r = torch.tensor([-3.0, 0.0, 1.0])   # same, rejected
print(-F.logsigmoid(beta * (d_c - d_r)))""",
             "answer": "About [0.474, 0.693, 0.798].", "why": "Margins 0.5, 0, −0.2."},
        ],
    },
]
