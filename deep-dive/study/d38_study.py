"""Study guide content for How LLMs Work: Deep Dive, episode 38: RLHF with PPO.

Build:  python framework/study_guide.py deep-dive d38 --video deep-dive/media/videos/d38_scene/1080p60/PPOVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d38_ppo/ppo.py (GPT-2 small + value head, Adam lr 2e-5, 200 iterations of 16 samples × 24
tokens, 4 PPO epochs, clip 0.2, GAE λ 0.95; seeded; KL = per-sequence sum of log p_policy − log p_reference in the last
training batch) or from the arithmetic shown.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 38",
    "title": "RLHF with PPO",
    "tagline": "Optimizing a model against a reward",
    "duration": "2:17",
    "intro": """<p>This lesson answers one question: how is a language model trained against a reward? <b>PPO</b> samples
text, scores it, estimates which tokens did better than expected, and takes small clipped steps toward them. Asked to
write positive text, GPT-2 with no constraint learns to output nothing but positive words: a perfect reward and useless
text (<b>reward hacking</b>). A <b>KL penalty</b> to the original model is the leash: too weak (β = 0.05) and it still
hacks; at β = 0.5 the reward rises from 0.14 to 0.92 and the text stays English.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 37 (reward models) and 35 (decoding).
Code: <code>code/d38_ppo</code> (GPT-2 small; each setting takes about 20 minutes on an 8-thread CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "The PPO loop",
        "segment": (8, 60),
        "figures": [{"t": 22.0, "caption": "RLHF: the model writes, a reward model scores, PPO updates the model."},
                    {"t": 58.8, "caption": "Each round: 16 samples, scores, advantages from a value head, a few clipped steps."}],
        "body": [
            """<p>In RLHF the model writes, a reward model scores, and an algorithm nudges the model toward higher scores.
Here the goal is positive continuations of openings like “The movie was”, and the reward is a stand-in for a reward
model: positive words minus negative words in 24 new tokens. Plain GPT-2 scores <b>0.14</b> on average.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Each PPO round: GPT-2 writes 16 continuations; each gets its score; a small <b>value head</b> on the
model's hidden state estimates the expected score, so every token gets an <b>advantage</b> (better or worse than
expected, computed with GAE). Then 4 gradient steps on the clipped objective
min(ratio · A, clip(ratio, 0.8, 1.2) · A), where ratio = p_new(token) / p_old(token): once a token's probability has
moved 20%, it gets no further push that round.</p>""",
            """<div class="box key"><b class="t">Key idea</b>PPO = policy gradient with advantages, plus clipping so one
batch can't move the policy too far.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The clipped objective min(ratio · A, clip(ratio, 0.8, 1.2) · A): what is it for ratio 1.5 "
                                    "and A = +1? For ratio 1.5 and A = −1?",
             "answer": "1.2; −1.5.", "why": "For A > 0 the clip caps the gain at 1.2; for A < 0 the minimum keeps the "
                                           "unclipped, more negative value, so bad moves are still corrected."},
        ],
    },
    {
        "title": "Reward hacking",
        "segment": (60, 75),
        "figures": [{"t": 74.0, "caption": "Without a penalty the reward reaches 24 of 24: every token a positive word."}],
        "body": [
            """<p>With no constraint, the reward climbs to <b>24 out of 24</b> within about 70 iterations: every token is a
positive word. “The movie was wonderful good wonderful good great great wonderful…”. The KL from the original GPT-2
reaches 59. The reward is perfect and the text useless: reward hacking, the failure episode 37 warned about.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>An optimizer finds the cheapest way to raise the reward, not
the way you meant.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Name one change to the reward itself that would make this particular hack less "
                                   "attractive.",
             "answer": "Cap the reward (e.g. at 1 or 2), count each word once, or penalise repetition/non-fluent text.",
             "why": "But every patch invites a new hack; that is why a KL leash is used as well."},
        ],
    },
    {
        "title": "The KL leash",
        "segment": (75, 126),
        "figures": [{"t": 101.3, "caption": "β = 0.05 still hacks (“great great great…”); KL at the end 25 vs 59."},
                    {"t": 116.6, "caption": "β = 0.5: reward 0.14 → 0.92 on fresh samples, text still English, KL ≈ 6."}],
        "body": [
            """<p>The standard fix: every token pays β · (log p_new − log p_GPT-2), so the summed penalty estimates the KL
divergence from the original model. With β = 0.05 the leash is too weak: the policy still collapses, now onto
“great great great…”, and the KL ends at 25 instead of 59. With β = 0.5 the reward rises only from 0.14 to 0.92, but
the text stays English: “…one of the great films of 2014…”. The KL stays around 6.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<table><tr><th>β</th><th>reward (fresh samples)</th><th>KL at the end</th><th>text</th></tr>
<tr><td>0</td><td>24.00</td><td>59</td><td>positive words only</td></tr>
<tr><td>0.05</td><td>24.00</td><td>25</td><td>“great great great …”</td></tr>
<tr><td>0.5</td><td>0.92</td><td>≈ 6</td><td>fluent, more positive</td></tr></table>""",
            """<div class="box key"><b class="t">Key idea</b>β sets the trade-off between reward and staying close to the
original model; it must be large relative to the reward the hack could earn.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With β = 0.5, what does one token cost if the new model gives it probability 0.5 and "
                                    "GPT-2 gave it 0.05?",
             "answer": "About 1.15.", "why": "0.5 × ln(0.5 / 0.05) = 0.5 × ln 10."},
            {"kind": "number", "q": "Why did β = 0.05 fail? Compare the total penalty at KL 25 with the hacked reward.",
             "answer": "Penalty ≈ 1.25, reward 24.", "why": "0.05 × 25 = 1.25 is tiny next to a reward of 24, so the hack "
                                                            "still pays."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Evaluate the clipped objective for a few ratios.",
             "code": """import torch
ratio = torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5])
for A in (1.0, -1.0):
    print(A, torch.min(ratio * A, ratio.clamp(0.8, 1.2) * A))""",
             "answer": "A = 1: [0.5, 0.9, 1.0, 1.1, 1.2]; A = −1: [−0.8, −0.9, −1.0, −1.1, −1.5].",
             "why": "Gains are capped beyond the clip range; losses are not."},
        ],
    },
]
