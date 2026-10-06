"""Study guide content for How LLMs Work: Deep Dive, episode 28: Mixture of Experts.

Build:  python framework/study_guide.py deep-dive d28 --video deep-dive/media/videos/d28_scene/1080p60/MoEVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d28_moe/moe.py (4-layer tiny GPTs trained 2,000 steps; torch 2.14.0, CPU), except Mixtral's
public configuration. The router scores in the second figure are an illustration.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 28",
    "title": "Mixture of Experts",
    "tagline": "Many experts, only a few awake per token",
    "duration": "2:13",
    "intro": """<p>This lesson answers one question: can a model have many more parameters without more compute per
token? A <b>mixture of experts</b> (MoE) replaces each MLP with several expert MLPs and a <b>router</b> that sends every
token to its top 1 or 2. In a tiny GPT, top-1 routing did not beat a dense model at the same compute, but top-2 routing
beat a dense model of the same active size (1.596 vs 1.628), and a balancing loss kept all 8 experts busy.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episode 15 (the MLP and its activation) and
episode 23 (scaling laws). Code: <code>code/d28_moe</code> (PyTorch only; five tiny models, about an hour on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Experts and a router",
        "segment": (8, 51),
        "figures": [{"t": 37.1, "caption": "The router scores the experts, keeps the best one or two, and mixes their outputs; "
                                           "the others don't run (scores illustrative)."},
                    {"t": 50.2, "caption": "8 experts per layer: 5.5x the parameters, the same compute per token with top-1."}],
        "body": [
            """<p>A <b>mixture of experts</b> has many more parameters than it uses for each token. Mixtral 8x7B, for
example, has 8 experts per layer and uses 2 for each token. The MLP in each block is replaced by several expert MLPs plus
a small <b>router</b> (one linear layer and a softmax). For every token, the router scores the experts, keeps the best
k, and mixes their outputs by those (renormalized) scores; the other experts don't run at all.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In our tiny GPT, 8 experts per layer make <b>4,510,273</b> parameters, 5.5 times the dense model's
818,241. With top-1 routing each token still uses about the same as before: <b>822,337</b> (the router adds a little).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Parameters and compute are decoupled: total size grows with the
number of experts, compute per token with k.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Router scores for two chosen experts are 0.41 and 0.27. With what weights are their "
                                    "outputs mixed?",
             "answer": "About 0.60 and 0.40.", "why": "Renormalized: 0.41 / 0.68 and 0.27 / 0.68."},
            {"kind": "number", "q": "One expert MLP here has 131,712 parameters. With top-2 instead of top-1, how many more "
                                    "parameters are active per token across the 4 layers?",
             "answer": "526,848.", "why": "4 layers × 1 extra expert × 131,712 = 526,848 (1,349,185 − 822,337)."},
        ],
    },
    {
        "title": "Top-1, balancing, top-2",
        "segment": (51, 101),
        "figures": [{"t": 82.0, "caption": "Without balancing the router plays favorites (4% to 20% of tokens); with a "
                                           "balancing loss, 11% to 14%."},
                    {"t": 100.1, "caption": "Top-2 MoE 1.596 vs a dense model of the same active size 1.628; top-1 1.687 vs "
                                            "dense 1.644."}],
        "body": [
            """<p>After 2,000 steps, the dense model reaches <b>1.644</b> and MoE with top-1 routing <b>1.687</b>: worse.
At this small scale and short training, each expert sees only an eighth of the tokens, and it doesn't pay off.</p>""",
            """<p>Without help, the router also plays favorites: in layer 2 one expert gets 20% of the tokens, another
only 4%. A small <b>load-balancing loss</b> (weight 0.01) that rewards even use fixes it: every expert then gets 11% to
14% (loss 1.695).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>With <b>top-2</b> routing the loss drops to <b>1.596</b>, the best of all. But each token now uses 1.35
million parameters, so the fair comparison is a dense model just as big: an MLP twice as wide (1.34 million) reaches
<b>1.628</b>. The mixture still wins.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Compare MoE with a dense model of the same compute per token;
here top-2 won that comparison and top-1 did not.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Order from best to worst: <i>dense · MoE top-1 · MoE top-2 · dense twice as wide</i>.",
             "answer": "MoE top-2 (1.596) → dense twice as wide (1.628) → dense (1.644) → MoE top-1 (1.687).",
             "why": "From the episode's runs."},
            {"kind": "tf", "q": "“Without a balancing loss, the router sends tokens evenly to all experts.”",
             "answer": "False.", "why": "Shares ranged from 4% to 20% in layer 2."},
            {"kind": "short", "q": "Why is a dense model “twice as wide” the fair baseline for top-2 MoE?",
             "answer": "It uses about the same number of active parameters (compute) per token, 1.34 million vs 1.35 million, "
                       "so any difference comes from MoE's extra total parameters, not extra compute.",
             "why": "Comparing top-2 MoE with the small dense model would mix two effects."},
        ],
    },
    {
        "title": "The deal, and the code",
        "segment": (101, 124),
        "figures": [{"t": 113.5, "caption": "Same compute per token, far more parameters; the price: memory for all experts and "
                                            "care to keep them busy."},
                    {"t": 122.5, "caption": "Router, top-k, and each expert run only on the tokens sent to it."}],
        "body": [
            """<p>That is the deal: at the same compute per token, a mixture of experts can hold far more parameters, and
it can come out ahead. The price is memory for all the experts (every one must be loaded, even if few run) and some care
to keep them all busy. In code, the router is a linear layer and a softmax, <code>topk</code> picks the experts, and each
expert runs only on the tokens sent to it.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>MoE trades memory for compute: big in storage, small in work per
token.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which cost does a mixture of experts NOT reduce?",
             "options": ["Compute per token", "Memory to hold the model", "Active parameters per token", "None of these"],
             "answer": "B.", "why": "All experts must be stored, even though only k run per token."},
        ],
    },
]
