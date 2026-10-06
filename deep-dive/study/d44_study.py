"""Study guide content for How LLMs Work: Deep Dive, episode 44: Sparse Autoencoders.

Build:  python framework/study_guide.py deep-dive d44 --video deep-dive/media/videos/d44_scene/1080p60/SAEVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d44_sae/sae.py (GPT-2 small, residual stream after layer 6, 768 → 6,144 features, L1 2.0,
3,000 Adam steps on Tiny Shakespeare activations, seeded), from the L1 sweep described in its README, or from the
arithmetic shown.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 44",
    "title": "Sparse Autoencoders",
    "tagline": "Unpacking features from superposition",
    "duration": "2:20",
    "intro": """<p>This lesson answers one question: if a model stores features in superposition, can we get them back
out? A <b>sparse autoencoder</b> (SAE) re-expresses each activation as a combination of many more features than
dimensions, only a few of them active at a time. On GPT-2's layer 6, 30 of 6,144 features per token rebuild 80% of the
variance and keep 93% of the layer's effect on the next-token loss, and many features are readable: “chief”, a speaker
name, a family of learned words.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 43 (superposition) and 13 (the residual
stream). Code: <code>code/d44_sae</code> (PyTorch + transformers, about 30 minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "The sparse autoencoder",
        "segment": (8, 55),
        "figures": [{"t": 20.0, "caption": "Superposition: one neuron responds to several unrelated features "
                                           "(illustration)."},
                    {"t": 38.0, "caption": "768 → 6,144 features → 768; the loss rebuilds the input and penalises active "
                                           "features."},
                    {"t": 54.0, "caption": "The penalty sets the trade-off: 923 active (not readable), 4 active (gives "
                                           "up), 30 active."}],
        "body": [
            """<p>Episode 43 showed that sparse features can be stored in overlapping directions, so a single neuron
responds to several of them. An SAE tries to undo this. It reads the residual stream after GPT-2's layer 6 (768
numbers per token) and writes it as 6,144 non-negative feature activations, eight times more, then rebuilds the 768
numbers from them:</p>
<p style="text-align:center"><code>f = ReLU(W_enc (x − b_dec) + b_enc)    x̂ = W_dec f + b_dec</code></p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The loss has two parts: rebuild x well (squared error), and keep few features on (an L1 penalty on f,
weighted by each feature's decoder-column length so the model cannot cheat by shrinking f and growing W_dec). The
penalty's weight decides the trade-off:</p>""",
            """<table><tr><th>L1 weight</th><th>active features per token</th><th>variance explained</th></tr>
<tr><td>0.23 (first run)</td><td>923</td><td>95%</td></tr>
<tr><td>2 (used)</td><td>30</td><td>80%</td></tr>
<tr><td>8 (800-step sweep)</td><td>4</td><td>2%</td></tr></table>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>An SAE trades a little reconstruction accuracy for sparsity:
each activation becomes a sum of a few, hopefully meaningful, features.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many weights do W_enc (6,144 × 768) and W_dec (768 × 6,144) hold together, "
                                    "plus the biases b_enc (6,144) and b_dec (768)?",
             "answer": "9,444,096.", "why": "2 × 6,144 × 768 = 9,437,184, plus 6,144 + 768 = 6,912.", "key": {'parts': [{'label': None, 'value': 9444096, 'tol': 0.5, 'unit': None}]}},
            {"kind": "short", "q": "Why is a run with 923 active features per token not useful, even though it explains "
                                   "95% of the variance?",
             "answer": "With hundreds of features on at once, no single feature stands for anything you can read: the "
                       "code is dense again.",
             "why": "Sparsity is what makes features interpretable."},
            {"kind": "tf", "q": "“A stronger L1 penalty always gives better features.”",
             "answer": "False.", "why": "Too strong and the SAE gives up: at weight 8 it kept 4 features on and explained "
                                        "2% of the variance.", "key": {'value': False}},
        ],
    },
    {
        "title": "Is the reconstruction good enough?",
        "segment": (55, 86),
        "figures": [{"t": 67.0, "caption": "Held-out text: 80% variance explained, 30 of 6,144 features active, 0 dead."},
                    {"t": 85.5, "caption": "Spliced into GPT-2: loss 4.69 → 4.90; the layer replaced by its mean: 7.86."}],
        "body": [
            """<p>Trained on 322,580 tokens of GPT-2 activations over Tiny Shakespeare, the SAE explains 79.9% of the
variance on held-out text, with 30 of 6,144 features active per token on average, and every feature fires
somewhere (no <b>dead features</b>).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Variance explained can mislead: the missing 20% might be exactly what the model needs. The stronger test
is to <b>splice</b> the reconstruction into GPT-2 in place of the real activations and let it continue. The next-token
loss goes from 4.686 to 4.903. For scale, replacing the layer with its average activation gives 7.861. So the SAE keeps
(7.861 − 4.903) / (7.861 − 4.686) = 93% of what the layer contributes.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Judge an SAE by what the model does with its reconstruction,
not only by reconstruction error.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "On average, what fraction of the 6,144 features is active on a token?",
             "answer": "About 0.5%.", "why": "30 / 6,144 ≈ 0.0049.", "key": {'parts': [{'label': None, 'value': 0.5, 'tol': 0.05, 'unit': '%'}]}},
            {"kind": "number", "q": "If a different SAE gave a spliced loss of 5.5 (base 4.686, mean-ablated 7.861), "
                                    "what share of the gap would it keep?",
             "answer": "About 74%.", "why": "(7.861 − 5.5) / (7.861 − 4.686) = 2.361 / 3.175 ≈ 0.74.", "key": {'parts': [{'label': None, 'value': 74, 'tol': 1.48, 'unit': '%'}]}},
        ],
    },
    {
        "title": "Reading features",
        "segment": (86, 129),
        "figures": [{"t": 102.0, "caption": "Where random features fire most: “chief”, “Mess(enger):”, learned words."},
                    {"t": 118.0, "caption": "Top-20 activations that are one token: SAE features 49% vs raw dimensions "
                                            "33%; all one token: 14% vs 1%."},
                    {"t": 128.0, "caption": "The SAE in code: encode with a ReLU, decode, L1 weighted by decoder norms."}],
        "body": [
            """<p>To read a feature, list the tokens (with context) where it fires most. Among eight random features with
moderate frequency: feature 4747 fires on “chief” (“three o' the chief”, “valour is the chief”); 2637 on “Mess”, the
start of the speaker name Messenger; 766 on learned words (arithmetic, history, mechanics, calendar, scripture). Others
are less clean, mixing several patterns.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>A rough comparison with the raw dimensions: for each unit, take its top 20 activations and count how many
are the most common token. SAE features: 49% on average, raw dimensions: 33%. Units whose top 20 are all one token:
14% of SAE features, 1% of raw dimensions. Token purity is only one kind of meaning (feature 766 is about a topic, not
a token), but it shows features are more specific than dimensions.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>SAE features are far more specific than raw dimensions, which is
why they are the main tool for reading what large models represent.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Feature 766 fires on arithmetic, history, mechanics, calendar, scripture. Why would "
                                   "the token-purity measure underrate it?",
             "answer": "It responds to a theme across different tokens, so its top 20 are not one token even though it is "
                       "meaningful.",
             "why": "Purity only measures single-token features."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Write the SAE forward pass and loss.",
             "code": """import torch
D, M, L1 = 768, 6144, 2.0
W_enc = torch.randn(M, D) / D ** 0.5; W_dec = W_enc.T.clone()
b_enc, b_dec = torch.zeros(M), torch.zeros(D)
x = torch.randn(16, D)
f = torch.relu((x - b_dec) @ W_enc.T + b_enc)
x_hat = f @ W_dec.T + b_dec
loss = ((x_hat - x) ** 2).sum(1).mean() + L1 * (f * W_dec.norm(dim=0)).sum(1).mean()
print(f.shape, (f > 0).float().sum(1).mean(), loss)""",
             "answer": "f has shape (16, 6144); untrained, about half the features are active (≈ 3,072).",
             "why": "Training with the L1 penalty brings that down to about 30 (step 1 of the episode's run: 3,072)."},
        ],
    },
]
