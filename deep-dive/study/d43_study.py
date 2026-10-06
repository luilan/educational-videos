"""Study guide content for How LLMs Work: Deep Dive, episode 43: Superposition.

Build:  python framework/study_guide.py deep-dive d43 --video deep-dive/media/videos/d43_scene/1080p60/SuperpositionVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d43_superposition/superposition.py (the toy model of Elhage et al., 2022, trained with Adam,
seeded; a feature counts as represented when its direction has norm > 0.5) or from the arithmetic shown.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 43",
    "title": "Superposition",
    "tagline": "More features than dimensions",
    "duration": "2:07",
    "intro": """<p>This lesson answers one question: why do single neurons in language models respond to many unrelated
things? A toy model gives the answer: when features are <b>sparse</b> (rarely active), a network can store more of them
than it has dimensions, in overlapping directions: <b>superposition</b>. Five features squeezed into two dimensions go from
two (dense) to all five (97% sparse, a pentagon); a hundred features in twenty dimensions go from 14 to all 100. The price
is neurons that are <b>polysemantic</b>.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 15 (activations) and 13 (the residual
stream). Code: <code>code/d43_superposition</code> (PyTorch only, a few minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "The toy model",
        "segment": (8, 47),
        "figures": [{"t": 36.2, "caption": "5 features → 2 numbers → ReLU(Wᵀh + b) → 5 features; feature i has importance "
                                           "0.8^i."},
                    {"t": 45.9, "caption": "Dense features: the two most important get perpendicular directions; three are "
                                           "dropped."}],
        "body": [
            """<p>The toy model compresses n features into m &lt; n numbers, h = Wx, and tries to read them back:
x' = ReLU(Wᵀh + b). Each column of W is a feature's direction in the small space. Each feature is zero with probability S
(the <b>sparsity</b>), otherwise uniform in [0, 1]; the loss weights feature i by its importance, here 0.8^i.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>With dense features (S = 0) and 5 features in 2 dimensions, the model keeps the two most important, with
directions of length 1.00 at right angles, and drops the other three (lengths 0.00–0.02): exactly what PCA would do.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Without sparsity, m dimensions hold m features; the rest are
lost.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With S = 0.97 and 5 features, how many features are active in an average example?",
             "answer": "0.15.", "why": "5 × (1 − 0.97) = 0.15: most examples have no active feature at all.", "key": {'parts': [{'label': None, 'value': 0.15, 'tol': 0.005, 'unit': None}]}},
        ],
    },
    {
        "title": "Sparsity enables superposition",
        "segment": (47, 95),
        "figures": [{"t": 61.2, "caption": "Directions of the 5 features: dense (2), sparsity 0.7 (4), sparsity 0.97 (all 5, "
                                           "about 72° apart)."},
                    {"t": 75.3, "caption": "Overlapping directions cause interference; rare co-activation plus ReLU and a "
                                           "negative bias keep it small."},
                    {"t": 93.6, "caption": "100 equally important features in 20 dimensions: 14, 40, 58, 95, 100, 100 represented "
                                           "at sparsity 0, 0.5, 0.8, 0.9, 0.95, 0.99."}],
        "body": [
            """<p>Make the features rare and the model packs more of them: 4 features at S = 0.7, all 5 at S = 0.97,
spread about 72° apart (angles −149°, −81°, −8°, 67°, 140°): a pentagon. The directions overlap, so reading one feature
picks up a bit of the others; but if features are rarely active together, that interference is rare, and the ReLU with
a negative bias filters out the small leftovers.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<table><tr><th>sparsity</th><th>features represented (of 100, in 20 dims)</th><th>signed cosine with closest
other feature</th></tr>
<tr><td>0</td><td>14</td><td>−0.18</td></tr><tr><td>0.5</td><td>40</td><td>−1.00 (opposite pairs)</td></tr>
<tr><td>0.8</td><td>58</td><td>−0.80</td></tr><tr><td>0.9</td><td>95</td><td>−0.50</td></tr>
<tr><td>0.95</td><td>100</td><td>−0.48</td></tr><tr><td>0.99</td><td>100</td><td>−0.49</td></tr></table>""",
            """<p>At sparsity 0.5 the model uses <b>antipodal pairs</b>: two features share one direction with opposite
signs, so the ReLU separates them. From 0.95 on, all 100 features fit: five per dimension.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>The sparser the features, the more of them fit in
superposition, at the cost of interference.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Five unit vectors evenly spaced in a plane: what is the angle between neighbours, and "
                                    "their cosine?",
             "answer": "72°; cos 72° ≈ 0.31.", "why": "360° / 5 = 72°.", "key": {'parts': [{'label': 'angle (degrees)', 'value': 72, 'tol': 0.5, 'unit': None}, {'label': 'cosine', 'value': 0.31, 'tol': 0.0062, 'unit': None}]}},
            {"kind": "short", "q": "Why can two features share one direction with opposite signs (cosine −1)?",
             "answer": "Each is non-negative; after the ReLU, a positive reading means one feature and a negative reading "
                       "the other. It only fails when both are active at once, which is rare when features are sparse.",
             "why": "Antipodal pairs are the cheapest superposition: no interference unless both fire."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Train the 5 → 2 model at S = 0.97 and print the angles.",
             "code": """import torch
torch.manual_seed(0)
W = torch.nn.Parameter(torch.randn(2, 5) * 0.1); b = torch.nn.Parameter(torch.zeros(5))
imp = 0.8 ** torch.arange(5.); opt = torch.optim.Adam([W, b], lr=1e-2)
for _ in range(6000):
    x = torch.rand(1024, 5) * (torch.rand(1024, 5) > 0.97)
    loss = (imp * (torch.relu(x @ W.T @ W + b) - x) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(W.norm(dim=0), torch.atan2(W[1], W[0]) * 180 / torch.pi)""",
             "answer": "Five norms near 1.1 and angles about 72° apart.", "why": "The same settings as part 1 of the "
                                                                                "episode's code."},
        ],
    },
    {
        "title": "What it means for real models",
        "segment": (95, 119),
        "figures": [{"t": 108.6, "caption": "Many sparse features share few neurons: each neuron becomes polysemantic."}],
        "body": [
            """<p>Real features (a topic, a language, a grammar rule, a named entity) are sparse: each is active on a
small fraction of tokens. So large models likely store many more features than they have neurons, and each neuron ends
up responding to several unrelated features: it is <b>polysemantic</b>. To read the features we need to unpack them,
which is what sparse autoencoders do (episode 44).</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Polysemantic neurons are a natural consequence of compressing
many sparse features into few dimensions.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“If a neuron responds to two unrelated concepts, the model must be poorly trained.”",
             "answer": "False.", "why": "Superposition is an efficient solution when features are sparse.", "key": {'value': False}},
        ],
    },
]
