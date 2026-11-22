"""Study guide content for How Generative Models Work: Images and Sound, episode 9: Flow Matching.

Build:  python framework/study_guide.py generative g09 --video generative/media/videos/g09_scene/1080p60/FlowMatchingVideo.mp4
Every number comes from code/g09_flow_matching (2026-11-22): episode 6's U-Net trained with flow matching on the same
data for the same 20,000 steps, 500 samples from the same starting noise scored by episode 3's judge; a 2D spiral with
diffusion, flow matching and reflow; diffusion with fresh noise at 20 and 50 steps from episode 7.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 9",
    "title": "Flow Matching",
    "tagline": "A straighter road from noise to picture.",
    "duration": "2:11",
    "intro": """<p><b>Flow matching</b> trains the same kind of network as diffusion, but on a simpler path: a
straight line from the picture to the noise. This lesson explains the idea and compares it with diffusion on a 2D
spiral and on Shapes, at the same network size and training budget. It shows why the learned paths still bend, and
how <b>reflow</b> straightens them.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 5–7. Code: <code>code/g09_flow_matching</code>
(PyTorch; the spiral takes about 10 minutes on a CPU, Shapes about 2.5 hours).</div>""",
}

CONCEPTS = [
    {
        "title": "Straight lines and a velocity target",
        "segment": (7.5, 50.1),
        "figures": [{"t": 19.3, "caption": "Five network calls per picture: diffusion with fresh noise (7% recognized) "
                                           "and the same U-Net trained with flow matching (76%)."},
                    {"t": 38.2, "caption": "The training input slides on a straight line from the picture (t = 0) to "
                                           "pure noise (t = 1); the network learns the line's direction."},
                    {"t": 49.8, "caption": "How much of the picture is left: a curved schedule for diffusion, a "
                                           "straight line for flow matching."}],
        "body": [
            "{fig0}",
            """<p>Flow matching (Lipman et al. 2022; the same idea is called <b>rectified flow</b>, Liu et al. 2022)
makes each training input by mixing a picture x₀ and noise ε on a straight line:</p>
<p style="text-align:center">x_t = (1 − t)·x₀ + t·ε,  t between 0 and 1</p>
<p>The network gets x_t and t and predicts the <b>velocity</b> v = ε − x₀, the direction of the line, with mean
squared error. To generate, start from pure noise at t = 1 and take Euler steps against the predicted velocity:
x ← x − Δt·v̂, down to t = 0.</p>""",
            "{fig1}",
            """<p>Diffusion mixes with a curved schedule (episode 5's √ᾱ_t and √(1 − ᾱ_t)), and its network predicts
only the noise. Flow matching's mix changes at a constant rate, and its target is the whole direction of travel. If
the paths it learns were straight, a single long step would follow them exactly.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Flow matching trains on straight lines between noise and
data and predicts the direction of travel; generating means following that direction back from noise.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A picture pixel has value x₀ = 0.6 and its noise is ε = −1.0. What is x_t at "
                                    "t = 0.25, and what velocity should the network predict?",
             "answer": "x_t = 0.2, v = −1.6.", "why": "x_t = 0.75·0.6 + 0.25·(−1.0) = 0.45 − 0.25 = 0.2; "
                                                      "v = ε − x₀ = −1.0 − 0.6 = −1.6.",
             "key": {"parts": [{"label": "x_t", "value": 0.2, "tol": 0.001, "unit": None},
                               {"label": "v", "value": -1.6, "tol": 0.001, "unit": None}]}},
            {"kind": "mc", "q": "To generate, flow matching starts from…",
             "options": ["a training picture at t = 0", "pure noise at t = 1, stepping toward t = 0",
                         "pure noise at t = 0, stepping toward t = 1", "the average picture"],
             "answer": "B.", "why": "t = 1 is pure noise; Euler steps x ← x − Δt·v̂ move it to t = 0.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "Why the paths bend, and reflow",
        "segment": (50.6, 92.9),
        "figures": [{"t": 61.8, "caption": "2,000 generated points (600 shown) on the spiral: diffusion (DDIM) and flow "
                                           "matching with 4 steps, and flow matching with 1 step."},
                    {"t": 77.6, "caption": "Left: training lines from noise to data cross. Right: the paths the flow "
                                           "model actually learned bend."},
                    {"t": 92.6, "caption": "Reflow: paths become straight, and one step lands on the spiral."}],
        "body": [
            """<p>On a 2D spiral (the same small network for each method, 10,000 training steps), we measure
<b>precision</b>, the median distance from a generated point to the data, and <b>coverage</b>, the median distance
from a data point to the generated points (lower is better for both):</p>
<table><tr><th>steps</th><th>diffusion (DDIM)</th><th>flow matching</th><th>reflow</th></tr>
<tr><td>1</td><td>1.831 / 1.363</td><td>0.006 / 1.093</td><td>0.007 / 0.019</td></tr>
<tr><td>2</td><td>1.785 / 1.336</td><td>0.209 / 0.191</td><td>0.007 / 0.017</td></tr>
<tr><td>4</td><td>0.017 / 0.064</td><td>0.011 / 0.029</td><td>0.007 / 0.017</td></tr>
<tr><td>16</td><td>0.007 / 0.020</td><td>0.007 / 0.016</td><td>0.007 / 0.016</td></tr></table>
<p>With 4 steps flow matching is already on the curve. With 1 step it puts all points in one spot near the middle.
The precision looks perfect, because the middle of the spiral is on the spiral, but the coverage shows the
collapse.</p>""",
            "{fig0}",
            """<p>Why? In training, each noise is paired with a <i>random</i> data point, so the straight lines cross.
At a crossing point the network sees many directions for the same input and can only learn their average. The paths
it learns therefore bend: their median straightness (straight-line distance divided by path length) is 0.55. One
Euler step follows the average direction from the start, and that leads every point to the middle.</p>""",
            "{fig1}",
            """<p><b>Reflow</b> (Liu et al. 2022) fixes the pairing. Run the trained flow from 10,000 noises, pair each
noise with the point it lands on, and train a new flow on just those pairs. Those lines do not cross, the learned
paths are straight (median straightness 1.00), and one step gives 0.007 / 0.019, almost as good as 200 steps.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Training lines are straight, but learned paths bend
where training lines cross. Reflow re-pairs noise and data so the lines stop crossing.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Because flow matching trains on straight lines, the paths it learns are always "
                                "straight.”",
             "answer": "False.", "why": "Training lines cross; the network learns their average direction, so the "
                                        "learned paths bend (median straightness 0.55 on the spiral).",
             "key": {"value": False}},
            {"kind": "short", "q": "With one step, flow matching scored precision 0.006 (better than with 200 steps!) "
                                   "but coverage 1.093. What happened?",
             "answer": "Every point landed in one spot near the middle of the spiral. That spot is on the data, so "
                       "precision is perfect, but most of the spiral has no generated point near it.",
             "why": "Precision alone can't detect collapse; coverage can."},
        ],
    },
    {
        "title": "On Shapes, and in practice",
        "segment": (93.4, 124.1),
        "figures": [{"t": 107.2, "caption": "Recognized by the judge at 5, 10, 20 and 50 network calls: flow matching "
                                            "vs diffusion with fresh noise."},
                    {"t": 123.8, "caption": "Flow matching in three lines, and its sampler in one."}],
        "body": [
            """<p>Episode 6's U-Net (1,182,371 parameters), trained with flow matching for the same 20,000 steps on the
same data, against episode 6's diffusion model, 500 samples from the same starting noise:</p>
<table><tr><th>network calls</th><th>5</th><th>10</th><th>20</th><th>50</th></tr>
<tr><td>flow matching (Euler)</td><td>76.2%</td><td>90.6%</td><td>92.2%</td><td>94.0%</td></tr>
<tr><td>diffusion, fresh noise (η = 1)</td><td>6.8%</td><td>85.0%</td><td>95.4%</td><td>95.2%</td></tr>
<tr><td>diffusion, DDIM (η = 0)</td><td>1.6%</td><td>16.4%</td><td>45.8%</td><td>70.0%</td></tr></table>
<p>Flow matching is far ahead at 5 and 10 steps; from 20 steps on, diffusion with fresh noise catches up. Compared
with DDIM, the other deterministic sampler, flow matching wins at every budget.</p>""",
            "{fig0}",
            """<p>Fewer steps for the same training is why recent large image models are trained this way: Stable
Diffusion 3 (Esser et al. 2024) and Flux use rectified flow. Large systems often add tricks on top, such as
distillation and reflow-style straightening, to go down to a handful of steps.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Same network, same data, same training: changing the path
and the target buys good pictures in far fewer steps.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "At 5 network calls, how many times more pictures did flow matching get "
                                    "recognized than diffusion with fresh noise? (76.2% vs 6.8%, one decimal)",
             "answer": "11.2.", "why": "76.2 / 6.8 ≈ 11.2.",
             "key": {"parts": [{"label": None, "value": 11.2, "tol": 0.05, "unit": None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> One flow-matching training step and the sampler.",
             "code": """t = torch.rand(len(x0)).view(-1, 1, 1, 1)
x_t = (1 - t) * x0 + t * eps
loss = ((model(x_t, t * 1000) - (eps - x0)) ** 2).mean()
# generate: t from 1 down to 0 in steps of dt
x = x - dt * model(x, t * 1000)""",
             "answer": "train_flow.py's training loop and sample_flow() run exactly this.",
             "why": "The step input is t·1,000 so the network is exactly episode 6's U-Net."},
        ],
    },
]
