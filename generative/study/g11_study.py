"""Study guide content for How Generative Models Work: Images and Sound, episode 11: Classifier-Free Guidance.

Build:  python framework/study_guide.py generative g11 --video generative/media/videos/g11_scene/1080p60/GuidanceVideo.mp4
Every number comes from code/g11_guidance (guidance.py, scan_checkpoints.py, when_to_guide.py, figures.py,
2026-11-22): episode 10's conditional model, finished and stopped early, sampled at guidance scales 0 to 8.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 11",
    "title": "Classifier-Free Guidance",
    "tagline": "Ask twice, then push.",
    "duration": "2:25",
    "intro": """<p>Episode 10's label dropout made one network both a conditional and an unconditional model.
<b>Classifier-free guidance</b> uses both at every sampling step and pushes the picture further in the direction
the request points. This lesson measures when that helps (a model that has not learned to listen well yet), when it
hurts (a model that already listens), and why applying it only in the less noisy steps is safer.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 7 (the fresh-noise sampler and the
clean-picture guess) and episode 10 (labels, label dropout). Code: <code>code/g11_guidance</code> (PyTorch; each
guidance sweep takes about 40 minutes on a CPU, the half-trained model about 85 minutes).</div>""",
}

CONCEPTS = [
    {
        "title": "Ask twice",
        "segment": (7.5, 52.2),
        "figures": [{"t": 21.2, "caption": "Episode 10's model stopped after 6,000 of 20,000 training steps: 74% "
                                           "of its pictures follow the request."},
                    {"t": 33.0, "caption": "Two guesses from the same network, shown as the clean picture each "
                                           "implies, at the first sampling step."},
                    {"t": 51.9, "caption": "w = 0 ignores the request, w = 1 is plain conditioning, w > 1 pushes "
                                           "past it."}],
        "body": [
            """<p>Because of label dropout, the network can predict the noise in two ways: with the request
(<i>conditional</i>) and with every label set to “not said” (<i>unconditional</i>). Their difference is exactly
what the request changes in the guess.</p>""",
            "{fig0}",
            "{fig1}",
            """<p>Classifier-free guidance (Ho and Salimans, 2022) starts from the unconditional guess and adds the
difference times a <b>guidance scale</b> <i>w</i>:</p>
<p style="text-align:center"><code>ε̂ = ε̂(nothing said) + w · (ε̂(request) − ε̂(nothing said))</code></p>
<p>At <i>w</i> = 1 the unconditional terms cancel and this is plain conditioning. Above 1, the guess moves past
the conditional one, exaggerating whatever the request changes. The sampler is otherwise unchanged. The two guesses
go through the network as one batch of twice the size, so each step costs two network evaluations. (The paper
writes (1 + <i>w′</i>)·conditional − <i>w′</i>·unconditional, so its <i>w′</i> is our <i>w</i> − 1.)</p>
<p>It is called “classifier-free” because an earlier method, classifier guidance (Dhariwal and Nichol, 2021), pushed
in the direction suggested by a separate classifier trained on noisy pictures. Here the network itself supplies
the direction.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Guidance compares the network's guess with and without
the request, and scales up the difference. It needs no retraining, only a model trained with label dropout.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "At one step, the unconditional noise guess for a pixel is 0.20 and the "
                                    "conditional guess is 0.30. What is the guided guess at w = 3?",
             "answer": "0.50.", "why": "0.20 + 3 × (0.30 − 0.20) = 0.50.",
             "key": {"parts": [{"label": None, "value": 0.5, "tol": 0.001, "unit": ""}]}},
            {"kind": "mc", "q": "What does w = 1 give?",
             "options": ["An unconditional model", "Plain conditioning, as in episode 10",
                         "Twice the effect of the request", "Nothing: the guess is zero"],
             "answer": "B.", "why": "unc + 1·(cond − unc) = cond.", "key": {"choice": 1}},
        ],
    },
    {
        "title": "When it helps, and when it hurts",
        "segment": (52.7, 87.9),
        "figures": [{"t": 68.0, "caption": "Half-trained model, same starting noise: w = 1 (top) and w = 1.5 "
                                           "(bottom). Requests it got wrong, now right."},
                    {"t": 76.7, "caption": "Half-trained model at w = 3: shapes in the wrong place, smeared "
                                           "pictures."},
                    {"t": 87.6, "caption": "Share of pictures exactly as asked, against the guidance scale."}],
        "body": [
            """<p>Each sweep samples all 127 trained combinations twice and the eight never-seen combinations 32
times, from the same starting noise at every <i>w</i>, and episode 3's judge reads every picture.</p>
<table><tr><th>w</th><th>half-trained (6,000 steps)</th><th>never-seen</th><th>finished (20,000 steps)</th><th>never-seen</th></tr>
<tr><td>0</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr>
<tr><td>1</td><td>73.6%</td><td>67.2%</td><td><b>96.9%</b></td><td>96.5%</td></tr>
<tr><td>1.5</td><td><b>84.6%</b></td><td><b>84.8%</b></td><td>96.5%</td><td>95.7%</td></tr>
<tr><td>2</td><td>76.4%</td><td>77.3%</td><td>58.7%</td><td>55.9%</td></tr>
<tr><td>3</td><td>9.8%</td><td>11.7%</td><td>3.9%</td><td>1.6%</td></tr>
<tr><td>5</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr></table>""",
            "{fig0}",
            """<p>On the half-trained model, <i>w</i> = 1.5 turns 31 of 254 wrong pictures right and 3 right ones
wrong. Most of the gain is in position (89.8% → 96.1% agreement) and in recognizable pictures (81.1% → 87.4%).
The finished model already gets 97% right, so there is nothing left to fix, and every step above 1.5 costs it.</p>""",
            "{fig1}",
            """<p>Checkpoints of the same training run show the pattern (<code>scan_checkpoints.py</code>, 127
pictures each, w = 1 → w = 2): 4,000 steps 20.5% → 34.6%; 6,000 steps 70.1% → 77.2%; 8,000 steps
87.4% → 80.3%; 20,000 steps 96.1% → 61.4%. At 2,000 steps nothing is recognizable at any <i>w</i>: guidance can
sharpen a model's preference, not supply a skill it lacks.</p>""",
            "{fig2}",
            """<p>Mild guidance also makes pictures of one request slightly more alike: the mean squared pixel
difference between two pictures of the same request falls from 0.0113 to 0.0099 on the half-trained model (w = 1
→ 1.5). Ho and Salimans describe the same trade on ImageNet: as <i>w</i> grows, samples match their class more
closely and vary less.</p>
<div class="box key"><b class="t">Key idea</b>Guidance helps a model that only partly follows requests, and hurts
one that already follows them. Too large a scale breaks every model.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Classifier-free guidance always makes a model follow requests more often.”",
             "answer": "False.", "why": "On the finished model, w = 2 drops it from 96.9% to 58.7%.",
             "key": {"value": False}},
            {"kind": "short", "q": "Why does guidance do nothing for the 2,000-step model?",
             "answer": "It draws nothing recognizable at all, so there is no picture for guidance to push toward "
                       "the request.",
             "why": "Guidance scales up the difference between two guesses; it cannot add skill the network lacks."},
        ],
    },
    {
        "title": "Guide late",
        "segment": (88.4, 137.3),
        "figures": [{"t": 102.7, "caption": "One run of “red circle, center” at w = 3: the clean-picture guesses "
                                            "at five steps. The guided one is saturated early and ends in a "
                                            "triangle."},
                    {"t": 114.4, "caption": "Finished model, same requests and noise: w = 1, 2, 3, and 3 guided "
                                            "only while t < 500."},
                    {"t": 137.0, "caption": "Classifier-free guidance in code."}],
        "body": [
            """<p>Where does large guidance do its damage? <code>when_to_guide.py</code> applies <i>w</i> = 3 to the
finished model in different ranges of steps (127 pictures, plain conditioning elsewhere):</p>
<table><tr><th>guided at w = 3</th><th>follows</th><th>shape</th><th>color</th><th>position</th></tr>
<tr><td>never (w = 1)</td><td>96.1%</td><td>99.2%</td><td>100%</td><td>99.2%</td></tr>
<tr><td>every step</td><td>2.4%</td><td>27.6%</td><td>89.8%</td><td>15.0%</td></tr>
<tr><td>only t ≥ 500 (noisy half)</td><td>2.4%</td><td>20.5%</td><td>59.8%</td><td>15.0%</td></tr>
<tr><td>only 100 ≤ t &lt; 700</td><td>36.2%</td><td>40.9%</td><td>98.4%</td><td>97.6%</td></tr>
<tr><td>only t &lt; 500 (quiet half)</td><td>84.3%</td><td>87.4%</td><td>100%</td><td>99.2%</td></tr></table>
<p>All of the damage comes from the noisy half. There, both guesses are vague, and exaggerating their difference
makes the implied clean picture fall far outside the pixel range: at t = 857, 10.1% of its pixels must be clipped
at <i>w</i> = 3, against 0.95% at <i>w</i> = 1 (t = 715: 3.1% vs 0.06%). The sampler commits to a layout in these
steps and cannot undo it later; position is the first thing to go.</p>""",
            "{fig0}",
            """<p>So apply guidance only in the second half of sampling (t &lt; 500), once the layout is settled,
and plain conditioning before. On the finished model: <i>w</i> = 2 goes from 58.7% to 95.7%, <i>w</i> = 3 from
3.9% to 78.0%. On the half-trained model the late-only version keeps most of the benefit at moderate scales (74.8%
at <i>w</i> = 2) and survives large ones (72.0% at <i>w</i> = 3 instead of 9.8%).</p>""",
            "{fig1}",
            """<p>Real text-to-image models use much larger scales: Stable Diffusion's default is 7.5 (same
convention as our <i>w</i>). Kynkäänniemi et al. (2024, “Applying guidance in a limited interval”) found that
guidance in the noisiest steps is harmful for large image models too, and limiting it to a middle range of noise
levels improved their results.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Large guidance breaks pictures in the noisiest steps,
where the layout is decided. Guiding only the less noisy steps keeps the benefit and removes most of the damage.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Sampling takes 50 steps. With guidance applied only while t &lt; 500 (about "
                                    "half the steps), how many network evaluations does one picture need?",
             "answer": "About 75.", "why": "25 steps × 1 + 25 steps × 2 = 75 (plain conditioning needs only one "
                                           "evaluation per step).",
             "key": {"parts": [{"label": None, "value": 75, "tol": 2, "unit": ""}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Guidance in a sampler.",
             "code": """both = model(cat([x, x]), cat([t, t]), cat([y, NOTHING_SAID]))
cond, uncond = both.chunk(2)
w_now = w if t < 500 else 1.0
eps = uncond + w_now * (cond - uncond)""",
             "answer": "guidance.py's sample() with late_only=True does this; diffusion.predict_eps holds the mix.",
             "why": "Change the 500 to find where guidance stops hurting your model."},
        ],
    },
]
