"""Study guide content for How Generative Models Work: Images and Sound, episode 7: Sampling: From Noise to Picture.

Build:  python framework/study_guide.py generative g07 --video generative/media/videos/g07_scene/1080p60/SamplingVideo.mp4
Every number comes from code/g07_sampling (sample_compare.py, 2026-11-22): episode 6's U-Net, 500 samples per
setting from the same starting noise, scored by episode 3's judge.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 7",
    "title": "Sampling: From Noise to Picture",
    "tagline": "How many steps does it take, and does the fresh noise matter?",
    "duration": "2:43",
    "intro": """<p>A trained denoiser does not make pictures by itself; a <b>sampler</b> walks it from pure noise to a
picture. This lesson takes episode 6's network unchanged and compares samplers. It covers <b>DDIM</b>, which adds no
new randomness, and <b>DDPM</b>-style sampling, which mixes in fresh noise at every step, at 10 to 1,000 steps. It
also watches one sample form and measures when its shape, color and position are decided.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 5 and 6. Code:
<code>code/g07_sampling</code> (PyTorch; about an hour on a CPU; uses episode 6's trained weights).</div>""",
}

CONCEPTS = [
    {
        "title": "One move, two kinds of noise",
        "segment": (7.5, 76.9),
        "figures": [{"t": 45.3, "caption": "Every sampler repeats one move: predict the noise, guess the clean picture, "
                                           "noise the guess back down to a lower level."},
                    {"t": 58.5, "caption": "One sample: x_t (top) and the clean guess (bottom). At t = 800 the guess is "
                                           "two faint blobs; by t = 600 it is a red circle."},
                    {"t": 76.6, "caption": "How often the clean guess already shows the final position, color and "
                                           "shape (480 samples): position first, shape last."}],
        "body": [
            """<p>From step t to a lower step s, every sampler does the same thing. It asks the network for the noise
ε̂, guesses the clean picture x̂₀ = (x_t − √(1 − ᾱ_t)·ε̂)/√ᾱ_t, and noises that guess back down to level s:</p>
<p style="text-align:center">x_s = √ᾱ_s·x̂₀ + √(1 − ᾱ_s − σ²)·ε̂ + σ·z</p>
<p>The only choice is σ, the share of <b>fresh</b> random noise z. With σ = 0 (<b>DDIM</b>, Song et al. 2020) the
walk is deterministic: the starting noise fixes the picture. With σ set to its maximum, the DDPM posterior noise
(η = 1), it mixes in fresh noise every step; with all 1,000 steps this is exactly DDPM's sampler. Both can skip
steps.</p>""",
            "{fig0}",
            """<p>Watching the clean guess shows how a picture forms. Early on the guess is an average of many possible
pictures. At t = 800 our example is two faint blobs, undecided. By t = 600 it is a red circle, and the remaining
steps only sharpen it.</p>""",
            "{fig1}",
            """<p>Across 480 samples the judge can read, decisions come <b>coarse to fine</b>. At t = 800 the clean
guess already has the final position 88% of the time, the color 73% and the shape 52%. By t = 500, 100%, 99% and
95%.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>A sampler is a loop of “guess the clean picture, then
re-noise it to a lower level”; samplers differ only in how much fresh noise they add.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What makes DDIM deterministic?",
             "options": ["It uses fewer steps", "It adds no fresh random noise (σ = 0)",
                         "It clips the clean guess", "It uses a different network"],
             "answer": "B.", "why": "With σ = 0 the only randomness is the starting noise.", "key": {"choice": 1}},
            {"kind": "tf", "q": "“In our model, the shape of a sample is decided before its position.”",
             "answer": "False.", "why": "At t = 800 the position matches the final one 88% of the time, the shape "
                                        "52%; position is decided first.", "key": {"value": False}},
        ],
    },
    {
        "title": "How many steps, and why fresh noise helped",
        "segment": (77.4, 116.2),
        "figures": [{"t": 94.2, "caption": "Recognized by the judge vs number of steps, with fresh noise (η = 1) and "
                                           "without (DDIM, η = 0)."},
                    {"t": 115.9, "caption": "1,000 steps without fresh noise: small bumps and smears pile up. With fresh "
                                            "noise: clean shapes."}],
        "body": [
            """<p>Results on 500 samples from the same starting noise:</p>
<table><tr><th>steps</th><th>10</th><th>20</th><th>50</th><th>100</th><th>250</th><th>1,000</th></tr>
<tr><td>with fresh noise (η = 1)</td><td>85.0%</td><td>95.4%</td><td>95.2%</td><td>98.4%</td><td>95.2%</td><td>96.0%</td></tr>
<tr><td>DDIM (η = 0)</td><td>16.4%</td><td>45.8%</td><td>70.0%</td><td>75.4%</td><td>69.4%</td><td>19.2%</td></tr></table>
<p>Episode 6's 70% came from DDIM with 50 steps. The same network with fresh noise reaches 95.2% at 50 steps and
already 85.0% at 10. Coverage follows: 132 of 135 combinations at 20 steps with fresh noise.</p>""",
            "{fig0}",
            """<p>Why would DDIM get <i>worse</i> with more steps? Our network is small and was still improving when
training stopped, so every noise prediction is slightly off. Without fresh noise nothing washes those errors out.
Over 1,000 steps they accumulate into bumps and smears the judge rejects. Fresh noise blurs away part of each
step's error, so the next step can correct it. This is a property of our small model: in large, well-trained
models DDIM is usually the better choice at few steps.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>The sampler matters as much as the network: the same
trained model went from 70% to 95% by changing only how it walks back.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With fresh noise, 20 steps gave 95.4% and 1,000 steps 96.0%. How many times more "
                                    "network calls did the 1,000-step run use?",
             "answer": "50.", "why": "1,000 / 20 = 50, for 0.6 percentage points.",
             "key": {"parts": [{"label": None, "value": 50, "tol": 0, "unit": None}]}},
            {"kind": "short", "q": "Give one reason fresh noise can make a sampler more robust to an imperfect network.",
             "answer": "Each step's error gets partly drowned in new noise, so later steps can correct it instead of "
                       "carrying it all the way to the end.",
             "why": "In our run DDIM at 1,000 steps fell to 19.2%, while the stochastic sampler stayed at 96.0%."},
        ],
    },
    {
        "title": "Same noise, and the cost",
        "segment": (116.7, 154.3),
        "figures": [{"t": 134.9, "caption": "8 starting noises (columns). DDIM at 10, 20 and 50 steps keeps colors and "
                                            "places; two DDPM runs from the same noise give different pictures."},
                    {"t": 149.3, "caption": "Time per picture on an 8-core CPU (batches of 500) and the share "
                                            "recognized, with fresh noise."},
                    {"t": 154.0, "caption": "The whole sampler step in four lines."}],
        "body": [
            """<p>DDIM has something DDPM lacks: the starting noise works like an <b>address</b>. With 10, 20 or 50
steps the same noise gives the same colors in the same places and mostly the same shapes. That makes results
reproducible, and it is what image-editing methods build on. Two DDPM runs from the same noise gave different
readings for all 8 pictures. One caution: our DDIM at 1,000 steps drifted to different pictures (1 of 8 read the
same as at 50 steps), because of the accumulated errors above.</p>""",
            "{fig0}",
            """<p>The cost is one network call per step: on our CPU, about 20 ms per picture for 10 steps and 2.0 s for
1,000. For this model, 20 steps with fresh noise is the sweet spot. Large image models typically sample in 20–50
steps; the next ideas for going faster are straighter paths (flow matching, episode 9) and distillation.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>Steps cost time linearly; choose the sampler and the
step count together, and measure, because the best setting depends on the model.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Two DDPM runs from the same starting noise give the same picture.”",
             "answer": "False.", "why": "DDPM adds fresh random noise at each step; all 8 of our pairs differed.",
             "key": {"value": False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> One sampler step from t to s (η = 0 for DDIM, 1 for DDPM).",
             "code": """eps = model(x, t)
x0 = ((x - (1-a_t).sqrt()*eps) / a_t.sqrt()).clamp(-1, 1)
sigma = eta * ((1-a_s)/(1-a_t) * (1 - a_t/a_s)).sqrt()
x = a_s.sqrt()*x0 + (1-a_s-sigma**2).sqrt()*eps + sigma*torch.randn_like(x)""",
             "answer": "sample_compare.py's sample() runs this for every step; a_t and a_s are ᾱ at the two steps.",
             "why": "Set eta = 0 and the last term vanishes: the step is deterministic."},
        ],
    },
]
