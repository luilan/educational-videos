"""Study guide content for How Generative Models Work: Images and Sound, episode 5: Diffusion I: Adding Noise.

Build:  python framework/study_guide.py generative g05 --video generative/media/videos/g05_scene/1080p60/AddingNoiseVideo.mp4
Every number comes from code/g05_adding_noise (forward_noise.py, 2026-11-21): T = 1,000 steps on 2,000 Shapes
pictures, linear and cosine schedules, and the judge from episode 3.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 5",
    "title": "Diffusion I: Adding Noise",
    "tagline": "The easy half: destroy a picture, one small step at a time",
    "duration": "2:13",
    "intro": """<p><b>Diffusion models</b> generate by learning to undo noise. This lesson covers the easy half, the
<b>forward process</b>. In 1,000 small steps it shrinks a picture and adds Gaussian noise until only pure noise is
left. You will see why any step can be reached in one jump, how the <b>signal-to-noise ratio</b> falls, and why the
choice of <b>noise schedule</b> matters.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–4, and Foundations episodes 5–6
(exponentials and logs, probability and the normal distribution). Code: <code>code/g05_adding_noise</code> (PyTorch,
about a minute on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "One step, a thousand steps",
        "segment": (7.5, 50.6),
        "figures": [{"t": 32.5, "caption": "One step: x_t = √(1 − β_t)·x_{t−1} + √β_t·ε, with β growing linearly from "
                                           "0.0001 to 0.02."},
                    {"t": 50.1, "caption": "A Shapes picture at steps 0, 50, 100, 200, 300, 500, 700 and 1,000 (linear "
                                           "schedule)."}],
        "body": [
            """<p>Pixels are scaled to [−1, 1]. One forward step shrinks the picture slightly and adds slight
Gaussian noise ε (a fresh standard-normal number per pixel):</p>
<p style="text-align:center"><code>x<sub>t</sub> = √(1 − β<sub>t</sub>) · x<sub>t−1</sub> + √β<sub>t</sub> · ε</code></p>
<p>The shrink factor is chosen so the overall variance stays near 1 rather than growing: if x<sub>t−1</sub> has
variance 1, so does x<sub>t</sub>, because (1 − β) + β = 1. In the <b>linear schedule</b> of the original DDPM paper
(2020), β rises from 0.0001 to 0.02 over T = 1,000 steps.</p>""",
            "{fig0}",
            """<p>After 50 steps the picture is grainy, after 300 the square can still be found by eye, and by 700 it
is gone. At step 1,000 the picture's share of the signal, √ᾱ (defined below), is √0.00004 ≈ 0.006. What remains is
almost exactly a sample from a standard normal distribution, and that is where generation will start.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>The forward process turns any picture into (almost) pure
standard-normal noise through many tiny shrink-and-add-noise steps.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With β = 0.02, by what factor does one step shrink the picture, to four "
                                    "decimals?",
             "answer": "About 0.9899.", "why": "√(1 − 0.02) = √0.98 ≈ 0.98995.",
             "key": {"parts": [{"label": None, "value": 0.9899, "tol": 0.0002, "unit": None}]}},
            {"kind": "tf", "q": "“If x<sub>t−1</sub> has variance 1, then x<sub>t</sub> also has variance 1.”",
             "answer": "True.", "why": "(1 − β)·1 + β·1 = 1 (the picture and the new noise are independent).",
             "key": {"value": True}},
        ],
    },
    {
        "title": "Jumping to any step",
        "segment": (51.1, 84.0),
        "figures": [{"t": 70.8, "caption": "ᾱ_t multiplies the shrink factors; 500 small steps and one jump give the same "
                                           "spread (0.9595 vs 0.9596; formula 0.9599)."},
                    {"t": 83.5, "caption": "The signal-to-noise ratio (linear schedule): +40 dB at step 1, 0 dB at step "
                                           "260, −44 dB at step 1,000."}],
        "body": [
            """<p>Multiplying the shrink factors gives ᾱ<sub>t</sub> = (1 − β<sub>1</sub>)(1 − β<sub>2</sub>) …
(1 − β<sub>t</sub>). Because a sum of independent Gaussians is Gaussian, all the small noises add up to one noise,
and you can jump straight to step t:</p>
<p style="text-align:center"><code>x<sub>t</sub> = √ᾱ<sub>t</sub> · x<sub>0</sub> + √(1 − ᾱ<sub>t</sub>) · ε</code></p>
<p>We checked by noising one picture 2,000 times to step 500, once with 500 small steps and once with the jump. The
average per-pixel spread was 0.9595 and 0.9596, and the formula √(1 − ᾱ<sub>500</sub>) predicts 0.9599. Training a
diffusion model uses the jump: pick a random step, noise the picture in one line.</p>""",
            "{fig0}",
            """<p>The <b>signal-to-noise ratio</b> ᾱ/(1 − ᾱ), in decibels (10·log<sub>10</sub>), tracks how much picture is
left. With the linear schedule it starts at +40 dB, crosses 0 dB (signal and noise equally strong) at step 260, and
ends at −44 dB.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>x<sub>t</sub> = √ᾱ<sub>t</sub>·x<sub>0</sub> +
√(1 − ᾱ<sub>t</sub>)·ε: any noise level can be reached in one step, which makes training cheap.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "ᾱ = 0.25 at some step. What is the signal-to-noise ratio in decibels, to one "
                                    "decimal?",
             "answer": "About −4.8 dB.", "why": "0.25 / 0.75 = 1/3; 10·log<sub>10</sub>(1/3) ≈ −4.77.",
             "key": {"parts": [{"label": None, "value": -4.8, "tol": 0.1, "unit": "dB"}]}},
            {"kind": "number", "q": "At the step where signal and noise are equally strong, what is ᾱ?",
             "answer": "0.5.", "why": "ᾱ/(1 − ᾱ) = 1 ⇒ ᾱ = 0.5 (0 dB).",
             "key": {"parts": [{"label": None, "value": 0.5, "tol": 0, "unit": None}]}},
            {"kind": "short", "q": "Why does training use the jump formula rather than running the steps?",
             "answer": "It reaches any step with one random draw instead of t sequential steps, so each training "
                       "example costs the same no matter which step is chosen.",
             "why": "Training picks a random step for every picture."},
        ],
    },
    {
        "title": "Schedules, and why noise helps",
        "segment": (84.5, 127.5),
        "figures": [{"t": 105.8, "caption": "Linear vs cosine: 326 vs 64 steps below −20 dB; the judge reads half the "
                                            "pictures until step 75 vs 175."},
                    {"t": 120.8, "caption": "Generation will run the process backward: undo a little noise at a time."},
                    {"t": 127.0, "caption": "The forward process in three lines."}],
        "body": [
            """<p>The linear schedule wastes steps: 326 of the 1,000 have a signal-to-noise ratio below −20 dB (the
picture is under 1% of the power), so they add little. The <b>cosine schedule</b> (Nichol &amp; Dhariwal, 2021) sets
ᾱ<sub>t</sub> = cos²(((t/T) + 0.008)/1.008 · π/2), which falls slowly at both ends and reaches 0 dB at step 497,
the middle. Only 64 of its steps are below −20 dB. Our judge (episode 3), which was never trained on noise, reads
shape, color and position correctly for half the pictures until step 75 with the linear schedule and until step 175
with the cosine one.</p>""",
            "{fig0}",
            """<p>Why destroy pictures on purpose? Each step is tiny, so <b>undoing one step</b> is an easy problem: the
noisy picture is very close to the slightly-less-noisy one. If a network learns to undo a step at any noise level,
it can start from pure noise and walk back 1,000 small steps to a new picture. That network is the next episode.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>The schedule decides how evenly the picture is destroyed;
diffusion turns one hard problem (make a picture) into many easy ones (remove a little noise).</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What is wrong with a schedule that spends many steps below −20 dB?",
             "options": ["The pictures become too sharp", "Those steps are nearly pure noise and add little",
                         "ᾱ grows above 1", "The jump formula stops working"],
             "answer": "B.", "why": "Below −20 dB the picture is under 1% of the power; those steps barely change it.",
             "key": {"choice": 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Noise a batch of pictures x0 (in [−1, 1]) to random steps.",
             "code": """beta = torch.linspace(1e-4, 0.02, 1000)
alpha_bar = torch.cumprod(1 - beta, dim=0)
t = torch.randint(0, 1000, (len(x0),))
a = alpha_bar[t].view(-1, 1, 1, 1)
eps = torch.randn_like(x0)
x_t = a.sqrt() * x0 + (1 - a).sqrt() * eps""",
             "answer": "This is exactly what diffusion training does before asking the network to find eps.",
             "why": "view(-1, 1, 1, 1) lets each picture use its own step."},
        ],
    },
]
