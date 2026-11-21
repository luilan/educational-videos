"""Study guide content for How Generative Models Work: Images and Sound, episode 6: Diffusion II: Learning to Denoise.

Build:  python framework/study_guide.py generative g06 --video generative/media/videos/g06_scene/1080p60/LearningToDenoiseVideo.mp4
Every number comes from code/g06_learning_to_denoise (train_diffusion.py, 2026-11-21): the 2D spiral run (6,000
steps) and the Shapes U-Net run (20,000 steps), scored by episode 3's judge.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 6",
    "title": "Diffusion II: Learning to Denoise",
    "tagline": "Train a network to spot the noise, then walk back from pure noise",
    "duration": "2:40",
    "intro": """<p>Episode 5 destroyed pictures with noise. This lesson trains a network to undo it. The network is
not asked for the clean picture but for the <b>noise</b> that was added, with a plain squared-error loss. We train
one on 2D points first, then a <b>U-Net</b> on Shapes. Then we generate by starting from pure noise and removing the
predicted noise a little at a time. The result beats the VAE and the GAN on every measure so far, at the cost of
many network calls per sample.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–5 (episode 5's jump formula is used
here). Code: <code>code/g06_learning_to_denoise</code> (PyTorch; the 2D run takes about two minutes, the Shapes run
about two hours on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Predict the noise",
        "segment": (7.5, 54.5),
        "figures": [{"t": 38.0, "caption": "One training step: a clean picture, a random step t, the jump to x_t, and "
                                           "the network's guess of the noise ε that was added."},
                    {"t": 54.3, "caption": "On a 2D spiral, the loss falls from 0.55 to 0.39 in 6,000 steps and then "
                                           "levels off."}],
        "body": [
            """<p>One training step: take a clean example x₀, pick a random step t between 1 and 1,000, draw noise ε,
and jump straight to x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε (episode 5). Give the network x_t and t, and train it with the
mean squared error between its output ε̂ and ε. That is the whole recipe of a <b>denoising diffusion</b> model
(DDPM, Ho et al. 2020). Predicting ε rather than x₀ keeps the target the same size at every step, since ε is always
standard normal.</p>""",
            "{fig0}",
            """<p>A tiny network (three hidden layers of 256) trained on 10,000 points of a spiral brings the loss
from 0.55 to 0.39 and then levels off. It cannot reach zero. At small t the added noise is smaller than the data's
own jitter, so part of ε is genuinely unpredictable. The network learns the best average guess.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>A diffusion model is trained by plain regression: noise a
clean example to a random step and ask the network for the noise.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What is the network trained to output?",
             "options": ["The clean picture x₀", "The noise ε that was added", "The step number t",
                         "A real/fake score"],
             "answer": "B.", "why": "It predicts ε; the loss is the squared error between ε̂ and ε.",
             "key": {"choice": 1}},
            {"kind": "tf", "q": "“Training needs to run all the noising steps from 1 to t for each example.”",
             "answer": "False.", "why": "Episode 5's jump reaches any step in one go.", "key": {"value": False}},
        ],
    },
    {
        "title": "Walking back from noise",
        "segment": (55.0, 111.0),
        "figures": [{"t": 76.4, "caption": "2,000 points of pure noise become the spiral after 1,000 small steps back."},
                    {"t": 88.2, "caption": "The Shapes denoiser: a U-Net of 1,182,371 parameters, trained for 20,000 "
                                           "steps (2 h 18 min on a CPU)."},
                    {"t": 110.8, "caption": "At t = 50 to 900: the noisy picture, the true noise, the network's "
                                            "guess, and the picture minus the guess. At t = 900 the one-shot answer "
                                            "is a dim blur."}],
        "body": [
            """<p>To generate, start from pure noise at t = 1,000. At each step, ask the network for the noise, remove
a little of it, and add back a smaller amount of fresh noise. Then go to t − 1. After 1,000 steps the 2D cloud has
become the spiral. The median generated point is 0.007 from the nearest real point, against 0.19 for pure noise.</p>""",
            "{fig0}",
            """<p>For 32×32 pictures the denoiser is a <b>U-Net</b>: it shrinks the picture to 8×8 and grows it back,
with skip connections across, and it also gets t as an input (episode 8 opens it up). Ours has 1,182,371 parameters
and was trained for 20,000 steps of 128 pictures.</p>""",
            "{fig1}",
            """<p>Why walk in small steps? Removing the guessed noise in one go recovers the triangle almost perfectly
up to t = 600. At t = 900 too little of the picture is left, and the best guess is a dim blur, an <b>average of
every picture</b> it might have been. It is the same kind of averaging that made the VAE's samples blurry. Small steps let the model commit to one
picture gradually instead of averaging over all of them.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Generation is many small denoising steps from pure noise;
each step only has to be roughly right, and the fresh noise keeps samples varied.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why is the one-shot clean guess at t = 900 blurry?",
             "answer": "So little of the picture is left that many clean pictures fit; the squared-error-optimal "
                       "guess is their average, which is a blur.",
             "why": "The same reason a VAE trained on pixel error is blurry."},
            {"kind": "number", "q": "In the 2D run, generated points were a median 0.007 from the spiral and pure "
                                    "noise 0.19. About how many times closer are the generated points (whole number)?",
             "answer": "About 27.", "why": "0.19 / 0.007 ≈ 27.",
             "key": {"parts": [{"label": None, "value": 27, "tol": 1, "unit": None}]}},
        ],
    },
    {
        "title": "Results, and the price",
        "segment": (111.5, 153.8),
        "figures": [{"t": 125.0, "caption": "Samples after 2,000, 5,000, 10,000 and 20,000 training steps; the judge "
                                            "recognizes 0%, 0%, 15% and 70%."},
                    {"t": 146.7, "caption": "Diffusion against the earlier models: 70% recognized (GAN 38.4%, VAE "
                                            "9.4%), 127 of 135 combinations, nearest training picture 0.0015."},
                    {"t": 153.6, "caption": "The training step in four lines."}],
        "body": [
            """<p>Quality grows late. After 2,000 steps the samples are smudges. After 5,000 they are colored blobs,
and the judge recognizes none of them. After 10,000 it recognizes 15.2%, and after 20,000 70.0% (real pictures:
99.95%). The training loss kept falling (0.0043 at 10,000 steps, 0.0028 at 20,000), so longer training would likely
help further.</p>""",
            "{fig0}",
            """<p>Against the earlier models, scored the same way on 500 samples: 70.0% recognized, against 38.4% for
the GAN and 9.4% for the VAE. The recognized samples cover 127 of the 135 shape-color-position combinations (GAN 97),
and they are the closest to real pictures so far: 0.0015 from the nearest training picture (GAN 0.0038). The price
is speed. Each sample here took 50 network calls (a DDIM sampler, episode 7); a GAN needs one.</p>""",
            "{fig1}",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Diffusion gives both quality and coverage from a simple
loss; what it costs is many network calls per sample.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many of the 135 combinations did the diffusion model miss?",
             "answer": "8.", "why": "135 − 127 = 8.",
             "key": {"parts": [{"label": None, "value": 8, "tol": 0, "unit": None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> One diffusion training step.",
             "code": """t = torch.randint(1, 1001, (len(x0),))
eps = torch.randn_like(x0)
x_t = ab[t-1].sqrt().view(-1,1,1,1) * x0 + (1 - ab[t-1]).sqrt().view(-1,1,1,1) * eps
loss = F.mse_loss(model(x_t, t), eps)""",
             "answer": "diffusion.py's training_loss does exactly this.",
             "why": "ab is the cumulative product ᾱ from episode 5 (here the cosine schedule)."},
        ],
    },
]
