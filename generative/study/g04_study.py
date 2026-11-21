"""Study guide content for How Generative Models Work: Images and Sound, episode 4: GANs: a Forger and a Judge.

Build:  python framework/study_guide.py generative g04 --video generative/media/videos/g04_scene/1080p60/GANVideo.mp4
Every number comes from code/g04_gans (train_gan.py, 2026-11-21): the "stable" run (30 epochs) and the "collapse"
run (15 epochs), scored by episode 3's judge.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 4",
    "title": "GANs: a Forger and a Judge",
    "tagline": "Let a second network teach the generator to be sharp",
    "duration": "2:38",
    "intro": """<p>A <b>generative adversarial network</b> (GAN) replaces the pixel error with a second network. A
<b>generator</b> makes pictures from random numbers, and a <b>discriminator</b> learns to tell them from real ones.
Each improves because of the other. This lesson trains one on Shapes and measures what that buys: sharp pictures and
better scores than the VAE. It also shows what it costs: unstable training and <b>mode collapse</b>.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–3 (the judge comes from episode 3).
Code: <code>code/g04_gans</code> (PyTorch; about an hour per run on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Two networks, one game",
        "segment": (7.5, 56.4),
        "figures": [{"t": 40.2, "caption": "The generator turns 64 random numbers into a picture; the discriminator "
                                           "outputs one number, real or fake."},
                    {"t": 55.9, "caption": "Training alternates a discriminator step and a generator step; neither has "
                                           "a fixed target."}],
        "body": [
            """<p>A VAE is trained so every pixel is close on average, and averages are blurry. A GAN has no pixel
error. The <b>generator</b> G maps random numbers z (here 64 of them) to a picture. The <b>discriminator</b> D maps
a picture to one number, a logit for “real”. D is trained with cross-entropy to output 1 for real pictures and 0
for G's pictures. G is trained to make D output 1 for its pictures. A forger and a detective improve together.
Ours have 663,043 (G) and 662,977 (D) parameters.</p>""",
            "{fig0}",
            """<p>Each training step has two halves: update D on a batch of real and a batch of fake pictures, then
update G through D. Because each network's loss depends on the other, neither has a fixed target. This is a
<b>game</b>, not a plain minimization, and that is the root of GAN instability.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>A GAN learns from a learned judge instead of a pixel
error: the generator improves by fooling a discriminator that keeps improving too.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What does the discriminator output?",
             "options": ["A picture", "64 random numbers", "One number: how real the picture looks",
                         "The pixel error"],
             "answer": "C.", "why": "A single logit, turned into a probability by a sigmoid.", "key": {"choice": 2}},
            {"kind": "tf", "q": "“The generator ever sees real pictures directly.”",
             "answer": "False.", "why": "It only learns through the discriminator's scores on its own pictures.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "Making it work",
        "segment": (56.9, 105.1),
        "figures": [{"t": 76.7, "caption": "Early in training nearly every shape is blue; the two fixes: an R1 penalty "
                                           "and averaged (EMA) generator weights."},
                    {"t": 93.3, "caption": "After 30 passes: the judge recognizes 38.4% (VAE 9.4%); 97 of 135 "
                                           "combinations; nearest training picture 0.0038."},
                    {"t": 104.6, "caption": "The generator itself jumps from pass to pass; the averaged weights climb "
                                            "steadily."}],
        "body": [
            """<p>Our first GANs (the classic DCGAN recipe) gave noisy blobs, collapsed colors, and one crashed in
its last pass. Two standard fixes helped. The <b>R1 penalty</b> adds ½·|∇D(real)|² to D's loss so D stays smooth
around real pictures. Generating from an <b>exponential moving average</b> (EMA) of G's weights (decay 0.999) smooths
out the oscillations. Our generator also upsamples and then convolves, rather than using transposed convolutions,
which avoids checkerboard artifacts. 30 passes over 20,000 pictures took 67 minutes on a CPU.</p>""",
            "{fig0}",
            """<p>The results are sharp. The judge recognizes 38.4% of 500 samples (real pictures 99.95%, VAE 9.4%).
The recognized samples cover 97 of the 135 shape-color-position combinations, and samples are 0.0038 from the nearest
training picture, the closest so far (VAE 0.0041).</p>""",
            "{fig1}",
            """<p>The raw generator is unstable: its score moves between about 27% and 45% from pass to pass, while the
averaged weights rise steadily from pass 14 to 38%.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>GANs give sharp samples, but training is a balancing act;
gradient penalties and weight averaging make it workable.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The GAN's samples were recognized 38.4% of the time, the VAE's 9.4%. About how "
                                    "many times more often, to one decimal?",
             "answer": "About 4.1.", "why": "38.4 / 9.4 ≈ 4.09.",
             "key": {"parts": [{"label": None, "value": 4.1, "tol": 0.1, "unit": None}]}},
            {"kind": "number", "q": "What share of the 135 combinations did the GAN cover, in percent (whole "
                                    "number)?",
             "answer": "About 72%.", "why": "97 / 135 ≈ 0.719.",
             "key": {"parts": [{"label": None, "value": 72, "tol": 1, "unit": "%"}]}},
            {"kind": "short", "q": "Why sample from an average of the generator's weights rather than the latest "
                                   "weights?",
             "answer": "The latest weights swing as the game goes back and forth; the average smooths those swings "
                       "and gives steadier, better samples.",
             "why": "In our run the raw score jumped between about 27% and 45%; the average rose steadily."},
        ],
    },
    {
        "title": "Mode collapse, and GANs today",
        "segment": (105.6, 150.4),
        "figures": [{"t": 126.8, "caption": "A generator 10× too fast: every sample the same pink dash at pass 3, the same "
                                            "green square at pass 10; at most 5 of 135 combinations."},
                    {"t": 140.5, "caption": "Diffusion has largely replaced GANs for pictures, but learned "
                                            "discriminators live on, for example in speech vocoders."},
                    {"t": 149.9, "caption": "One GAN training step in code."}],
        "body": [
            """<p>Give the generator a 10× larger learning rate and 5 updates per discriminator update, and it
<b>collapses</b>. It finds one output the discriminator currently accepts and makes nearly every sample that. At
pass 3 every sample was the same pink dash, and at pass 10 the same green square. In 15 passes it never covered more
than 5 of the 135 combinations. When D catches on, G hops to another single trick instead of spreading out. A
high score on single samples can hide this, so measure <b>coverage</b> as well as quality.</p>""",
            "{fig0}",
            """<p>For pictures, diffusion models (episodes 5–9) have largely replaced GANs: they train with a plain
regression loss and cover the data well. But learned judges remain common. Many speech vocoders, including the one
in the voice model that narrates this series (episode 17), are trained with adversarial discriminators.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>A GAN can win its game by making one convincing thing over
and over; always check how much of the data a generator covers.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“A collapsed GAN can have high-quality samples and still be a bad generator.”",
             "answer": "True.", "why": "Every sample may look real while covering only a few of the possible "
                                       "pictures.", "key": {"value": True}},
            {"kind": "code", "q": "<b>Try it yourself.</b> One GAN training step.",
             "code": """fake = G(torch.randn(64, 64))
d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)
opt_d.zero_grad(); d_loss.backward(); opt_d.step()
g_loss = bce(D(fake), ones)          # fool the discriminator
opt_g.zero_grad(); g_loss.backward(); opt_g.step()""",
             "answer": "train_gan.py does this (plus the R1 penalty and the weight average).",
             "why": "fake.detach() stops the discriminator's step from changing the generator."},
        ],
    },
]
