"""Study guide content for How Generative Models Work: Images and Sound, episode 8: The Denoiser: U-Nets and Time.

Build:  python framework/study_guide.py generative g08 --video generative/media/videos/g08_scene/1080p60/UNetsAndTimeVideo.mp4
Every number comes from code/g08_unets_and_time (2026-11-22): episode 6's run repeated without skip connections and
without the time input; 500 samples from the same starting noise, 50 steps, scored by episode 3's judge.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 8",
    "title": "The Denoiser: U-Nets and Time",
    "tagline": "Two small parts of the network, and what happens without them.",
    "duration": "2:32",
    "intro": """<p>Episode 6 trained a <b>U-Net</b> to predict noise. This lesson opens it up. It shows the U shape, then
tests two of its parts by retraining the same network without each one: the <b>skip connections</b> that carry
detail across the U, and the <b>time input</b> that tells the network which step it is on. Both runs repeat episode 6
exactly, apart from the one missing part.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 6 and 7. Code:
<code>code/g08_unets_and_time</code> (PyTorch; two training runs of about 2.5 hours each on an 8-core CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "The denoiser's job, and the U shape",
        "segment": (7.5, 57.6),
        "figures": [{"t": 23.3, "caption": "Samples from the full network (70% recognized), without skip connections "
                                           "(0%) and without the time input (1.4%). DDIM, 50 steps."},
                    {"t": 38.2, "caption": "In: a noisy picture and the step number. Out: a noise guess the same size "
                                           "as the picture."},
                    {"t": 57.3, "caption": "The U-Net: down to 8×8 and back up, with skip connections across."}],
        "body": [
            """<p>Remove one part of episode 6's network and retrain it the same way, and the results collapse:</p>
<table><tr><th>network</th><th>parameters</th><th>DDIM, 50 steps</th><th>fresh noise (η = 1), 50 steps</th></tr>
<tr><td>full U-Net (episode 6)</td><td>1,182,371</td><td>70.0%</td><td>95.2%</td></tr>
<tr><td>no skip connections</td><td>1,125,763</td><td>0.0%</td><td>8.4%</td></tr>
<tr><td>no time input</td><td>1,182,371</td><td>1.4%</td><td>32.8%</td></tr></table>
<p>(Share of 500 samples the judge from episode 3 recognizes; the two samplers are episode 7's.)</p>""",
            "{fig0}",
            """<p>The denoiser's job fixes its shape. It gets a noisy picture x_t and the step t, and returns a guess of
the noise, 3 × 32 × 32 numbers, one for every pixel value. So the output must keep every detail of the input.</p>""",
            "{fig1}",
            """<p>A <b>U-Net</b> (Ronneberger et al., 2015) does this in two halves. The down half halves the
resolution twice, from 32×32 to 16×16 to 8×8, while the channels grow from 32 to 64 to 128. Deep in the U, each
unit sees a large part of the picture, enough to tell a triangle from a square. The up half grows back to 32×32.
<b>Skip connections</b> hand the down half's features at each size straight across to the up half, where they are
joined with the upsampled features.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>The down half sees the whole shape; the skip connections
keep every pixel's detail; the up half combines the two into a full-size answer.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why must the denoiser's output have the same size as its input?",
             "options": ["To save memory", "It predicts a noise value for every pixel value of the picture",
                         "So it can be trained faster", "Because the judge needs 32×32 pictures"],
             "answer": "B.", "why": "The noise added to every pixel is different; the guess needs one number per pixel "
                                    "value.", "key": {"choice": 1}},
        ],
    },
    {
        "title": "Without skip connections",
        "segment": (58.1, 75.6),
        "figures": [{"t": 75.3, "caption": "Noise-guess error by noise level (log scale), and samples from the "
                                           "network without skip connections: blobs with ragged edges."}],
        "body": [
            """<p>Without the skip connections, everything the up half knows must pass through the 8×8 middle. (The
residual connections inside each block stay; only the long connections across the U are removed.) We measured the
noise-guess error on 256 fresh pictures at each noise level:</p>
<table><tr><th>step t</th><th>10</th><th>50</th><th>100</th><th>200</th><th>400</th><th>600</th><th>800</th><th>950</th></tr>
<tr><td>full</td><td>0.0114</td><td>0.0029</td><td>0.0017</td><td>0.0010</td><td>0.0011</td><td>0.0018</td><td>0.0023</td><td>0.0007</td></tr>
<tr><td>no skips</td><td>0.0261</td><td>0.0088</td><td>0.0077</td><td>0.0089</td><td>0.0113</td><td>0.0094</td><td>0.0050</td><td>0.0011</td></tr></table>
<p>At low and middle noise the error is 2 to 10 times larger, 4.6 times at step 100. The samples are blobs in the
right colors but with ragged edges, not shapes: none recognized with DDIM, 8.4% with fresh noise.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>A small bottleneck can hold what the picture is, but not
where every edge goes; skip connections carry the detail around it.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "At step 100, the full network's error is 0.00166 and the no-skip network's is "
                                    "0.00767. How many times larger is the no-skip error? (one decimal)",
             "answer": "4.6.", "why": "0.00767 / 0.00166 ≈ 4.6.",
             "key": {"parts": [{"label": None, "value": 4.6, "tol": 0.05, "unit": None}]}},
        ],
    },
    {
        "title": "The time input",
        "segment": (76.1, 144.8),
        "figures": [{"t": 88.5, "caption": "The step number as sines and cosines at 64 frequencies, for t = 50, 400 "
                                           "and 900."},
                    {"t": 102.8, "caption": "Without the time input, one denoising step is still nearly as good: at "
                                            "most 1.6× the full model's error."},
                    {"t": 121.8, "caption": "But sampling fails: many samples are still noisy after 50 steps."},
                    {"t": 144.5, "caption": "The time embedding, and the one line that adds it into every block."}],
        "body": [
            """<p>The step t is turned into a vector of sines and cosines at many frequencies, the same trick as a
transformer's position encoding. A small MLP mixes it, and every block adds it to its channels. This tells the
network how much noise to expect.</p>""",
            "{fig0}",
            """<p>Hide it (always pass step 0) and retrain. Surprisingly, a single denoising step still works. The
network can judge the noise level from the picture itself, and its error is at most 1.6 times the full model's at
every level. Its final training loss is 0.0037, against 0.0028 for the full model and 0.0086 without skips.</p>""",
            "{fig1}",
            """<p>Sampling is another matter: 1.4% recognized with DDIM, 32.8% with fresh noise, and many samples are
still covered in noise at the end. The sampler computes each step from the schedule's ᾱ_t, so when the network
misjudges the noise level, it removes the wrong amount. The next picture is then off the schedule, which makes the
next judgment harder. A good average loss is not enough: a denoiser has to be right at every step of the walk.</p>""",
            "{fig2}",
            """<p>Larger models are more forgiving. Sun et al. (2025, “Is Noise Conditioning Necessary for Denoising
Generative Models?”) found that many models work almost as well without the time input, and that deterministic DDIM
sampling failed worst, the same pattern as here. Standard models still pass the time in: it costs almost nothing
and makes every step easier.</p>""",
            "{fig3}",
            """<div class="box key"><b class="t">Key idea</b>Telling the network the step keeps every step of the
walk consistent with the schedule the sampler follows.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Without the time input, our network's single-step noise guesses were more than "
                                "twice as bad as the full model's.”",
             "answer": "False.", "why": "At most 1.6 times the full model's error; the damage shows up in sampling, "
                                        "not in single steps.", "key": {"value": False}},
            {"kind": "short", "q": "Why can a network with a small single-step error still produce bad samples?",
             "answer": "Sampling chains 50 or more steps; small, systematic mistakes (here, misjudging the noise "
                       "level) push the picture off the schedule the sampler assumes, and they build up.",
             "why": "Our no-time network: at most 1.6× the error per step, but 1.4% recognized with DDIM."},
            {"kind": "code", "q": "<b>Try it yourself.</b> The time embedding from <code>diffusion.py</code>.",
             "code": """freqs = torch.exp(-math.log(10_000) * torch.arange(dim // 2) / (dim // 2))
args = t[:, None] * freqs[None]
temb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)""",
             "answer": "Each block then runs h = h + self.time(temb)[:, :, None, None].",
             "why": "Pass zeros for t (as train_ablation.py notime does) and the network can no longer see the step."},
        ],
    },
]
