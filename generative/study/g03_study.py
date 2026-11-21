"""Study guide content for How Generative Models Work: Images and Sound, episode 3: Variational Autoencoders.

Build:  python framework/study_guide.py generative g03 --video generative/media/videos/g03_scene/1080p60/VAEVideo.mp4
Every number comes from code/g03_variational_autoencoders (train_vae.py, judge.py; 2026-11-21): episode 2's network
with an 8-number code trained 40 epochs as a plain autoencoder and as a VAE, a 2-number VAE, and a judge network.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 3",
    "title": "Variational Autoencoders",
    "tagline": "Make every code mean something",
    "duration": "2:30",
    "intro": """<p>A <b>variational autoencoder</b> (VAE) changes two things about an autoencoder. Each picture is
encoded as a cloud of codes rather than one point, and a <b>KL penalty</b> pulls every cloud toward a standard normal
distribution. The clouds overlap, the gaps between islands close, and generating becomes simple: draw a code from a
standard normal and decode it. This lesson measures what that buys (samples near real pictures, a smooth code map) and
what it costs (blur). It also introduces the <b>judge</b>, a small network that scores generated pictures and is used
for the rest of the series.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 2, and Foundations episode 6 (probability,
the normal distribution). Code: <code>code/g03_variational_autoencoders</code> (PyTorch on a CPU, about 10
minutes).</div>""",
}

CONCEPTS = [
    {
        "title": "Clouds and the KL penalty",
        "segment": (7.5, 56.8),
        "figures": [{"t": 35.4, "caption": "The encoder outputs a mean and a spread per code number; training samples "
                                           "code = mean + spread × noise."},
                    {"t": 56.3, "caption": "Rebuilding pulls the clouds apart; the KL penalty pulls them toward a "
                                           "standard normal; they end up overlapping."}],
        "body": [
            """<p>For each code number, a VAE's encoder outputs a <b>mean</b> μ and a <b>spread</b> σ (in practice
the log variance, log σ², so the network can output any real number). During training the code is sampled from that
cloud: <code>code = μ + σ × ε</code> with ε drawn from a standard normal. Writing the sample this way is the
<b>reparameterization trick</b>: the randomness sits in ε, so gradients still flow to μ and σ.</p>""",
            "{fig0}",
            """<p>The second change is a penalty, the <b>KL divergence</b> between each picture's cloud and a standard
normal N(0, 1). Per code number it is ½(μ² + σ² − 1 − log σ²), which is zero only when μ = 0 and σ = 1. The loss is
the rebuilding error plus this penalty. Rebuilding wants small clouds far apart, so pictures can be told apart. The
penalty wants wide clouds centered on zero. The compromise is clouds that overlap and fill the space, so the
gaps of episode 2 close. In our 8-number VAE the penalty settles at about 13 nats in total and every code number has a
spread close to 1 across the test set.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>A VAE encodes each picture as a cloud and pulls all clouds
toward one standard normal, so every code near zero means something.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "For one code number with μ = 0 and σ = 1, what is the KL penalty "
                                    "½(μ² + σ² − 1 − log σ²)?",
             "answer": "0.", "why": "½(0 + 1 − 1 − 0) = 0: the cloud already is the standard normal.",
             "key": {"parts": [{"label": None, "value": 0, "tol": 0, "unit": None}]}},
            {"kind": "number", "q": "μ = 2, σ = 1. What is the KL penalty for this code number?",
             "answer": "2.", "why": "½(4 + 1 − 1 − 0) = 2.",
             "key": {"parts": [{"label": None, "value": 2, "tol": 0, "unit": None}]}},
            {"kind": "short", "q": "Why write the sampled code as μ + σ × ε instead of sampling it directly?",
             "answer": "So the randomness is in ε, an input, and the code is a differentiable function of μ and σ; "
                       "gradients can flow back to the encoder.",
             "why": "The reparameterization trick."},
        ],
    },
    {
        "title": "A map with no gaps",
        "segment": (57.3, 89.8),
        "figures": [{"t": 73.9, "caption": "Left: the autoencoder's 2-number codes, spread from −13 to +10 with gaps. "
                                           "Right: the VAE's decoded grid from −2 to +2: every cell is a shape."},
                    {"t": 89.3, "caption": "Samples from a standard normal (8-number code): about half as far from the "
                                           "nearest training picture as the autoencoder's random codes."}],
        "body": [
            """<p>With a 2-number code the difference is visible. The autoencoder's codes spread from about −13 to
+10 and form islands. The VAE's codes fill one round patch around zero: they range from −3.4 to +3.6, and 77% lie
within distance 2 of the center. Decoding a 10 × 10 grid of codes from −2 to +2 gives a shape in every cell, with
position and color changing smoothly across the grid.</p>""",
            "{fig0}",
            """<p>Generating is now one line: draw a code from N(0, 1) and decode. With an 8-number code, samples are
0.0041 (squared error per number) from the nearest of the 20,000 training pictures, against 0.0076 for the
autoencoder's random codes, about half. Most samples are one blob in one place. The autoencoder also needed us to
measure the mean and spread of its codes first. The VAE was trained to make N(0, 1) the right distribution to sample
from.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Because a VAE's codes fill a known distribution, you can
generate by sampling from that distribution.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“To generate with a trained VAE, you need the encoder.”",
             "answer": "False.", "why": "Draw a code from N(0, 1) and use only the decoder.", "key": {"value": False}},
            {"kind": "number", "q": "VAE samples were 0.0041 from the nearest training picture, autoencoder samples "
                                    "0.0076. What fraction of the autoencoder's distance is the VAE's, to two "
                                    "decimals?",
             "answer": "About 0.54.", "why": "0.0041 / 0.0076 ≈ 0.54.",
             "key": {"parts": [{"label": None, "value": 0.54, "tol": 0.01, "unit": None}]}},
        ],
    },
    {
        "title": "The price: blur",
        "segment": (90.3, 138.5),
        "figures": [{"t": 107.3, "caption": "The judge recognizes 99.8% of real pictures, 9.4% of VAE samples and 9.8% "
                                            "of autoencoder samples."},
                    {"t": 126.4, "caption": "Codes near several pictures are trained toward all of them, so the decoder "
                                            "learns their average."},
                    {"t": 138.0, "caption": "The VAE loss in code."}],
        "body": [
            """<p>VAE samples are blurry and often carry a faint second shape. To put a number on quality we trained
a <b>judge</b> (<code>judge.py</code>): a small classifier that names a picture's shape, color and position and also
says whether it is one clean shape or junk. It was trained on junk such as averaged pairs, two shapes, off-palette
colors, heavy blur and noise. A picture is <b>recognizable</b> when the judge is at least 90% sure of all three
readings and finds it clean rather than junk. It passes 99.8% of real test pictures, 95% of slightly blurred ones
(σ = 0.5 pixel) and almost no junk. Of 500 samples, it recognizes 9.4% from the VAE and 9.8% from the autoencoder.
About 41% of either are readable, but most fail the clean test.</p>""",
            "{fig0}",
            """<p>The blur comes from the training noise. Because clouds overlap, the same code is sampled during
training for several different pictures. The rebuilding error is smallest when the decoder outputs their average,
and averages of sharp pictures are blurry. Reconstructions pay too: test error 0.00087 for the VAE against 0.00055 for
the autoencoder. That is why VAEs today are rarely the generator itself. Instead they are the <b>compressor</b> in
front of latent diffusion models such as Stable Diffusion (episode 13).</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>A VAE fixes the gaps but not the quality: a decoder trained
on overlapping codes learns to draw averages, which are blurry.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why are VAE samples blurry?",
             "options": ["The code is too short", "The decoder learns the average of pictures whose codes overlap",
                         "The KL penalty adds noise to the output pixels", "The judge is too strict"],
             "answer": "B.", "why": "The pixel loss is minimized by the average of the pictures a code may stand for.",
             "key": {"choice": 1}},
            {"kind": "tf", "q": "“The judge counts a picture as recognizable if it can name the shape, even if the "
                                "picture is junk.”",
             "answer": "False.", "why": "It must also find the picture clean rather than junk.", "key": {"value": False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> The VAE loss for one batch.",
             "code": """h = body(x)
mean, logvar = to_mean(h), to_logvar(h)
code = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
x_hat = decoder(code)
recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / len(x)
kl = 0.5 * (mean**2 + logvar.exp() - 1 - logvar).sum() / len(x)
loss = recon + kl""",
             "answer": "train_vae.py's VAE.loss; the KL term settles near 13 nats per picture.",
             "why": "exp(0.5 · logvar) is the spread σ."},
        ],
    },
]
