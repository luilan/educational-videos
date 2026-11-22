"""Study guide content for How Generative Models Work: Images and Sound, episode 13: Latent Diffusion.

Build:  python framework/study_guide.py generative g13 --video generative/media/videos/g13_scene/1080p60/LatentDiffusionVideo.mp4
Every number comes from code/g13_latent_diffusion (train_autoencoder.py, latent_diffusion.py, figures.py,
2026-11-22). Stable Diffusion's numbers are from its published v1 architecture (Rombach et al. 2022).
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 13",
    "title": "Latent Diffusion",
    "tagline": "Compress first, then diffuse.",
    "duration": "2:46",
    "intro": """<p>Diffusing every pixel of a large picture, fifty times per sample, is expensive. Latent diffusion
first trains an autoencoder that compresses pictures into a small grid of numbers, runs diffusion there, and decodes
only the final result. This lesson builds the whole Stable Diffusion recipe on Shapes: the autoencoder, diffusion on
its latents, and a caption fed in through episode 12's CLIP text encoder, either as one vector or through
<b>cross-attention</b>.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 6–8 (diffusion and the U-Net), 10 and 11
(conditioning, guidance), 12 (CLIP). Code: <code>code/g13_latent_diffusion</code> (PyTorch; the autoencoder trains
in about 70 minutes on a CPU, each latent diffusion model in 1–2 hours).</div>""",
}

CONCEPTS = [
    {
        "title": "Compress first",
        "segment": (7.5, 62.9),
        "figures": [{"t": 43.2, "caption": "Encoder → 8×8×4 latent (256 numbers) → decoder. 99.8% of rebuilt test "
                                           "pictures read like the originals."},
                    {"t": 51.4, "caption": "Four pictures and the four channels of their latents (red positive, "
                                           "blue negative)."},
                    {"t": 62.6, "caption": "The KL penalty's weight: too strong (1e-4) blurs everything; very "
                                           "light (1e-6) keeps the pictures."}],
        "body": [
            """<p>A 32×32 color picture is 3,072 numbers; a 512×512 one is 786,432. A diffusion model runs its
U-Net over all of them at every one of its ~50 sampling steps, and over all of them at every training step.</p>
<p>An <b>autoencoder</b> learns to avoid that. Its encoder (convolutions that halve the resolution twice) maps a
32×32×3 picture to an 8×8×4 <b>latent</b>: 256 numbers, 12 times fewer. Its decoder maps the latent back to pixels.
It is trained only to rebuild its input: after 6,000 steps the test error is 0.000117, and 99.8% of 1,000 rebuilt
test pictures are read by episode 3's judge exactly like their originals.</p>""",
            "{fig0}",
            """<p>The latent is not an arbitrary code: it is still a small map of the picture. Each of the four
channels is strong where the shape is and quiet elsewhere. That is why a convolutional U-Net, built for pictures,
also works on latents.</p>""",
            "{fig1}",
            """<p>Like Stable Diffusion's, the autoencoder is a lightly regularized <b>VAE</b>: next to the
reconstruction error, a KL penalty pulls the latents toward a smooth, standard-normal shape. Its weight matters. At
1e-4, no rebuilt picture was recognizable (blurry, colorless blobs); at 1e-6, 99.8%. The latents are then scaled to
unit standard deviation (×0.306), the scale diffusion's noise schedule expects.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>An autoencoder turns pictures into a small map of
numbers and back, almost losslessly. Diffusion can then work on the small map.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Stable Diffusion's autoencoder turns a 512×512×3 picture into a 64×64×4 latent. "
                                    "How many times fewer numbers is that?",
             "answer": "48.", "why": "786,432 / 16,384 = 48.",
             "key": {"parts": [{"label": None, "value": 48, "tol": 0.5, "unit": "×"}]}},
            {"kind": "mc", "q": "Why does a U-Net still make sense on latents?",
             "options": ["Latents are random codes, and U-Nets work on any numbers",
                         "The latent is still a spatial map: neighboring positions describe neighboring parts of "
                         "the picture", "The U-Net is replaced by a transformer", "The latent is one vector"],
             "answer": "B.", "why": "Convolutions assume nearby numbers belong together.", "key": {"choice": 1}},
        ],
    },
    {
        "title": "Diffusion on latents, with text",
        "segment": (63.4, 113.3),
        "figures": [{"t": 77.9, "caption": "Drawing: denoise a random latent for 50 steps, decode once. A training "
                                           "step is 4× cheaper than on pixels."},
                    {"t": 100.1, "caption": "Cross-attention: each latent position looks at the vectors of the "
                                            "caption's words (weights drawn for illustration)."},
                    {"t": 113.0, "caption": "Never-seen “purple triangle at the top left”: one caption vector vs "
                                            "cross-attention."}],
        "body": [
            """<p>Encode all training pictures once. Then diffusion is episode 6's, on latents instead of pixels:
noise a latent, train the U-Net to guess the noise. To draw, start from a random 8×8×4 latent, denoise it with
episode 7's sampler, and decode it once at the end. The latent U-Net here has 2.5 million parameters, twice episode
6's pixel U-Net, yet one training step takes 261 ms instead of 1,038 ms (batch 128, CPU): it works on 16 times fewer
positions.</p>""",
            "{fig0}",
            """<p>The request is now a caption, “a red triangle in the center”, read by episode 12's CLIP text
encoder, kept frozen. Two ways in:</p>
<ul><li><b>One caption vector:</b> CLIP's 64-number summary of the caption is added to the time embedding, like
episode 10's labels.</li>
<li><b>Cross-attention:</b> after five of the U-Net's blocks, every latent position becomes a query in an attention
layer whose keys and values are CLIP's vectors for the individual words (episode 12's transformer outputs, before
averaging). Each position takes what it needs from the words. Stable Diffusion feeds text in this way.</li></ul>""",
            "{fig1}",
            """<table><tr><th>caption in via</th><th>127 trained captions</th><th>8 never-seen captions</th></tr>
<tr><td>one caption vector</td><td>99.0%</td><td>94.5%</td></tr>
<tr><td>cross-attention</td><td>99.4%</td><td><b>98.8%</b></td></tr></table>
<p>Both after 40,000 steps, w = 1, “follows” = recognizable with the right shape, color and position. With 17 words
and three facts per caption, one vector almost suffices; cross-attention mainly helps on combinations never seen in
training, where the single vector draws the purple triangle bluish. Real captions say far more than one vector can
hold, which is why large models attend to the words.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Latent diffusion is ordinary diffusion on the
autoencoder's latents, decoded once at the end. Text comes in through cross-attention to the words.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“During sampling, the decoder runs at every one of the 50 steps.”",
             "answer": "False.", "why": "Only once, on the final latent; the 50 steps all happen in latent space.",
             "key": {"value": False}},
            {"kind": "short", "q": "In cross-attention, what are the queries, and what are the keys and values?",
             "answer": "Queries: the latent positions (from the U-Net's features). Keys and values: the caption's "
                       "word vectors from the text encoder.",
             "why": "Each position asks which words matter to it, and takes a weighted mix of their values."},
        ],
    },
    {
        "title": "A trap, and the real thing",
        "segment": (113.8, 158.2),
        "figures": [{"t": 131.7, "caption": "Clipping the clean-latent guess at ±4 (top) vs not (bottom), same model "
                                            "and caption."},
                    {"t": 151.2, "caption": "Our model and Stable Diffusion v1, side by side."},
                    {"t": 157.9, "caption": "Latent diffusion in code."}],
        "body": [
            """<p>Pixel samplers clip their guess of the clean picture to the pixel range, −1 to 1 (episode 7). We
first clipped latents too, at ±4. But latents have no fixed range: their standard deviation is 1, yet the shape's
own values reach about ±12. Clipping real latents at ±4 and decoding leaves only 46.3% of 1,000 test pictures
exactly right (no clipping: 99.3%), and in sampling it held the cross-attention model to 43.5% of captions followed,
mostly with wrong, muddy colors. Clipping only as a guard (±16) gives 99.4%.</p>""",
            "{fig0}",
            """<p>Stable Diffusion (Rombach et al., 2022) is the same recipe at scale. Version 1 encodes 512×512×3
pictures into 64×64×4 latents (48× fewer numbers), runs a U-Net of about 860 million parameters on them, and
reads prompts with the text encoder of OpenAI's CLIP ViT-L/14, attended to through cross-attention at several
resolutions. Classifier-free guidance (episode 11) steers the result.</p>""",
            "{fig1}",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Latents are not pixels: anything the sampler assumes
about the data's range has to be checked again in latent space.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Our latent U-Net takes 261 ms per training step and the pixel U-Net 1,038 ms. "
                                    "How many times cheaper is the latent step? (one decimal)",
             "answer": "4.0×.", "why": "1,038 / 261 ≈ 3.98.",
             "key": {"parts": [{"label": None, "value": 4.0, "tol": 0.05, "unit": "×"}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Latent diffusion around episode 6's loop.",
             "code": """z = autoencoder.encode(pictures)            # once
loss = mse(unet(noise(z, t, eps), t, caption), eps)
z = sample(unet, caption, torch.randn(n, 4, 8, 8))
pictures = autoencoder.decode(z)""",
             "answer": "latent_diffusion.py does exactly this; set X0_CLIP=4 with EVAL=1 to reproduce the trap.",
             "why": "Everything in between is episode 6–7's code, unchanged."},
        ],
    },
]
