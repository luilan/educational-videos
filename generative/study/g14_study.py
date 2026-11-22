"""Study guide content for How Generative Models Work: Images and Sound, episode 14: Diffusion Transformers.

Build:  python framework/study_guide.py generative g14 --video generative/media/videos/g14_scene/1080p60/DiTVideo.mp4
Every number comes from code/g14_dit (dit.py, 2026-11-22) and episode 13's code/g13_latent_diffusion. DiT is
Peebles & Xie 2023; Sora's "spacetime patches" are from OpenAI's 2024 technical report.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 14",
    "title": "Diffusion Transformers",
    "tagline": "Pictures as tokens.",
    "duration": "2:23",
    "intro": """<p>Episode 13's denoiser was a U-Net, built from convolutions. Many of today's largest image and
video models, such as Stable Diffusion 3, FLUX and Sora, use a <b>transformer</b> instead. This lesson swaps episode
13's U-Net for a small <b>Diffusion Transformer</b> (DiT, Peebles &amp; Xie 2023) and keeps everything else the same.
It compares the two at equal training and looks at what patch size costs.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 13 (latent diffusion, cross-attention), episode
11 (guidance) and the attention lessons of the earlier series. Code: <code>code/g14_dit</code> (PyTorch; about 3 hours
on a CPU for the 40,000-step model).</div>""",
}

CONCEPTS = [
    {
        "title": "Pictures as tokens",
        "segment": (7.5, 72.1),
        "figures": [{"t": 34.0, "caption": "Patchify: the 8×8×4 latent cut into 16 patches of 2×2×4 = 16 numbers, "
                                           "each mapped to a 128-number token, plus its position."},
                    {"t": 48.2, "caption": "One DiT block: self-attention over the patches, cross-attention to the "
                                           "caption's words, an MLP. Eight blocks."},
                    {"t": 61.5, "caption": "adaLN-Zero: time and caption set each norm's scale and shift and a gate "
                                           "on each branch; the gates start at zero."}],
        "body": [
            """<p>A transformer reads a list of tokens, so the latent must become one. <b>Patchify</b>: cut the 8×8×4
latent into 2×2 squares. That gives 16 patches of 2·2·4 = 16 numbers each. A linear layer turns each patch into a
token of 128 numbers. A learned <b>position embedding</b> is added, because attention by itself does not know where a
token came from.</p>""",
            "{fig0}",
            """<p>Then come 8 transformer blocks. In each one:</p>
<ul><li><b>self-attention</b>: every patch looks at every other patch;</li>
<li><b>cross-attention</b>: every patch looks at CLIP's vectors for the caption's words, exactly as in episode 13
(PixArt-α adds text to a DiT the same way);</li>
<li>an <b>MLP</b> on each token.</li></ul>""",
            "{fig1}",
            """<p>Time and the caption steer every block through <b>adaLN-Zero</b>. They are summed into one
conditioning vector <i>c</i>. From <i>c</i>, each block computes the scale and shift of its two LayerNorms and a
<b>gate</b> that multiplies each branch before it is added back. Those layers start at zero, so every gate starts at
0 and every block starts out as the identity; training opens the gates. Peebles &amp; Xie found this the best of the
ways they tried to feed in the conditioning.</p>""",
            "{fig2}",
            """<p>At the end, a final adaLN and a linear layer turn each token back into its 2×2×4 patch of predicted
noise (<b>unpatchify</b>). The loss, sampler, autoencoder, text encoder and 40,000 training steps are all episode
13's. The model has 2.85 million parameters, against 2.48 million for the U-Net.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A DiT is a plain transformer on latent patches. Time and
text steer it through gated, scaled normalizations that start out doing nothing.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Stable Diffusion's latents are 64×64×4. With 2×2 patches, how many tokens is that?",
             "answer": "1,024.", "why": "(64/2)·(64/2) = 32·32 = 1,024 tokens of 16 numbers each.",
             "key": {"parts": [{"label": None, "value": 1024, "tol": 0.5, "unit": "tokens"}]}},
            {"kind": "mc", "q": "Why do the adaLN-Zero gates start at zero?",
             "options": ["So the model ignores the caption", "So every block starts as the identity, which makes the "
                         "deep network easy to train from the start", "To save memory",
                         "Because time must be zero at the start"],
             "answer": "B.", "why": "With all gates at 0, the whole network starts as a simple pass-through, and "
                                    "training gradually lets each block contribute.", "key": {"choice": 1}},
        ],
    },
    {
        "title": "Transformer vs U-Net",
        "segment": (72.6, 118.2),
        "figures": [{"t": 84.6, "caption": "Share of captions followed after 40,000 steps each; below, the "
                                           "never-seen “purple triangle at the top left”."},
                    {"t": 101.4, "caption": "Training loss: the transformer stays above the U-Net and is still "
                                            "falling at 40,000 steps."},
                    {"t": 117.9, "caption": "16 vs 64 tokens after 10,000 steps: each step about 3× slower, no better."}],
        "body": [
            """<table><tr><th>40,000 steps</th><th>U-Net</th><th>transformer</th></tr>
<tr><td>trained captions, w = 3</td><td>99.2%</td><td>99.4%</td></tr>
<tr><td>never-seen captions, w = 3</td><td>99.2%</td><td>98.8%</td></tr>
<tr><td>trained captions, w = 1</td><td><b>99.4%</b></td><td>97.6%</td></tr>
<tr><td>never-seen captions, w = 1</td><td><b>98.8%</b></td><td>96.1%</td></tr>
<tr><td>final training loss</td><td>0.0122</td><td>0.0169</td></tr></table>
<p>“Followed” means the picture is recognizable and its shape, color and position match the caption. The same
judge, captions and seeds were used as in episode 13. With guidance (episode 11) the two are level; without it the
U-Net stays ahead.</p>""",
            "{fig0}",
            """<p>Why? A convolution looks only at neighbors, so the assumption that nearby positions belong together
is built in. A transformer compares every token with every other and must <i>learn</i> the layout from its position
embedding. With 2.9 million parameters and 40,000 steps that costs it, and its loss was still falling when training
stopped (0.0179 at 38,000 steps, 0.0169 at 40,000). The transformer's strength is scale. In the DiT paper, quality
improved steadily with the model's compute (Gflops) across all sizes, and DiT-XL/2 set a new state of the art on
ImageNet 256×256.</p>""",
            "{fig1}",
            """<p><b>Patch size.</b> Smaller patches mean more tokens: 1×1 patches give 64 tokens instead of 16.
Attention compares every pair of tokens, so its cost grows with the square of their number. After 10,000 steps
each:</p>
<table><tr><th></th><th>16 tokens (2×2)</th><th>64 tokens (1×1)</th></tr>
<tr><td>time per training step</td><td>1×</td><td>about 3× (2.6–3.8× across measurements)</td></tr>
<tr><td>trained captions, w = 1</td><td>95.1%</td><td>94.1%</td></tr>
<tr><td>never-seen captions, w = 1</td><td>90.6%</td><td>89.8%</td></tr></table>
<p>On a latent this small, 2×2 patches lose nothing. On 256×256 pictures, the DiT paper found smaller patches
better: the extra tokens are extra compute.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>At small scale, the U-Net's built-in assumption about
neighbors gives it a head start. Transformers trade that assumption for a design that keeps improving as it
grows.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Going from 16 to 64 tokens, by how much does the number of token pairs that "
                                    "attention compares grow?",
             "answer": "16×.", "why": "64² / 16² = 4,096 / 256 = 16. Other parts of the step (the MLPs, the "
                                      "embeddings) grow only 4×, which is why the measured step was about 3× slower, "
                                      "not 16×.",
             "key": {"parts": [{"label": None, "value": 16, "tol": 0.1, "unit": "×"}]}},
            {"kind": "tf", "q": "“The transformer lost to the U-Net, so transformers are worse denoisers.”",
             "answer": "False.", "why": "Only at this tiny scale and training length, and only without guidance. "
                                        "With guidance they tie, the transformer's loss was still falling, and at "
                                        "scale the DiT paper found the opposite.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "Beyond pictures",
        "segment": (118.7, 135.2),
        "figures": [{"t": 129.6, "caption": "Video as spacetime patches: small blocks across space and time, each "
                                            "a token (OpenAI, Sora report, 2024)."},
                    {"t": 135.0, "caption": "A diffusion transformer in code."}],
        "body": [
            """<p>A transformer only sees a list of tokens, and nothing in it is specific to two-dimensional
pictures. OpenAI's Sora report (2024) describes video compressed into a latent and cut into <b>spacetime
patches</b>, small blocks spanning a few pixels and a few frames, which a diffusion transformer then denoises. The
same idea carries to any data that can be cut into patches.</p>""",
            "{fig0}",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Once data is a list of patches, the same transformer
denoiser works for pictures, video and more.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A video latent of 16 frames × 32×32 positions is cut into patches of 2 frames × "
                                    "2×2 positions. How many tokens is that?",
             "answer": "2,048.", "why": "(16/2)·(32/2)·(32/2) = 8·16·16 = 2,048.",
             "key": {"parts": [{"label": None, "value": 2048, "tol": 0.5, "unit": "tokens"}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> The DiT's forward pass.",
             "code": """tokens = embed(patchify(z_noisy)) + position
c = time_mlp(t) + text(caption_vector)
for block in blocks:
    tokens = block(tokens, c, word_vectors)
noise_guess = unpatchify(final(tokens, c))""",
             "answer": "dit.py does exactly this; PATCH=1 gives 64 tokens.",
             "why": "Everything around it (loss, sampler, autoencoder) is episode 13's, unchanged."},
        ],
    },
]
