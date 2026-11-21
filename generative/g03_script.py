"""How Generative Models Work: Images and Sound, episode 3 — Variational Autoencoders. One entry per narration section.

Every number comes from code/g03_variational_autoencoders (train_vae.py and judge.py, 2026-11-21): episode 2's network
with an 8-number code, trained 40 epochs on 20,000 Shapes pictures as a plain autoencoder and as a VAE, plus a VAE
with a 2-number code; generated pictures scored by a small judge network.
"""
TITLE = "Variational Autoencoders"
TAGLINE = "Make every code mean something"
NEXT = "GANs: a Forger and a Judge"
SECTIONS = [
    # 1. Hook
    "An autoencoder's codes form islands, and random codes land in the gaps between them. A variational autoencoder "
    "changes two things, so that the space of codes has no gaps.",
    # 2. Change one: clouds
    "First change: the encoder no longer outputs a point. For every code number it outputs a mean and a spread: a "
    "small cloud. Training picks a random code inside the cloud, mean plus spread times noise, and the decoder has "
    "to rebuild the picture from wherever it landed.",
    # 3. Change two: KL
    "Second change: a penalty, called the K L divergence, pulls every cloud toward the same target, a standard normal "
    "centered on zero. Two forces now pull against each other. Rebuilding wants each cloud small and far from the "
    "others. The penalty wants them wide and on top of each other. So the clouds overlap, and the gaps close.",
    # 4. The map
    "With a two-number code you can see it. The autoencoder's codes were islands spread over twenty units. The "
    "VAE's codes fill one round patch around zero. Decode a grid of codes across that patch, and every cell is a "
    "shape. Position and color change smoothly from cell to cell.",
    # 5. Sampling
    "Generating is now simple: draw a code from a standard normal, and decode it. With an eight-number code, the "
    "samples land about half as far from a real picture as the autoencoder's random codes did. Most of them are one "
    "blob, in one place.",
    # 6. The blur
    "But look closer. They're blurry, often with a faint second shape. We trained a small judge network that names "
    "the shape, color and position, and rejects junk. It passes almost every real picture, and only about one in ten "
    "samples, from the VAE or from the autoencoder.",
    # 7. Why blur
    "The blur comes from the noise. During training, nearby codes stand for several different pictures, so the "
    "decoder learns to draw their average. Averages of sharp pictures are blurry. That's why today a VAE is "
    "rarely the generator. It's the compressor inside image generators like Stable Diffusion.",
    # 8. Code
    "In code, the encoder returns a mean and a log variance. The code is the mean plus the spread times random "
    "noise, and the loss adds the K L penalty to the rebuilding error.",
    # 9. Next
    "Next: GANs. Instead of measuring pixel errors, train a second network to spot fakes, and let it teach the "
    "generator to be sharp.",
]
