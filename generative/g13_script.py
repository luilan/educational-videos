"""How Generative Models Work: Images and Sound, episode 13 — Latent Diffusion. One entry per narration section.

Every number comes from code/g13_latent_diffusion (train_autoencoder.py, latent_diffusion.py, figures.py,
2026-11-22): an autoencoder for Shapes pictures, and diffusion on its latents, conditioned on episode 12's CLIP text
encoder. Stable Diffusion's numbers are from its published architecture (Rombach et al. 2022, v1 models).
"""
TITLE = "Latent Diffusion"
TAGLINE = "Compress first, then diffuse."
NEXT = "Diffusion Transformers"
SECTIONS = [
    # 1. Hook
    "Our pictures are 32 by 32 pixels: 3,072 numbers. A 512 by 512 picture has over 780,000. Running a U-Net over "
    "every one of them, fifty times per picture, is slow. So don't diffuse the pixels.",
    # 2. Autoencoder
    "First, train an autoencoder. The encoder squeezes each picture into an 8 by 8 grid with 4 channels: 256 "
    "numbers, 12 times fewer. The decoder turns it back into pixels. 99.8 percent of the rebuilt test pictures "
    "read exactly like the originals.",
    # 3. What a latent looks like
    "The latent is still a small map. Each channel lights up where the shape is. So a U-Net, built for pictures, "
    "still makes sense here.",
    # 4. A light penalty
    "Like a VAE, it gets a small penalty that keeps the latents smooth and well scaled. Make it too strong, and "
    "every picture comes back as a blurry blob. We use a very light one.",
    # 5. Diffusion in latent space
    "Then run diffusion on the latents: add noise, guess the noise, the same loss as episode 6. To draw, denoise a "
    "random latent and decode it once at the end. Even with a U-Net twice as big, a training step is four times "
    "cheaper.",
    # 6. Text in
    "The request is now a sentence, read by episode 12's CLIP text encoder. The simple way in: add its one "
    "caption vector to the time embedding, like episode 10's labels.",
    # 7. Cross-attention
    "Stable Diffusion does more. Inside the U-Net, every latent position looks at the vectors of the individual "
    "words, and takes what it needs. This is cross-attention.",
    # 8. Results
    "Both draw trained captions right 99 percent of the time. On the eight combinations never seen, the single "
    "vector gets 94.5 percent right. Cross-attention gets 98.8.",
    # 9. The trap
    "One trap we fell into. Pixels live between minus one and one, so samplers clip their guess of the clean "
    "picture. We clipped latents too, at four. But latents have no fixed range: the shape's own values reach "
    "twelve. That one line held our models to 44 percent. Without it, 99.",
    # 10. Real models
    "Stable Diffusion uses the same recipe at scale: 512 by 512 pictures become 64 by 64 by 4 latents, 48 times "
    "fewer numbers, a U-Net of about 860 million parameters, and CLIP's text encoder, attended to through "
    "cross-attention.",
    # 11. Code
    "In code: encode the pictures once, train diffusion on the latents, and decode only the final sample.",
    # 12. Next
    "Next: diffusion transformers, replacing the U-Net with a transformer.",
]
