"""How Generative Models Work: Images and Sound, episode 14 — Diffusion Transformers. One entry per narration section.

Every number comes from code/g14_dit (dit.py, 2026-11-22): episode 13's latent diffusion with the U-Net replaced by a
small Diffusion Transformer (Peebles & Xie 2023). Sora's "spacetime patches" are from OpenAI's 2024 technical report.
"""
TITLE = "Diffusion Transformers"
TAGLINE = "Pictures as tokens."
NEXT = "Sound as Data"
SECTIONS = [
    # 1. Hook
    "Episode 13's denoiser was a U-Net, built from convolutions, made for pictures. Many of today's largest image "
    "and video models use a transformer instead. Let's swap ours.",
    # 2. Patchify
    "A transformer reads a list of tokens. So cut the 8 by 8 latent into 2 by 2 squares: 16 patches of 16 numbers "
    "each. A linear layer turns each patch into a token, and a position embedding says where it came from.",
    # 3. The block
    "Then 8 blocks, each doing three things. Self-attention: every patch looks at every other patch. "
    "Cross-attention: every patch looks at the caption's words, as in episode 13. And a small network on each token.",
    # 4. adaLN-Zero
    "Time and the caption steer every block. They set the scale and shift of each normalization, and a gate on "
    "each branch. The gates start at zero, so each block starts out doing nothing, and training opens them.",
    # 5. Unpatchify
    "At the end, each token turns back into its patch of predicted noise. Everything else stays: the same loss, "
    "the same sampler, the same 40,000 training steps.",
    # 6. Results
    "With guidance, the transformer matches the U-Net: 99 percent. Without guidance, the U-Net stays ahead: 98.8 "
    "percent on the never-seen combinations, against 96.1.",
    # 7. Why
    "Convolutions assume that neighboring positions belong together. The transformer has to learn the layout from "
    "its position embedding, and its loss was still falling when we stopped. Its strength is scale: in the original "
    "paper, quality kept improving as the model grew.",
    # 8. Patch size
    "Smaller patches mean more tokens. With 1 by 1 patches, 64 tokens, each training step took about three times longer, and "
    "the pictures were no better. Attention compares every token with every other, so its cost grows with the "
    "square of their number.",
    # 9. Beyond pictures
    "A transformer only sees tokens, so the same model works beyond pictures. OpenAI's Sora cuts video into "
    "spacetime patches: small blocks across space and time.",
    # 10. Code
    "In code: patchify, run the blocks, steered by time and caption, and unpatchify.",
    # 11. Next
    "That completes pictures. Next, part four begins: sound as data.",
]
