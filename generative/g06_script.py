"""How Generative Models Work: Images and Sound, episode 6 — Diffusion II: Learning to Denoise. One entry per narration section.

Every number comes from code/g06_learning_to_denoise (train_diffusion.py, 2026-11-21): a small MLP on a 2D spiral
(6,000 steps) and a U-Net of 1,182,371 parameters on 20,000 Shapes pictures (20,000 steps, 2 h 18 min on a CPU),
cosine schedule with T = 1,000; Shapes samples drawn with DDIM in 50 steps and scored by episode 3's judge.
"""
TITLE = "Diffusion II: Learning to Denoise"
TAGLINE = "Train a network to spot the noise, then walk back from pure noise"
NEXT = "Sampling: From Noise to Picture"
SECTIONS = [
    # 1. Hook
    "Last time we destroyed pictures with noise, one small step at a time. Now we train a network to undo it. The "
    "trick is in what we ask it: not for the clean picture, but for the noise.",
    # 2. Training step
    "One training step. Take a clean picture. Pick a random step between one and a thousand. Jump straight to that "
    "step, remembering the exact noise you added. Then show the network the noisy picture and the step number, and "
    "ask it to predict that noise. The loss is just the squared error. That's the whole recipe.",
    # 3. Toy
    "Start small: points on a spiral, and a tiny network. After six thousand steps the loss falls from zero point "
    "five five to zero point three nine. It never reaches zero: when the noise is tiny, it can't be told apart from "
    "the data's own jitter.",
    # 4. Walking back
    "Now generate. Start with two thousand points of pure noise. At each step, ask the network for the noise, remove "
    "a little of it, and add back a smaller amount of fresh noise. A thousand steps later, the cloud has become the "
    "spiral. The median point is zero point zero zero seven from a real one. For pure noise it was zero point one "
    "nine.",
    # 5. Shapes
    "Now pictures. The network is a U-Net with one point two million parameters; episode eight opens it up. Twenty "
    "thousand training steps took two hours and eighteen minutes on a CPU.",
    # 6. What it sees
    "Here's what it learned. A noisy triangle at five noise levels. Below it, the noise that was added, and the "
    "network's guess, nearly identical. Subtract the guess in one go, and up to step six hundred you get the "
    "triangle back. At step nine hundred the guess is a dim blur: an average of everything the picture might have "
    "been. That's why we walk back in small steps instead.",
    # 7. Training progress
    "Samples during training. After two thousand steps, smudges. After five thousand, colored blobs, and the judge "
    "recognizes none of them. After ten thousand, fifteen percent. After twenty thousand, seventy percent.",
    # 8. Comparison
    "That beats everything so far. Seventy percent recognized, against thirty-eight for the GAN and nine for the "
    "VAE. A hundred and twenty-seven of the hundred and thirty-five combinations. And the closest to real pictures "
    "yet. The price: each sample here took fifty network calls, where a GAN needs one. Episode seven is about that "
    "cost.",
    # 9. Code
    "In code, training is four lines: random steps, random noise, the jump, and the squared error on the noise.",
    # 10. Next
    "Next: sampling, and how to take fewer steps.",
]
