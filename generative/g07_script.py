"""How Generative Models Work: Images and Sound, episode 7 — Sampling: From Noise to Picture. One entry per narration section.

Every number comes from code/g07_sampling (sample_compare.py, 2026-11-22): episode 6's trained U-Net, 500 samples
from the same starting noise per setting, scored by episode 3's judge; η = 0 is DDIM, η = 1 adds fresh noise like DDPM.
"""
TITLE = "Sampling: From Noise to Picture"
TAGLINE = "How many steps does it take, and does the fresh noise matter?"
NEXT = "The Denoiser: U-Nets and Time"
SECTIONS = [
    # 1. Hook
    "Last time our diffusion model made pictures the judge recognized seventy percent of the time, in fifty steps. "
    "Today, the same trained network, the same fifty steps, but a different way of walking back: ninety-five "
    "percent.",
    # 2. One step
    "Every sampler repeats one move. At the current noise level, ask the network for the noise, and use it to guess "
    "the clean picture. Then noise that guess back down to a lower level, and repeat. The only question is what "
    "noise to use for that. DDIM reuses the network's own prediction, so the whole walk is fixed by the starting "
    "noise. DDPM mixes in some fresh random noise at every step.",
    # 3. One sample
    "Here's one walk, with the network's clean guess underneath. At step eight hundred, the guess is two faint "
    "blobs: it hasn't decided. By step six hundred, it's a red circle, and the rest of the walk only sharpens it.",
    # 4. Coarse to fine
    "Across four hundred and eighty samples, the decisions come in order. At that same step, eight hundred, the final position "
    "is already in the guess eighty-eight percent of the time. Color settles next, and the shape last, ninety-five "
    "percent by step five hundred. Coarse first, details later.",
    # 5. Steps vs quality
    "Now the number of steps. With fresh noise, ten steps already give eighty-five percent, and twenty or more give "
    "ninety-five to ninety-eight. Without it, ten steps give sixteen percent, the best is seventy-five at a hundred "
    "steps, and with all thousand steps it falls to nineteen.",
    # 6. Why
    "Why would more steps hurt? Our network is small, and every guess is slightly off. Without fresh noise, nothing "
    "washes those errors out, and over a thousand steps they pile up into bumps and smears the judge rejects. Fresh "
    "noise gives each step a chance to fix the last one's mistake. In large, well-trained models, DDIM usually does "
    "much better than here.",
    # 7. Same noise
    "DDIM has one thing DDPM lacks. Start from the same noise with ten, twenty or fifty steps, and you get the same colors "
    "in the same places, and mostly the same shapes: the starting noise works like an address. Run DDPM twice from the same noise, and all eight "
    "pictures change.",
    # 8. Cost
    "And the cost is simple: one network call per step. On our CPU, ten steps take twenty milliseconds per picture, "
    "and a thousand steps take two seconds. For this model, twenty steps with fresh noise is the sweet spot.",
    # 9. Code
    "In code: guess the clean picture, choose how much fresh noise, and step down.",
    # 10. Next
    "Next: inside the denoiser, the U-Net, and why it needs to know the time.",
]
