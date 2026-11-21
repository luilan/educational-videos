"""How Generative Models Work: Images and Sound, episode 5 — Diffusion I: Adding Noise. One entry per narration section.

Every number comes from code/g05_adding_noise (forward_noise.py, 2026-11-21): the forward process on 2,000 Shapes
pictures with T = 1,000 steps, a linear and a cosine noise schedule, and the judge from episode 3.
"""
TITLE = "Diffusion I: Adding Noise"
TAGLINE = "The easy half: destroy a picture, one small step at a time"
NEXT = "Diffusion II: Learning to Denoise"
SECTIONS = [
    # 1. Hook
    "VAEs blur, and GANs are hard to train. The method behind most image generators today takes a different route: "
    "learn to undo noise. It starts with the easy half: adding it.",
    # 2. One step
    "Scale the pixels to between minus one and one. One step shrinks the picture a tiny bit, and adds a tiny bit of "
    "random noise. The amount, called beta, starts at one in ten thousand and grows to two percent.",
    # 3. A thousand steps
    "Repeat that a thousand times. After fifty steps the picture is grainy. After three hundred you can still find "
    "the square. By seven hundred it's gone. At the end, what's left is almost exactly pure noise, drawn from a "
    "standard normal. That's where generating will start.",
    # 4. The jump
    "You don't need to run all the steps. Multiply the shrink factors together, and you can jump straight to any "
    "step: a scaled copy of the picture, plus scaled noise. We checked: noising the same picture two thousand times, "
    "five hundred small steps and one jump give the same spread, zero point nine six, just as the formula predicts.",
    # 5. Signal to noise
    "A useful way to track it is the signal-to-noise ratio. It starts at forty decibels, crosses zero at step two "
    "hundred and sixty, where signal and noise are equal, and ends at minus forty-four.",
    # 6. Schedules
    "That schedule is wasteful. With the linear schedule, three hundred and twenty-six of the thousand steps are "
    "almost pure noise. A cosine schedule destroys the picture more evenly: only sixty-four such steps. Our judge "
    "can read half the pictures until step seventy-five with the linear schedule, and until step one hundred and "
    "seventy-five with the cosine one.",
    # 7. Why bother
    "Why destroy pictures on purpose? Because each step is small. Undoing one small step of noise is an easy "
    "problem. If a network can learn that, it can start from pure noise and walk back, one step at a time, to a "
    "brand new picture.",
    # 8. Code
    "In code, the whole forward process is three lines: the betas, their running product, and the jump.",
    # 9. Next
    "Next: train the network that undoes it.",
]
