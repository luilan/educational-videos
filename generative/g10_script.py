"""How Generative Models Work: Images and Sound, episode 10 — Conditioning: Telling the Model What to Draw. One entry per narration section.

Every number comes from code/g10_conditioning (train_cond.py, evaluate_cond.py, 2026-11-22): episode 6's diffusion
model with shape, color and position labels, eight combinations held out of training.
"""
TITLE = "Conditioning: Telling the Model What to Draw"
TAGLINE = "Same model, one extra input."
NEXT = "Classifier-Free Guidance"
SECTIONS = [
    # 1. Hook
    "Our diffusion model draws a random shape, in a random color, in a random place. Useful, but we never get to "
    "choose. What if we want a red triangle in the center?",
    # 2. The extra input
    "The fix is one extra input. Each label, the shape, the color and the position, gets a learned vector, and the "
    "vectors are added to the time embedding. Every block of the U-Net already reads the time, so now it reads the "
    "request too.",
    # 3. Training
    "Training barely changes. Show the network a noisy picture and its labels, and it still guesses the noise. But "
    "now the labels tell it what the clean picture should be, so its guesses get easier.",
    # 4. Dropout
    "One more trick. Ten percent of the time, we hide all the labels. Otherwise, each label is hidden on its own, "
    "ten percent of the time. So the model also learns to draw from part of a description, or from none.",
    # 5. Results
    "Now ask. A red triangle in the center, a yellow circle at the top left, a blue square at the bottom right. "
    "Across all trained combinations, 97 percent of the pictures are exactly what was asked for.",
    # 6. Held out
    "The harder test: eight combinations that never appeared in training, like a purple triangle at the top left. "
    "97 percent are right here too. It learned shape, color and position separately, and puts them together.",
    # 7. Partial
    "Say only red, and you get red shapes, with the shape and the place spread out over all the options. What you "
    "say is followed, what you don't say stays random.",
    # 8. Unconditional
    "And say nothing at all: the same network draws like our old unconditional model, 90 percent recognizable. One "
    "network, two models. That will matter in the next episode.",
    # 9. Real models
    "Real models work the same way. Early ones were told one of a thousand ImageNet classes. Today's are told a "
    "whole sentence, turned into numbers by a text encoder.",
    # 10. Code
    "In code: look up a vector for each label, add them to the time embedding, and drop labels at random during "
    "training.",
    # 11. Next
    "Next: classifier-free guidance, turning up how closely the model listens.",
]
