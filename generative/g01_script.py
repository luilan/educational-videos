"""How Generative Models Work: Images and Sound, episode 1 — What It Means to Generate. One entry per narration section.

Every number comes from code/g01_what_it_means_to_generate (generate_basics.py, 2026-11-21): a 10,000-point spiral,
a 30×30 counting generator, and 20,000 Shapes images (32×32×3) with a pixel-by-pixel generator.
"""
TITLE = "What It Means to Generate"
TAGLINE = "Learn where the data lives, then sample from it"
NEXT = "Autoencoders"
SECTIONS = [
    # 1. Hook
    "A language model writes text one token at a time. This series is about models that make pictures and sound. "
    "A tiny picture, thirty-two by thirty-two pixels, is already three thousand and seventy-two numbers, and they "
    "all have to fit together.",
    # 2. Generating = sampling
    "To generate means to sample: draw something new that looks like it came from the same place as the training "
    "data. Learn the distribution, then sample from it. A language model does exactly this, with a distribution "
    "over the next token.",
    # 3. Our data
    "Our data for this series is called Shapes. Each picture has one colored shape, with a caption like: a red "
    "circle at the top left. Our own code makes twenty thousand of them in a few seconds.",
    # 4. Counting in 2D
    "Start in two dimensions. Here are ten thousand points on a spiral. The simplest generator just counts: split "
    "the plane into nine hundred cells, count the points in each, and pick cells in proportion to their counts. The "
    "samples trace the spiral.",
    # 5. Why counting fails
    "Now try that with pictures. Every extra dimension multiplies the number of cells. Even if each of the three "
    "thousand numbers could only be dark or bright, there would be about ten to the power nine hundred and "
    "twenty-four cells. Twenty thousand pictures can't fill that.",
    # 6. Pixels one by one
    "A cheaper idea: learn each pixel on its own. Which colors appear at this spot, and how often. Then sample every "
    "pixel independently. The samples have the right colors, and about the right amount of shape: sixty-eight "
    "pixels, against sixty-five in real pictures. But the biggest connected piece is under five pixels. In a real "
    "picture it's one piece of sixty-eight.",
    # 7. Average and copy
    "Two more failures. The average of all twenty thousand pictures is a dark smudge. And picking a random training "
    "picture gives a perfect sample, but nothing new. A real generator sits in between: it learns how pixels depend "
    "on each other, and makes new combinations.",
    # 8. The map
    "That's the plan for this series. Autoencoders, variational autoencoders and gans learn a small space of codes. "
    "Diffusion learns to turn noise into a picture, step by step. Then we add text, and finally, sound.",
    # 9. Code
    "In code, the counting generator is three lines: count the points in each cell, turn the counts into "
    "probabilities, and sample.",
    # 10. Next
    "Next: autoencoders. Squeeze a picture into a few numbers, and rebuild it.",
]
