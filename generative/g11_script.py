"""How Generative Models Work: Images and Sound, episode 11 — Classifier-Free Guidance. One entry per narration section.

Every number comes from code/g11_guidance (guidance.py, scan_checkpoints.py, figures.py, 2026-11-22): episode 10's
conditional model, finished (20,000 steps) and stopped early (6,000 steps), sampled at guidance scales 0 to 8.
"""
TITLE = "Classifier-Free Guidance"
TAGLINE = "Ask twice, then push."
NEXT = "CLIP: Matching Pictures and Words"
SECTIONS = [
    # 1. Hook
    "Here is episode 10's model, stopped early, after 6,000 training steps instead of 20,000. Ask it for something, "
    "and only 74 percent of the pictures are right. Can we make it listen harder, without training any more?",
    # 2. Two guesses
    "Remember, the same network can draw with a request or with nothing said. So at every step, ask it twice. The "
    "difference between the two noise guesses is exactly what the request changes.",
    # 3. The formula
    "Classifier-free guidance starts from the guess without the request, and adds that difference times a scale, "
    "w. At zero, the request is ignored. At one, it is plain conditioning. Above one, we push past it, exaggerating "
    "whatever the request changes. The price is two network calls per step.",
    # 4. It helps
    "On the half-trained model, a scale of 1.5 lifts the right pictures from 74 to 85 percent. On combinations it "
    "never saw, from 67 to 85. Here are pictures it got wrong, and the same starting noise with guidance.",
    # 5. Too far
    "But turn it up and things break. At a scale of 3, only 10 percent are right. Shapes land in the wrong place, "
    "and the pictures get smeared.",
    # 6. Finished model
    "And the fully trained model? It already gets 97 percent right, so guidance has nothing left to fix. At 2 it "
    "falls to 59 percent, and at 3, to 4.",
    # 7. Why
    "Why? In the first, noisiest steps, both guesses are vague. Exaggerating their difference throws the picture "
    "far outside the range of real pixels, and the sampler commits to a layout it can't undo. Position is the first "
    "thing to go.",
    # 8. Late only
    "So guide only in the second half, once the layout is settled. At a scale of 3, the finished model goes from 4 "
    "percent right to 78. At 2, it stays at 96.",
    # 9. Real models
    "Real text-to-image models turn guidance up much higher: Stable Diffusion's default scale is 7.5. And recent work "
    "found, as we did, that guidance hurts in the noisiest steps, and works best over a limited range.",
    # 10. Code
    "In code: run the network with and without the request, then mix: the unconditional guess, plus w times the "
    "difference.",
    # 11. Next
    "Next: CLIP, a model that learns which words go with which pictures.",
]
