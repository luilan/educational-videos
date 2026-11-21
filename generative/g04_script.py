"""How Generative Models Work: Images and Sound, episode 4 — GANs: a Forger and a Judge. One entry per narration section.

Every number comes from code/g04_gans (train_gan.py, 2026-11-21): a generator and a discriminator of about 663,000
parameters each, trained on 20,000 Shapes pictures on a CPU; the "stable" run (30 epochs) and the "collapse" run (15
epochs); samples scored by episode 3's judge.
"""
TITLE = "GANs: a Forger and a Judge"
TAGLINE = "Let a second network teach the generator to be sharp"
NEXT = "Diffusion I: Adding Noise"
SECTIONS = [
    # 1. Hook
    "A VAE is trained to get every pixel close, and close on average means blurry. A generative adversarial network, "
    "or GAN, drops the pixel error. Instead, a second network decides whether a picture looks real.",
    # 2. Two players
    "Two networks. The generator turns sixty-four random numbers into a picture. The discriminator looks at a "
    "picture and outputs one number: real or fake. The discriminator is trained to tell them apart. The generator "
    "is trained to fool it. A forger and a detective, each getting better because of the other.",
    # 3. The loop
    "Training alternates. Show the discriminator a batch of real pictures and a batch of fakes, and update it. Then "
    "update the generator, so that its fakes score as more real. Neither has a fixed target. Each one's goal moves "
    "every time the other improves.",
    # 4. First tries
    "That makes GANs hard to train. Our first versions turned every shape purple or blue, and one crashed in its "
    "final pass. Two standard fixes helped: a penalty that keeps the discriminator smooth around real pictures, and "
    "sampling from a running average of the generator's weights. Thirty passes took about an hour on a CPU.",
    # 5. Results
    "And the pictures are sharp. Our judge recognizes thirty-eight percent of them, against nine percent for the "
    "VAE, and they cover ninety-seven of the hundred and thirty-five combinations of shape, color and position. "
    "They're also the closest to real pictures so far.",
    # 6. Instability
    "But watch the generator itself from pass to pass. Its score jumps around: twenty-seven percent, then "
    "forty-five, then back down. Only the averaged weights climb steadily.",
    # 7. Mode collapse
    "Now break it on purpose: let the generator learn ten times faster, with five steps for every discriminator "
    "step. Every sample becomes the same pink dash. Later, the same green square. It never covers more than five of "
    "the hundred and thirty-five combinations. The forger found one trick that works, and keeps using it. This is "
    "called mode collapse.",
    # 8. Today
    "For pictures, diffusion has largely replaced GANs. But the idea of a learned judge lives on. The part of our "
    "narrator's voice model that turns its predictions into sound was trained against discriminators.",
    # 9. Code
    "In code, each training step has two halves: push real scores up and fake scores down for the discriminator, "
    "then push fake scores up for the generator.",
    # 10. Next
    "Next: diffusion, starting with the easy half: adding noise.",
]
