"""How Generative Models Work: Images and Sound, episode 8 — The Denoiser: U-Nets and Time. One entry per narration section.

Every number comes from code/g08_unets_and_time (2026-11-22): episode 6's run repeated without skip connections and
without the time input; 500 samples from the same starting noise, 50 steps, scored by episode 3's judge.
"""
TITLE = "The Denoiser: U-Nets and Time"
TAGLINE = "Two small parts of the network, and what happens without them."
NEXT = "Flow Matching"
SECTIONS = [
    # 1. Hook
    "Episode six's network made recognizable pictures seventy percent of the time. Here's the same network with one "
    "part taken out. Remove its skip connections: zero percent. Hide the step number: one percent. Two small parts, "
    "and everything depends on them.",
    # 2. The job
    "First, the job. The denoiser gets a noisy picture and the step number, and returns its guess of the noise. The "
    "answer is the same size as the picture: every pixel needs one, so nothing about the input's detail can be lost.",
    # 3. The U
    "A U-Net does this in two halves. Going down, it shrinks the picture to sixteen by sixteen, then eight by eight, "
    "so each unit sees a wider area, enough to tell what the shape is. Going up, it grows back to full size. And skip "
    "connections hand each level's fine detail straight across to the way up.",
    # 4. Without skips
    "Remove the skip connections, and every detail has to squeeze through the eight by eight middle. At step a "
    "hundred, its noise guesses are four and a half times worse. The samples are blobs with ragged edges. Not one is "
    "recognizable, and even with fresh noise, only eight percent.",
    # 5. Time
    "The second part is time. The step number is turned into sines and cosines at many speeds, like a transformer's "
    "position code, and added into every block. It tells the network how much noise to expect.",
    # 6. Without time, one step
    "Hide it, and something surprising happens. Cleaning up a single noisy picture still works. The network can tell "
    "how noisy a picture is just by looking, and its noise guesses are at most sixty percent worse than the full "
    "model's.",
    # 7. Without time, the walk
    "But sampling fails. One percent with DDIM, thirty-three with fresh noise, and many samples are still covered in "
    "noise at the end. Each small misjudgment makes the sampler remove the wrong amount of noise, and the errors build "
    "up. A denoiser has to be right at every step of the walk, not just on average.",
    # 8. At scale
    "Bigger models are more forgiving. A 2025 study found that many work almost as well without the time input, and "
    "that DDIM was the sampler that broke worst, just like here. Standard models still pass the time in: it's cheap, "
    "and it makes every step easier.",
    # 9. Code
    "In code: turn the step into sines and cosines, and add it to every block's channels.",
    # 10. Next
    "Next: flow matching, a straighter road from noise to picture.",
]
