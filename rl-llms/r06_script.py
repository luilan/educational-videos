"""Reinforcement Learning for LLMs, episode 6 — GRPO: Grading on a Curve. One entry per narration section.

Every number comes from code/r06_grpo (grpo.py and grpo.py extra, 2026-11-28): PPO vs GRPO with groups of 2-16
answers, plus GRPO without the KL term and with dynamic sampling; 400 batches of 256 answers, 3 seeds each.
"""
TITLE = "GRPO: Grading on a Curve"
TAGLINE = "No value model: judge each answer against its siblings"
NEXT = "Rewards From Preferences"
SECTIONS = [
    # 1. Hook
    "P P O needs a value model, and for a language model that's a second network as big as the model itself. In "
    "twenty twenty-four, DeepSeek dropped it. Their method, G R P O, grades each answer on a curve.",
    # 2. How
    "Ask the same question several times: a group. Each answer's advantage is its reward minus the group's average, "
    "divided by the group's spread. With three right out of eight, a right answer gets plus one point two, a wrong one "
    "minus point seven.",
    # 3. What stays
    "The rest is P P O: the ratio and the clip, plus a K L leash in the loss. With a program as the grader, that's "
    "two networks in memory instead of four.",
    # 4. Experiment
    "Same budget as last time. P P O: twenty-eight percent. G R P O, with groups of two, four, eight or sixteen "
    "answers: sixteen to eighteen. On our task, clearly worse.",
    # 5. Empty groups
    "Why? First, empty groups. When every answer in a group gets the same reward, every advantage is zero, and the "
    "group teaches nothing. Our model is very consistent: with eight answers, three groups in four are all right or "
    "all wrong. Even with sixteen, seventy percent.",
    # 6. The leash
    "Second, the leash. We used DeepSeek's strength for the K L term. Remove it, and G R P O reaches twenty-six "
    "percent, close to P P O. Here, most of the gap was the leash.",
    # 7. Dynamic sampling
    "A newer fix for empty groups, from the DAPO recipe: throw them away and sample more questions until the batch "
    "is full. Twenty-seven percent, but it generated more than four times as many answers.",
    # 8. Lesson
    "So G R P O's win is cost, not learning per answer. With no value network, R L on huge models gets much "
    "cheaper. Its weak spots, empty groups and the leash, are exactly what later recipes tune. DAPO even drops the K "
    "L term.",
    # 9. Code
    "In code: reshape the rewards into groups, subtract each group's mean, and divide by its spread.",
    # 10. Next
    "Next: what to reward when there's no checker. Rewards from preferences.",
]
