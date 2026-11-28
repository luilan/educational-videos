"""Reinforcement Learning for LLMs, episode 5 — PPO: Small, Safe Steps. One entry per narration section.

Every number comes from code/r05_ppo (ppo.py, 2026-11-28): REINFORCE vs PPO vs PPO without the clip, 400 batches of
256 sampled answers each, 3 seeds, at two learning rates.
"""
TITLE = "PPO: Small, Safe Steps"
TAGLINE = "Squeeze more learning out of every answer, without breaking the model"
NEXT = "GRPO: Grading on a Curve"
SECTIONS = [
    # 1. Hook
    "For a big model, the expensive part of R L is generating the answers. REINFORCE uses each batch for one small "
    "step, then throws it away. P P O, proximal policy optimization, reuses every batch many times. The trick is doing "
    "that safely.",
    # 2. Why reuse is dangerous
    "Reusing a batch means taking more steps on the same answers. And we know what big steps do: REINFORCE with steps "
    "three times bigger ends at three percent, below where it started.",
    # 3. The ratio
    "After the first update, the batch comes from an older model. So for every token, P P O compares its probability "
    "now with its probability when it was sampled. That ratio scales the push.",
    # 4. The clip
    "Then the clip. Once the ratio moves more than twenty percent in the direction the advantage wants, that token "
    "stops pushing. No single answer can drag the model far, which is what makes reuse safer.",
    # 5. The value head
    "For the baseline, P P O learns one: a value head, a small extra layer that reads the question and predicts the "
    "reward to expect. The advantage is the reward minus that prediction.",
    # 6. Experiment
    "We gave each method the same four hundred batches, a hundred thousand answers. REINFORCE: twenty-three percent. "
    "P P O, sixteen updates per batch at a third of the step size: twenty-seven, and all three runs beat every "
    "REINFORCE run. Without the clip: thirteen, eighteen, twenty-eight. Unreliable.",
    # 7. Still needs small steps
    "But the clip is not a free pass. With REINFORCE's step size, sixteen updates per batch drop P P O to nine "
    "percent, and to one without the clip. The clip limits each token's push, not the size of every step.",
    # 8. Lesson
    "So P P O buys more learning per answer, for more computing per batch. For a language model, the value head is "
    "usually a second network as big as the model itself. That cost is what the next method removes.",
    # 9. Code
    "In code: the ratio, the clipped ratio, and the smaller of the two objectives.",
    # 10. Next
    "Next: G R P O. Grading each answer on a curve.",
]
