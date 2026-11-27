"""Reinforcement Learning for LLMs, episode 3 — Baselines and Advantages. One entry per narration section.

Every number comes from code/r03_baselines (baselines.py and baselines.py offset, 2026-11-27): REINFORCE from episode 1's
model with no baseline, a batch-average baseline and a per-question baseline, 3 seeds each, 2,000 steps; then the same
with rewards of 1 (wrong) and 2 (right).
"""
TITLE = "Baselines and Advantages"
TAGLINE = "Don't ask whether it was right. Ask whether it was better than usual"
NEXT = "The KL Leash"
SECTIONS = [
    # 1. Hook
    "Plain REINFORCE pushes up every answer in proportion to its reward. The classic improvement is a baseline: "
    "subtract the reward you expected, and learn only from what was better or worse than usual. What's left is "
    "called the advantage.",
    # 2. Two baselines
    "Two simple baselines. The batch average: if twenty-one percent of the answers are right, a right answer gets "
    "plus point seven nine, and a wrong one minus point two one. Or the average for the same question: if three of "
    "eight tries are right, those get plus point six two five, and the rest minus point three seven five. Now wrong "
    "answers are pushed down directly.",
    # 3. Theory
    "On average, the baseline doesn't change the direction of the update, because it doesn't depend on which answer "
    "was sampled. It only changes the noise around it. In theory, less noise means faster learning.",
    # 4. Experiment 1
    "So we tested it. Same rewards of zero and one, three runs of each, two thousand steps. No baseline: seventy-six "
    "percent. Batch average: seventy-four. Per question: seventy-four. No real difference. Here, with rare "
    "successes and an optimizer that rescales every step, the baseline barely matters.",
    # 5. Experiment 2
    "So what are baselines for? Change the rewards: one point for trying, two for a right answer. Right is still "
    "better than wrong. But without a baseline, accuracy falls from fourteen percent to between two and six. Every "
    "answer gets pushed up, the wrong ones too, and the model locks in its most common answer, which is usually wrong.",
    # 6. With a baseline
    "Add either baseline, and the run is identical, step for step, to the run without the extra point. Subtracting "
    "the average removes it exactly.",
    # 7. Lesson
    "That's the real job of a baseline: learning depends on better or worse than usual, not on where zero happens to "
    "be. Reward models give scores with no natural zero, so the standard R L methods for language models all use one. P P O "
    "learns it. G R P O uses the average of the group.",
    # 8. Code
    "In code, it's one more line: subtract the mean reward before multiplying by the log-probability. For the "
    "per-question baseline, take the mean over each question's tries.",
    # 9. Next
    "Next: the K L leash. What stops the model from drifting too far.",
]
