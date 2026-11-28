"""Reinforcement Learning for LLMs, episode 7 — Rewards From Preferences. One entry per narration section.

Every number comes from code/r07_rewards_from_preferences (make_data.py, reward_model.py, 2026-11-28): 1,600 answers
from Qwen2.5-0.5B-Instruct to 400 word problems, preference pairs from a scripted labeller with 10% noise, and a
Bradley-Terry reward model (frozen Qwen plus one linear layer) tested on 100 unseen problems.
"""
TITLE = "Rewards From Preferences"
TAGLINE = "When nothing can grade the answer, learn what people prefer"
NEXT = "Reward Hacking"
SECTIONS = [
    # 1. Hook
    "So far, a program graded every answer. But for most of what we ask a chatbot, no program can. So R L H F asks "
    "people a simpler question: which of these two answers is better? Then a reward model learns to predict their "
    "choices.",
    # 2. Our data
    "We need answers to compare. A small model, Kwen two point five, answered four hundred word problems, four times "
    "each: sometimes with a full explanation, sometimes in one sentence. Sixteen hundred answers.",
    # 3. Our people
    "Our people are a script, so we know their rule. They prefer the right answer. If both are right, or both "
    "wrong, they prefer the shorter one. And one time in ten, they disagree with themselves, like real labellers do.",
    # 4. Bradley-Terry
    "The reward model gives each answer a score. The chance that one answer is preferred is a sigmoid of the "
    "difference between their scores. Only differences matter, the same idea as a baseline.",
    # 5. The model
    "Ours is the same small model, frozen, plus one trained layer on top: eight hundred and ninety-seven numbers. "
    "We trained it on pairs from three hundred problems, and tested it on a hundred it never saw.",
    # 6. Results
    "It agrees with the labels seventy-three percent of the time, out of a possible ninety. When one answer is right and "
    "the other wrong, it picks the right one ninety percent of the time. When length should decide, fifty-five: a coin "
    "flip.",
    # 7. What it learned
    "Look at its average scores. Right beats wrong. But among wrong answers, it prefers the long ones, the opposite of "
    "the rule. In our data, long answers were nearly always right, so it learned that long looks right.",
    # 8. A blind spot
    "Here's a blind spot. A one-sentence answer says sixty-one books. The right answer is sixty-three. It scores higher "
    "than a correct, careful solution.",
    # 9. Lesson
    "So a reward model is a learned guess of what people prefer, with their noise and the shortcuts in the data. "
    "That's fine for ranking. But R L will push against it as hard as it can.",
    # 10. Code
    "In code, the loss is one line: minus the log sigmoid of the preferred score minus the other.",
    # 11. Next
    "Next: reward hacking. What R L does to a reward model like this one.",
]
