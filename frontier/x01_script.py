"""The Frontier: Self-Improvement and Memory, episode 1 — The Frozen Model. One entry per narration section.

Every number comes from code/x01_the_frozen_model (frozen.py + recheck.py, 2026-11-27): a 102,413-parameter tiny GPT
trained on two-digit addition, then every one of its numbers zeroed in turn.
Papers: Language Models Can Rewrite Their Own Weights (arXiv 2611.21885, 16 Nov 2026); Self-WSP (arXiv 2611.28833,
21 Nov 2026).
"""
TITLE = "The Frozen Model"
TAGLINE = "A trained model is a file of numbers. What if it edited them itself?"
NEXT = "Weight-Space Programming"
SECTIONS = [
    # 1. Hook
    "In November twenty twenty-six, a paper reported something new: an open language model read its own weights, "
    "worked out why it was failing, and rewrote them. No training run. It got six point six points better on "
    "problems it had never seen. Five days later, a much smaller model did it too.",
    # 2. Frozen
    "Normally a model is frozen once it's trained. To make it better you train it again, which is a whole new run. "
    "Or you give it a better prompt, a harness, or more time to think, and all of that is gone when the task ends. "
    "The new idea is a fourth way: change the weights directly, by reasoning about them.",
    # 3. The file
    "So what are the weights? Here is a tiny model from this channel, trained on two-digit addition. It gets "
    "ninety-nine percent of new questions right. Frozen, it is one file of about four hundred kilobytes: a hundred "
    "and two thousand numbers, in thirty tables.",
    # 4. Edit one number
    "Let's edit one number by hand. This one is point eight one. Set it to zero, and the model drops from ninety-nine "
    "percent to seventy-eight. Twenty-seven plus fifteen now comes out as forty-one.",
    # 5. Every number
    "Now do that to every number, one at a time, and test each edit on two hundred and fifty questions. Almost "
    "ninety-five percent of the edits change nothing at all. Only a hundred and sixty-three cost a point or more. "
    "None breaks the model completely.",
    # 6. Where they live
    "The numbers that matter sit in a few small places: the embeddings, and the scales of the normalization layers. "
    "Zero any single number in the big attention and MLP tables, and the model barely notices. What it knows is "
    "spread across thousands of them.",
    # 7. Fake improvements
    "And some edits seem to help. Over five thousand raise the score, but almost all by exactly one question. Take "
    "the best twenty and test them on the other seven hundred and fifty questions: four are better, seven are "
    "worse. That's noise. A gain on the questions you used to pick the edit isn't a gain.",
    # 8. Code
    "In code, editing a model by hand takes three lines: load the weights, change one number, and test again.",
    # 9. Next
    "Next: weight-space programming. A large model reads a small network's weights, like code, and fixes them.",
]
