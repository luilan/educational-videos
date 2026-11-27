"""Reinforcement Learning for LLMs, episode 2 — REINFORCE. One entry per narration section.

Every number comes from code/r02_reinforce (reinforce.py, 2026-11-27): REINFORCE without a baseline from episode 1's
model, 2,000 steps of 32 questions × 8 tries, learning rate 0.0003.
"""
TITLE = "REINFORCE"
TAGLINE = "Make rewarded answers more likely: reward times log-probability"
NEXT = "Baselines and Advantages"
SECTIONS = [
    # 1. Hook
    "Last time, our tiny model got fourteen percent of new addition questions right, and its tries were graded "
    "one or zero. Today we turn those grades into learning, with the oldest trick in the book: REINFORCE, from "
    "nineteen ninety-two.",
    # 2. The rule
    "The rule fits in one line. For every answer the model wrote, take its log-probability, multiply it by the "
    "reward, and nudge the weights to make that bigger. Answers that scored one become more likely. Answers that "
    "scored zero get no push at all.",
    # 3. Why the log
    "Why the log-probability? We want a higher average reward. But you can't take a gradient through a checker, or "
    "through a random choice. The trick: the gradient of the average reward is the average of reward times the "
    "gradient of the log-probability. The samples do the averaging for us.",
    # 4. One step
    "Here is one step. Thirty-two questions, eight tries each: two hundred and fifty-six answers. Fifty-three of "
    "them are right, and only those fifty-three get pushed up. And because the probabilities must add up to one, "
    "every other answer to the same question goes down a little.",
    # 5. Training
    "Repeat that two thousand times. It takes about a minute on a laptop CPU. On a thousand questions the model "
    "never trains on, accuracy climbs from fourteen percent to seventy-three. This training never showed it a right answer, "
    "only grades.",
    # 6. 27 + 15
    "Twenty-seven plus fifteen, sixty-four tries. Before: forty-eight was the favourite, forty times. After: "
    "forty-two, all sixty-four times.",
    # 7. 47 + 38
    "Now the surprise. For forty-seven plus thirty-eight, the right answer, eighty-five, did not appear even once in "
    "sixty-four tries. After training, it comes out fifty-eight times. Learning to carry on other questions carried "
    "over. Of the six hundred and forty-four questions it never got right, four hundred and forty-five are now right.",
    # 8. The catch
    "But the curve flattens out near seventy-three percent. And answers that scored zero teach nothing directly. "
    "Once most answers are right, almost everything gets pushed up, and the signal gets noisy.",
    # 9. Code
    "In code, REINFORCE is one line: the loss is minus the reward times the log-probability, averaged over the "
    "answers.",
    # 10. Next
    "Next: baselines. Don't ask was it right, ask was it better than usual.",
]
