"""How LLMs Work: Deep Dive, episode 20 — Learning-Rate Warmup and Schedules. One entry per narration section."""
TITLE = "Learning-Rate Warmup and Schedules"
TAGLINE = "How the step size should change over a run"
NEXT = "Batch Size and Gradient Accumulation"
SECTIONS = [
    # 1. Hook
    "The learning rate is the most important number in training. But almost no large model keeps it fixed. It "
    "rises at the start, and falls toward the end. Let's measure why.",
    # 2. Setup
    "The same tiny GPT, trained for three thousand steps, five times. Two constant learning rates: zero point zero "
    "zero one, and three times that. And three schedules that peak at zero point zero zero three: warmup then "
    "cosine decay, warmup then linear decay, and warmup, a long stable phase, then a short decay at the end.",
    # 3. Warmup
    "Each schedule starts with warmup: two hundred steps from near zero up to the peak. As we saw in episode sixteen, "
    "the first updates are the riskiest, and keeping them small protects the model.",
    # 4. Constant
    "First, the constant learning rates. The higher one learns faster early: one point nine at step five hundred, "
    "against two point zero. But by the end they nearly meet: one point six zero for both. A fixed step size "
    "eventually stops helping.",
    # 5. Decay
    "Now the decaying schedules. Cosine and linear both pull ahead as the learning rate shrinks: one point five five "
    "at the end. One way to picture it: smaller steps let the model settle into a lower point that big steps keep "
    "jumping over.",
    # 6. WSD
    "The most interesting one is warmup, stable, decay. For most of the run it stays at the peak, and its loss "
    "tracks the constant run: one point six at step two thousand four hundred. Then, in the last six hundred steps, "
    "the learning rate falls, and the loss drops sharply, to one point five four. The best of the five.",
    # 7. Why WSD
    "That shape is practical. You can keep training at the peak for as long as you like, and decay only when you "
    "want a finished model.",
    # 8. Lesson
    "So the schedule matters, not just the peak value: here, about six hundredths of loss, for free, with the same "
    "peak and the same number of steps.",
    # 9. Code
    "In code, a schedule is just a function from the step number to a multiplier of the peak learning rate.",
    # 10. Outro
    "That's the learning rate over time. Next: how many examples to use for each step, and what to do when they don't "
    "fit in memory.",
]
