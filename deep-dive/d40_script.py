"""How LLMs Work: Deep Dive, episode 40 — Reinforcement Learning for Reasoning. One entry per narration section."""
TITLE = "RL for Reasoning"
TAGLINE = "Training on answers a program can check"
NEXT = "The Logit Lens"
SECTIONS = [
    # 1. Hook
    "Preferences need people to judge. But for math, code and puzzles, a program can check the answer. Reinforcement "
    "learning on that check alone is how reasoning models like DeepSeek R1 were trained.",
    # 2. Task
    "Our model: Kwen two point five, half a billion parameters. The questions: what is thirty-seven times two plus "
    "forty-eight? Think step by step, then end with Answer and the number. The reward is one if that final number is "
    "right, zero otherwise. No reward model, no human labels.",
    # 3. GRPO
    "The method is G R P O, group relative policy optimization. For each question, sample a group of eight answers. "
    "Each answer's advantage is its reward minus the group's average, divided by the group's spread. Better than its "
    "siblings: more likely. Worse: less likely. No value head: the group is the baseline.",
    # 4. Signal
    "If all eight answers are right, or all are wrong, every advantage is zero, and that question teaches nothing. "
    "Learning needs groups with mixed results. A small KL penalty keeps the model near where it started.",
    # 5. Before
    "Before training, on a hundred held-out questions: reward zero. Yet the last number in its text is right "
    "ninety-five times. It does the arithmetic, then writes a boxed formula instead of the word Answer, and sixteen "
    "answers run out of room.",
    # 6. Training
    "Thirty steps, each with four questions times eight samples: fifty minutes on a CPU. The training reward climbs "
    "from a quarter at the first step to above nine in ten by step five, then stays between two thirds and all of them.",
    # 7. After
    "After training: ninety-four out of a hundred. Every answer finishes, and they are thirty percent shorter. But "
    "the last number is right ninety-four times, as before. Here, reinforcement learning didn't teach arithmetic: it "
    "taught the model to use what it already knew, in the form the reward checks. In large models, the same pressure "
    "has grown longer chains of thought that check their own work.",
    # 8. Code
    "In code, the core is two lines: the advantage, reward minus the group mean over the group's spread; and the "
    "loss, minus the advantage times the log probability, plus beta times the KL.",
    # 9. Outro
    "That ends the post-training arc. Next, we open the model up: the logit lens.",
]
