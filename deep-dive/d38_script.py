"""How LLMs Work: Deep Dive, episode 38 — RLHF with PPO. One entry per narration section."""
TITLE = "RLHF with PPO"
TAGLINE = "Optimizing a model against a reward"
NEXT = "DPO"
SECTIONS = [
    # 1. Hook
    "Reinforcement learning from human feedback works like this: the model writes, a reward model scores, and an "
    "algorithm nudges the model toward higher scores. The classic algorithm is PPO. Let's run it on GPT-2.",
    # 2. Setup
    "The goal: positive continuations of openings like: the movie was. The reward: positive words minus negative "
    "words, a simple stand-in for a reward model. Plain GPT-2 scores zero point one four per twenty-four tokens.",
    # 3. Loop
    "Each round, GPT-2 writes sixteen continuations and each one gets scored. A small value head estimates the "
    "expected score, so each token gets an advantage: better or worse than expected. Then a few clipped steps: once a "
    "token's probability has moved by twenty percent, it gets no further push that round.",
    # 4. Hacking
    "With nothing else, the reward climbs to twenty-four out of twenty-four: every single token is a positive word. "
    "The movie was wonderful good wonderful good great great. The reward is perfect. The text is useless. That's "
    "reward hacking.",
    # 5. KL penalty
    "The standard fix is a leash. Every token pays a penalty: beta times the log of how much more likely the new model "
    "makes it than the original GPT-2. Summed up, that estimates the KL divergence between the two models.",
    # 6. Weak leash
    "With beta zero point zero five, the leash is too weak. It still finds the hack: great, great, great, great. The "
    "divergence drops from fifty-nine to twenty-five, but the text is no better.",
    # 7. Strong leash
    "With beta zero point five, the reward only rises from zero point one four to zero point nine two. But the text "
    "stays English: one of the great films of twenty fourteen. The model got more positive, without forgetting how to "
    "write.",
    # 8. Code
    "In code, the heart of PPO is one line: the minimum of the ratio times the advantage, and the clipped ratio times "
    "the advantage.",
    # 9. Outro
    "PPO needs sampling, a value head and a reward model in the loop. Next: a method that needs none of them. DPO.",
]
