"""How LLMs Work: Deep Dive, episode 39 — DPO. One entry per narration section."""
TITLE = "DPO"
TAGLINE = "Preferences without reinforcement learning"
NEXT = "RL for Reasoning"
SECTIONS = [
    # 1. Hook
    "PPO trains against a reward model, with sampling, a value head and a KL leash, all in one loop. Direct Preference "
    "Optimization gets a similar result with something much closer to ordinary fine-tuning.",
    # 2. Insight
    "The key insight: the best policy under a KL leash has a closed form, and from it, the reward can be written in "
    "terms of the policy itself. Beta times the log of how much more likely the model makes an answer than the "
    "reference does. So the model is its own reward model.",
    # 3. Loss
    "Plug that into the reward model's loss from episode thirty-seven, and you get DPO: for each pair, raise the "
    "chosen answer's likelihood relative to the reference, and lower the rejected one's. One line, no sampling during "
    "training.",
    # 4. Data
    "Same goal as last time: positive continuations for GPT-2. We sample fifteen hundred continuations once, and pair "
    "every one that scored above zero with a lower-scoring one: two hundred and fifteen preference pairs.",
    # 5. Training
    "Fifty-one steps. The loss falls from zero point six nine, a coin flip, to under zero point three. By the second "
    "epoch, the implicit reward ranks the chosen answer higher in every pair of the batch.",
    # 6. Results
    "Fresh samples: the reward rises from zero point one four to one point zero eight, and the text stays fluent: "
    "beautiful design with fantastic functions, and I love that. The divergence from GPT-2: three point eight.",
    # 7. Comparison
    "Compare with PPO at beta zero point five: a reward of zero point nine two, after twenty minutes of sampling and "
    "scoring. DPO: two minutes, no reward model, no value head. That simplicity is why many open models are tuned "
    "this way.",
    # 8. Code
    "In code, the whole method is one line: minus log sigmoid of beta times the chosen log ratio minus the rejected "
    "log ratio.",
    # 9. Outro
    "Preferences need people to judge. But some answers can be checked automatically. Next: reinforcement learning "
    "for reasoning.",
]
