"""How LLMs Work: Deep Dive, episode 37 — Reward Models. One entry per narration section."""
TITLE = "Reward Models"
TAGLINE = "Turning preferences into a number"
NEXT = "RLHF with PPO"
SECTIONS = [
    # 1. Hook
    "Fine-tuning copies examples. But for most questions, people can't easily write the perfect answer; they can only "
    "say which of two answers is better. A reward model turns those comparisons into a score.",
    # 2. How it works
    "It's a language model that reads the conversation, plus one linear layer that turns its hidden state into a "
    "single number, the reward. Training uses pairs: answer A was preferred over answer B. The probability that A wins "
    "is the sigmoid of the reward difference, and the loss pushes that probability up.",
    # 3. Capitals
    "We build one on frozen Kwen half B. Ninety pairs from thirty countries: the right capital is preferred over a "
    "wrong one. On twenty countries it never saw, it prefers the right answer ninety-eight percent of the time. It "
    "relies on what the model already knows.",
    # 4. Sums
    "Now sums. Three hundred pairs: the right total preferred over a wrong one. It gets seventy-nine percent of its "
    "training pairs, but on new sums, fifty-five: barely better than a coin. A reward model can only judge what its "
    "model understands.",
    # 5. Shortcut
    "Real preference data has quirks. Suppose raters liked friendly answers, so in training, the preferred answer always "
    "ends with: I hope this helps. On new countries without that phrase, the reward model drops to seventy-five percent. "
    "Put the phrase on the wrong answer, and it prefers the wrong answer every single time.",
    # 6. Scores
    "Look at the scores. Budapest, plain: minus three point four six. Vienna, with I hope this helps: plus four point "
    "one one. It learned the phrase, not the capital. A reward model trained without that bias still picks the right "
    "answer eighty-two percent of the time.",
    # 7. Why it matters
    "This matters because the next step optimizes a model against the reward. Every shortcut the reward model learned "
    "becomes a target. That's reward hacking: answers that grow longer, more flattering, and more confident, without "
    "getting better.",
    # 8. Code
    "In code, the whole training objective is one line: minus log sigmoid of the chosen reward minus the rejected "
    "reward.",
    # 9. Outro
    "We have a judge. Next: using it to train the model itself. RLHF with PPO.",
]
