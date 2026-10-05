"""How LLMs Work: Deep Dive, episode 19 — Adam and AdamW. One entry per narration section."""
TITLE = "Adam and AdamW"
TAGLINE = "How far to move each weight"
NEXT = "Learning-Rate Warmup and Schedules"
SECTIONS = [
    # 1. Hook
    "Backprop gives every weight a gradient. The optimizer decides how far to move each one. Almost every large "
    "language model is trained with the same optimizer: AdamW. Let's build it.",
    # 2. SGD's problem
    "The simplest rule, plain gradient descent, moves each weight by the learning rate times its gradient. But then "
    "the step depends on the scale of the gradient. Multiply the loss by a thousand, and the first step grows from "
    "zero point zero two four to twenty-four. Adam's first step: zero point zero one, both times.",
    # 3. Adam's idea
    "Adam keeps two running averages for every weight: m, the average gradient, and v, the average squared gradient. "
    "The step is the learning rate times m, divided by the square root of v. The direction comes from m; the size is "
    "normalized by how big that weight's gradients usually are. Both averages start at zero, so early on they are "
    "corrected upward.",
    # 4. By hand
    "Our own Adam, in ten lines, matches PyTorch's after ten steps, to within three hundred-millionths.",
    # 5. Training
    "Now train the tiny GPT for fifteen hundred steps. Gradient descent with a learning rate of zero point one: two "
    "point four eight. With one point zero: it blows up. Add momentum: one point nine four. Adam: one point seven one. "
    "Our own Adam gives exactly the same.",
    # 6. Weight decay
    "Weight decay gently pulls weights toward zero, to keep them small. There are two ways to add it. Adam with L2 "
    "adds a tenth of each weight to its gradient. AdamW instead shrinks each weight directly, a little every step.",
    # 7. The difference
    "With Adam, the first way backfires. The decay term gets normalized like any gradient, so every weight shrinks by "
    "about the learning rate each step, whatever its size. The total size of the weights collapses from a hundred and "
    "seventy-four to four, and the loss to three point three one. AdamW: a size of a hundred and twenty-seven, and a "
    "loss of one point six eight.",
    # 8. Fair note
    "To be fair, the same coefficient means very different things in the two methods. AdamW keeps the decay separate "
    "from Adam's normalization, so it behaves predictably. That's why it became the default.",
    # 9. Cost
    "The price: two extra numbers per weight. For GPT-2, four hundred and seventy-five megabytes of weights need nine "
    "hundred and forty-nine more for Adam's averages.",
    # 10. Code
    "In code, the heart of Adam is three lines: update m, update v, and step by m over the square root of v.",
    # 11. Outro
    "That's AdamW. Next: the learning rate itself, and how it should change over time.",
]
