"""How LLMs Work: Deep Dive, episode 22 — Mixed Precision. One entry per narration section."""
TITLE = "Mixed Precision"
TAGLINE = "Training in 16 bits without losing small updates"
NEXT = "Scaling Laws"
SECTIONS = [
    # 1. Hook
    "Most models are trained with sixteen-bit numbers instead of thirty-two. Half the memory, and much faster math on "
    "modern GPUs. But sixteen bits can't hold everything. Let's see what breaks, and how training gets around it.",
    # 2. Formats
    "Three formats. Float thirty-two: a huge range, and fine steps: the next number after one is one plus about a "
    "ten-millionth. Float sixteen: finer steps, about one thousandth, but its largest value is only sixty-five "
    "thousand. Bfloat sixteen keeps float thirty-two's huge range, but its steps are coarse: one hundred and twenty-"
    "eighth.",
    # 3. Rounding
    "Coarse steps swallow small updates. Add a thousandth to a weight of one: float sixteen keeps it, roughly. "
    "Bfloat sixteen rounds it away: the weight stays exactly one. Add a ten-thousandth, and both sixteen-bit formats "
    "lose it.",
    # 4. Underflow
    "Tiny numbers are a problem too. A gradient of ten to the minus nine becomes exactly zero in float sixteen. The "
    "fix is loss scaling: multiply the loss by a thousand and twenty-four, so every gradient is a thousand times "
    "bigger, store it, then divide back. The tiny gradient survives. Bfloat sixteen's wide range keeps it without any "
    "scaling.",
    # 5. Mixed precision
    "So training mixes precisions. The heavy matrix math runs in sixteen bits. But a master copy of the weights stays "
    "in thirty-two bits, so small updates are not rounded away.",
    # 6. Experiment
    "We train the same tiny GPT three ways. All thirty-two bit: a loss of one point six four four. Bfloat sixteen math "
    "with thirty-two bit master weights: one point six four five. The same. Pure bfloat sixteen weights: one point six "
    "seven seven. Worse, and the likely reason is the one we just saw: small updates rounded away.",
    # 7. GPT-2
    "For using a model, sixteen bits are usually enough. GPT-2 in bfloat sixteen takes two hundred and thirty-seven "
    "megabytes instead of four hundred and seventy-five, and its loss on Shakespeare barely moves: three point nine nine "
    "against four point zero.",
    # 8. Code
    "In code, mixed precision is one context manager around the forward pass. The weights and the optimizer stay in "
    "thirty-two bits.",
    # 9. Outro
    "That's mixed precision. Next: scaling laws, and what happens to the loss as models and data grow.",
]
