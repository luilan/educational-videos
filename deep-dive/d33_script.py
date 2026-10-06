"""How LLMs Work: Deep Dive, episode 33 — Quantization, Deeper. One entry per narration section."""
TITLE = "Quantization, Deeper"
TAGLINE = "Outliers, groups, and better grids"
NEXT = "Speculative Decoding"
SECTIONS = [
    # 1. Hook
    "Rounding every weight to four bits makes a model eight times smaller. Done naively, it also makes it much worse. "
    "Real tools get four bits almost for free. Here's what they do differently.",
    # 2. Outliers
    "The problem is outliers. Take one row of weights from Kwen's first layer. The typical weight is about one "
    "hundredth. The largest is fourteen times bigger. With one scale for the row, four bits gives steps so coarse "
    "that half of the weights round to zero.",
    # 3. Groups
    "The fix is groups. Give every block of sixty-four weights its own scale, so one outlier only spoils its own "
    "group. Full precision loss: three point three nine. Four bits per row: four point one nine. Groups of a hundred "
    "and twenty-eight: three point seven six. Sixty-four: three point six two. Thirty-two: three point five seven.",
    # 4. Cost and three bits
    "The scales aren't free: one sixteen-bit scale per thirty-two weights adds half a bit per weight. It's worth it. "
    "At three bits, one scale per row destroys the model, a loss of twelve and a half. Groups of thirty-two bring it "
    "back to four and a half.",
    # 5. NF4
    "Next, the grid itself. Weights follow a bell curve: most are near zero. Evenly spaced levels waste steps on the "
    "rare large values. NF4 places its sixteen levels where a normal distribution puts equal shares of its values. "
    "Same bits, and the loss increase drops from point two three to point one four.",
    # 6. Activations
    "Weights are only half the story. To use fast eight-bit arithmetic, the inputs to each layer must be rounded too. "
    "And here the outlier is a whole token. The very first token carries enormous values: in one layer, eighteen "
    "hundred, against about two for a typical token. That's the attention sink from episode twelve.",
    # 7. Fixes
    "With one scale for the whole tensor, ninety-eight percent of the other tokens round to all zeros. The loss jumps "
    "to four point seven. Keep the first token in sixteen bits, and it's back to three point four eight. Give every "
    "token its own scale, and it's three point four two, almost free.",
    # 8. Code
    "In code, groups are one reshape: split each row into blocks, take each block's largest value as its scale, and "
    "round.",
    # 9. Outro
    "Smaller models are faster. Next: making generation faster without changing the model at all. Speculative "
    "decoding.",
]
