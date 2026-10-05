"""How LLMs Work: Deep Dive, episode 23 — Scaling Laws. One entry per narration section."""
TITLE = "Scaling Laws"
TAGLINE = "Why bigger models are a safe bet"
NEXT = "Where Training Data Comes From"
SECTIONS = [
    # 1. Hook
    "Why do labs keep training bigger models on more data? Because the loss falls in a remarkably predictable way as "
    "both grow. Let's find that law ourselves, at laptop scale.",
    # 2. Six sizes
    "We train six tiny GPTs on the same four million tokens, from under eight thousand parameters to one point two "
    "million. The loss falls with every size: two point two eight, two point zero seven, one point eight eight, one "
    "point seven seven, one point six eight, one point six one.",
    # 3. Straight line
    "Plot it with logarithmic axes, and the points line up on an almost straight line. That's a power law: loss "
    "equals four point two four times the parameter count to the power minus zero point zero seven one. Every ten "
    "times more parameters multiplies the loss by about zero point eight five.",
    # 4. Off the line
    "The biggest model sits a little above the line: one point six one, where the law predicts one point five eight. "
    "A likely reason: it saw only about three tokens per parameter. It's starting to run short of data.",
    # 5. Data
    "So now vary the data. One model, with four hundred and fifty thousand parameters, measured as its training tokens "
    "double: two point four one, two point one seven, one point nine nine, one point eight, one point six eight, one "
    "point five nine. Another power law: every doubling multiplies the loss by about zero point nine two.",
    # 6. Real scaling laws
    "Research labs found the same shape on real language models, over many orders of magnitude of size, data and "
    "compute. A widely cited result, from the Chinchilla paper, is that for a fixed compute budget, the best results "
    "come from about twenty training tokens per parameter.",
    # 7. Prediction
    "This is what makes huge training runs possible to plan. Fit the law on small, cheap runs, and predict the loss of "
    "a model a thousand times larger before spending the money.",
    # 8. Limits
    "The laws describe the loss, not every ability, and they bend when something runs short, as our biggest model "
    "showed. But for the loss itself, they've held remarkably well.",
    # 9. Code
    "In code, fitting a power law is fitting a straight line through the logarithms of size and loss.",
    # 10. Outro
    "That's scaling. Next: the data itself, and where trillions of training tokens come from.",
]
