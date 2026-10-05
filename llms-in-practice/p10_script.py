"""LLMs in Practice, episode 10 — Quantization: Shrinking a Model to Fit a Laptop. One entry per narration section."""
TITLE = "Quantization"
TAGLINE = "Shrinking a model to fit a laptop"
NEXT = "Evaluating LLMs: How We Know It's Better"
SECTIONS = [
    # 1. Hook
    "A model is mostly numbers: its weights. Our small Kwen model has four hundred and ninety-four million of them. "
    "Stored the usual way, that's almost two gigabytes. Big models need hundreds. "
    "So how do people run them on a laptop, or a phone?",
    # 2. Where the size comes from
    "Each weight is normally a thirty-two bit number, four bytes, or sixteen bits, two bytes. "
    "But does a weight really need that much precision? Quantization says no: store each weight with fewer bits.",
    # 3. How it works
    "Here's the simplest recipe. Take one row of a weight matrix. Find its largest value. "
    "Then round every weight to the nearest step on a small grid, from minus that maximum to plus it. "
    "With eight bits, that's two hundred and fifty-five steps. With four bits, only fifteen. "
    "Store the small integers, plus one scale for the row.",
    # 4. Real weights
    "These are real first weights from the model's first layer, before and after four-bit rounding. "
    "Close, but not the same. Two small ones became exactly zero.",
    # 5. The results
    "Now we measure. For the weight matrices, which hold seventy-two percent of the parameters: "
    "thirty-two bits take one thousand three hundred and sixty-five megabytes, and the loss on our test text is "
    "two point eight three six. Eight bits: a quarter of the size. Weights move by about one percent, "
    "and the loss is two point eight three seven. Practically identical.",
    # 6. Going lower
    "Four bits: one eighth of the size. The weights move by nineteen percent, and the loss rises to three point "
    "five eight. The answers still make sense, but they're vaguer. Three bits, and the loss explodes to twelve. "
    "Two bits: sixteen. The model writes gibberish.",
    # 7. A subtle point
    "One subtle thing. At eight bits, the loss barely moved, but the model's answer to our question still changed. "
    "When two tokens are almost tied, a tiny change can flip the choice, and everything after it.",
    # 8. Better methods
    "Real tools do much better than our simple rounding. They use small groups of weights with their own scale, "
    "keep the most sensitive weights at higher precision, and calibrate on sample text. "
    "That's why four-bit versions of big models are popular: the right trade-off for many laptops.",
    # 9. Code
    "In code, the simple version is three lines per matrix: compute the scale from the largest value, "
    "round to integers, and multiply back.",
    # 10. Outro
    "A smaller model is only useful if it's still good. Next up: how we measure that. Evaluating LLMs.",
]
