"""Foundations F6 — Probability and Sampling."""
TITLE = "Probability and Sampling"
TAGLINE = "Distributions, weighted dice, and randomness"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F6"
NEXT = "F7 · Softmax, Properly"
USED_IN = [(1, "What is an LLM?"), (10, "From Vectors Back to Words"), (14, "From GPT to Chatbot")]
SECTIONS = [
    "An LLM never simply knows the next word. It produces probabilities. "
    "So let's be clear about what a probability is.",
    "A probability is a number between zero and one: how likely something is. "
    "Zero means impossible, one means certain, and a half means it happens about half the time.",
    "A probability distribution spreads one whole unit of belief across all the options. "
    "For the word after: the cat sat on the, maybe mat gets zero point four, and floor, zero point two five. "
    "Together, all the probabilities must add up to exactly one.",
    "Sampling means picking one option at random, according to those probabilities. "
    "Think of a spinner, where each word gets a slice sized by its probability.",
    "Spin it many times, and each word comes up about as often as its probability says. "
    "Mat about forty percent of the time, floor about a quarter of the time.",
    "Always choosing the biggest slice is predictable. Sampling adds variety, "
    "which is why the same prompt can give different answers.",
    "Computers make randomness with pseudo-random number generators. Start one with the same seed, "
    "and you get the same sequence of random choices every time. That keeps experiments reproducible.",
    "In NumPy, a random generator with a seed can sample one word, or many, from a list of probabilities.",
    "You'll see this in episode one, where the model predicts a distribution, in episode ten, where we sample from it, "
    "and in episode fourteen. Next: softmax, the function that turns scores into probabilities.",
]
