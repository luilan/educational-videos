"""Foundations F5 — Exponentials and Logarithms."""
TITLE = "Exponentials and Logarithms"
TAGLINE = "Fast growth, and the function that undoes it"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F5"
NEXT = "F6 · Probability and Sampling"
USED_IN = [(6, "Attention II: The Math"), (10, "From Vectors Back to Words"), (11, "Training"), (12, "Build a Tiny GPT")]
SECTIONS = [
    "Two functions show up again and again in LLMs: the exponential, and its opposite, the logarithm. "
    "Let's build an intuition for both.",
    "Exponential growth means multiplying, not adding. Two, four, eight, sixteen: doubling every step. "
    "After just ten doublings, you're past a thousand.",
    "In machine learning, the base is usually the number e, about two point seven one eight. "
    "The function e to the x has two properties we care about.",
    "First, it's always positive. Even e to the minus ten is a tiny positive number, less than one ten-thousandth. "
    "It never reaches zero, and it never goes negative.",
    "Second, it exaggerates differences. The inputs one and three differ by just two. "
    "But e to the one is about two point seven, and e to the three is about twenty. Now one is more than seven times the other.",
    "The logarithm undoes the exponential. The natural log of e to the x is just x. "
    "The log of one is zero, and as numbers shrink toward zero, the log dives toward minus infinity.",
    "That makes logs perfect for probabilities. A probability of zero point nine has a log close to zero. "
    "A probability of zero point zero one has a log of about minus four point six. The less likely, the more negative.",
    "Logs also turn multiplication into addition. And they give us log scale charts, where each step is ten times "
    "bigger than the last, so a thousand and a billion fit on the same axis.",
    "In NumPy, it's np dot exp and np dot log.",
    "You'll see the exponential inside softmax, in episodes six and ten, and the logarithm in the loss of episode eleven, "
    "and the log scale chart of episode twelve. Next: probability and sampling.",
]
