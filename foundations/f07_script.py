"""Foundations F7 — Softmax, Properly."""
TITLE = "Softmax, Properly"
TAGLINE = "From any scores to probabilities"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F7"
NEXT = "F8 · Averages and Spread"
USED_IN = [(5, "Attention I"), (6, "Attention II: The Math"), (10, "From Vectors Back to Words")]
SECTIONS = [
    "A neural network outputs raw scores, called logits. They can be any number: large, small, or negative. "
    "But we need probabilities: positive, and adding up to one. Softmax does that conversion.",
    "Step one: exponentiate every score. Thanks to e to the x, every result is positive, even for negative scores.",
    "Step two: divide each result by their total. Now they add up to exactly one. That's the whole recipe.",
    "Take the scores three, two, and zero. Exponentiated, they're about twenty, seven point four, and one. "
    "The total is about twenty-eight and a half. Divided out, that's about seventy percent, twenty-six percent, "
    "and three and a half percent.",
    "Why the name? A hard max would give all the probability to the top score. Softmax leans toward the top score, "
    "but still gives the others a share: a soft version of the max.",
    "Only the differences between scores matter. Add ten to every score, and the probabilities don't change at all.",
    "We can also divide the scores by a temperature before the softmax. Below one, the differences grow, "
    "and the top choice dominates. Above one, the distribution flattens out.",
    "One practical trick. E to the power of a thousand is too big for a computer. So code subtracts the largest score first. "
    "Since only differences matter, the answer is identical, and nothing overflows.",
    "In code, that's three lines: divide by the temperature, exponentiate after subtracting the max, and normalize.",
    "Softmax turns attention scores into weights in episodes five and six, and logits into word probabilities "
    "in episode ten. Next: averages and spread.",
]
