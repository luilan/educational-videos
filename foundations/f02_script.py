"""Foundations F2 — The Dot Product."""
TITLE = "The Dot Product"
TAGLINE = "One number that says how much two vectors agree"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F2"
NEXT = "F3 · Matrices: Many Dot Products at Once"
USED_IN = [(3, "Embeddings: Words as Vectors"), (5, "Attention I"), (6, "Attention II: The Math"), (8, "The MLP"), (10, "From Vectors Back to Words")]
SECTIONS = [
    "Given two vectors, one of the most useful questions is: how much do they point the same way? "
    "The dot product answers that, with a single number.",
    "The recipe: multiply matching numbers, then add everything up. For three, one, and two, two: "
    "three times two is six, one times two is two, and six plus two is eight.",
    "It works in any number of dimensions, as long as both vectors have the same length. "
    "Seven hundred sixty-eight multiplications, then one big sum.",
    "Geometrically, it measures agreement. Arrows pointing the same way give a large positive number. "
    "At right angles, the dot product is exactly zero. Pointing in opposite directions, it goes negative.",
    "Here's the picture: project one arrow onto the other, like a shadow. "
    "The dot product is the length of that shadow, times the length of the other arrow.",
    "That gives a second formula: the length of a, times the length of b, times the cosine of the angle between them.",
    "If we only care about direction, we divide by both lengths. What's left is the cosine of the angle, "
    "called cosine similarity. One means the same direction, zero means unrelated, and minus one means opposite.",
    "Language models use dot products everywhere. Attention compares a query with a key. "
    "The output layer compares a vector with every word. And each MLP neuron is a dot product too.",
    "In NumPy, the dot product is the at sign. Divide by the two lengths, and you get cosine similarity.",
    "You'll meet the dot product in episodes three, five, six, eight, and ten. "
    "Next: doing lots of dot products at once, with matrices.",
]
