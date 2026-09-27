"""Foundations F1 — Vectors: Lists of Numbers as Arrows."""
TITLE = "Vectors: Lists of Numbers as Arrows"
TAGLINE = "The basic building block of every LLM"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F1"
NEXT = "F2 · The Dot Product"
USED_IN = [(1, "What is an LLM?"), (3, "Embeddings: Words as Vectors"), (4, "Where Am I? Position"), (9, "The Transformer Block")]
SECTIONS = [
    "Everything an LLM does, it does with vectors. Every token, every hidden state, every prediction starts as one. "
    "So what exactly is a vector?",
    "At its simplest, a vector is just a list of numbers. Like this one: two, one. Or this one, with four numbers. "
    "How many numbers it has is called its dimension.",
    "With two numbers, we can draw it. The first number says how far to go right, the second, how far to go up. "
    "So the vector two, one is an arrow from the origin to that point.",
    "Three numbers give an arrow in three-dimensional space. Beyond that, we can't draw it, "
    "but the math works exactly the same. GPT two uses vectors with seven hundred sixty-eight numbers.",
    "To add two vectors, add their numbers, position by position. Two, one, plus one, two, is three, three. "
    "As arrows, that means placing one arrow at the tip of the other.",
    "Multiplying by a single number, called a scalar, stretches the arrow. Two times two, one, is four, two: "
    "the same direction, twice as long. A negative number flips it around.",
    "The length of a vector comes from Pythagoras: square each number, add them up, and take the square root. "
    "For three, four, that's five.",
    "In an LLM, directions can come to mean something. One direction might lean toward animals, another toward plurals. "
    "A token's vector is a point in this space of meaning.",
    "In NumPy, a vector is an array. Adding, scaling, and measuring length take one line each.",
    "You'll see vectors in almost every episode: as embeddings in episode three, added to positions in episode four, "
    "and flowing along the residual stream in episode nine. Next: how to compare two vectors, with the dot product.",
]
