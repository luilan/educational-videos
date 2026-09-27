"""Foundations F3 — Matrices: Many Dot Products at Once."""
TITLE = "Matrices: Many Dot Products at Once"
TAGLINE = "The table of numbers behind every layer"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F3"
NEXT = "F4 · Straight Lines and Bends"
USED_IN = [(5, "Attention I"), (6, "Attention II: The Math"), (7, "Multi-Head Attention"), (8, "The MLP"), (10, "From Vectors Back to Words")]
SECTIONS = [
    "Language models compute billions of dot products. Writing them one by one would be hopeless. "
    "Matrices let us do many of them at once.",
    "A matrix is a table of numbers, with rows and columns. This one has two rows and three columns, "
    "so we call it a two by three matrix.",
    "To multiply a matrix by a vector, take the dot product of each row with the vector. "
    "Each row gives one number, so the result is a new vector, with one entry per row.",
    "So a matrix turns one vector into another. In two dimensions you can watch it happen: "
    "a matrix can stretch space, rotate it, or shear it, and every arrow moves along.",
    "The shapes must line up. A two by three matrix needs a vector with three numbers, and gives back two. "
    "In general, an m by n matrix turns n numbers into m numbers.",
    "Multiplying two matrices is the same idea, repeated: every row of the first, dotted with every column of the second. "
    "A five by four matrix, times a four by four matrix, gives a five by four matrix.",
    "That's exactly how LLMs use it. Stack our five tokens as the rows of a matrix, multiply by one weight matrix, "
    "and every token is transformed in a single step.",
    "One more operation: the transpose. It flips a matrix over its diagonal, so rows become columns. "
    "Q times K transposed compares every query with every key, all at once.",
    "In NumPy, the at sign multiplies matrices too, and dot T gives the transpose. Always check the shapes.",
    "Matrices power almost every step from episode five to episode ten: queries, keys and values, "
    "attention scores, the MLP, and the final logits. Next: why stacking matrices isn't enough, and the bend that fixes it.",
]
