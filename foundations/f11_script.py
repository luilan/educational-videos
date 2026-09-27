"""Foundations F11 — NumPy in Three Minutes."""
TITLE = "NumPy in Three Minutes"
TAGLINE = "Just enough to read the code in the series"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F11"
NEXT = "Episode 1 · What is an LLM?"
USED_IN = [(3, "Embeddings"), (6, "Attention II: The Math"), (7, "Multi-Head Attention"), (9, "The Transformer Block"), (13, "The KV Cache")]
SECTIONS = [
    "The code in this series uses NumPy, Python's library for fast math on arrays of numbers. "
    "Here's just enough to read every snippet.",
    "An array holds numbers in a grid. A vector is a one-dimensional array. A matrix is two-dimensional. "
    "Every array has a shape: this one is five by four, five rows of four numbers.",
    "Arithmetic works on every element at once. Add two arrays of the same shape, and matching elements are added. "
    "Multiply by two, and every number doubles.",
    "The at sign is matrix multiplication: dot products of rows with columns. "
    "A five by four array, at a four by four array, gives five by four.",
    "Many functions take an axis. Summing with axis equals minus one adds along the last dimension, "
    "giving one total per row. Keep dims equals true keeps that result as a column.",
    "That column lines up with the rows because of broadcasting. Divide a five by five array by a five by one column, "
    "and each row is divided by its own number. That's exactly how softmax normalizes every row.",
    "Reshape regroups the same numbers into a new shape. Seven hundred sixty-eight numbers can become twelve heads of sixty-four. "
    "Transpose swaps the axes.",
    "Finally, masks. NumPy's upper-triangle function builds a triangle of ones above the diagonal. "
    "Use it to pick out elements, and set them to minus infinity. That's the causal mask from episode six.",
    "Here it all is in a few lines: shapes, the at sign, a sum along an axis, a reshape, and a mask.",
    "That's the toolkit: vectors, dot products, matrices, bends, exponentials, probability, softmax, statistics, waves, "
    "gradients, and NumPy. You're ready for the main series. Next up: episode one, what is an LLM?",
]
