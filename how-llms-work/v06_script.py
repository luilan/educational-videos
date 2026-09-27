"""Video 6 — Attention II: The Math. One entry per narration section."""
TITLE = "Attention II: The Math"
TAGLINE = "Queries, keys and values, one matrix at a time"
NEXT = "Multi-Head Attention"
SECTIONS = [
    # 1. Intro
    "Last time, every token asked a question and borrowed meaning from the others. "
    "Now let's do it for real, with actual numbers. We'll follow our five tokens: the cat sat on the.",
    # 2. The input matrix
    "Start with the input: one row per token. That's a matrix called X, with five rows, and d columns. "
    "We'll use d equals four, so every number fits on screen.",
    # 3. Projections
    "Multiply X by three learned matrices: W Q, W K, and W V. "
    "That gives us Q, K, and V: a query, a key, and a value for every token, all computed at once.",
    # 4. Scores
    "Next, compare every query with every key. That's a single matrix multiplication: Q times K transposed. "
    "The result is a five by five grid of scores. Row i, column j, says how well token i's question "
    "matches token j's key.",
    # 5. Scaling
    "Then divide every score by the square root of the key size. Without this, scores grow with the vector length, "
    "the softmax gets too extreme, and training struggles.",
    # 6. The causal mask
    "Now the causal mask. A token must not see the future, so every score above the diagonal "
    "is set to minus infinity. After the next step, those become exactly zero.",
    # 7. Softmax
    "Softmax turns each row into weights. Exponentiate every score, then divide by the row's total. "
    "Every row is now positive, and adds up to one.",
    # 8. Weighted sum of values
    "Finally, multiply the weights by V. Each token's output is a weighted blend of the values it attends to. "
    "The first token can only see itself, so it simply gets its own value back.",
    # 9. The whole formula
    "Put together, it's one line: softmax of Q times K transposed, over the square root of d, times V. "
    "Every token, in parallel, in a handful of matrix multiplications. "
    "That's a big reason transformers run so well on GPUs.",
    # 10. Code
    "In NumPy, it's just as short. Project. Score and scale. Mask the future. Softmax. And mix the values.",
    # 11. Outro
    "One attention layer asks one kind of question. But a token might want to know several things at once. "
    "Next time: multi-head attention.",
]
