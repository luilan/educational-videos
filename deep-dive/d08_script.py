"""How LLMs Work: Deep Dive, episode 8 — The Attention Matrix, Entry by Entry. One entry per narration section."""
TITLE = "The Attention Matrix, Entry by Entry"
TAGLINE = "One GPT-2 head, computed by hand"
NEXT = "Multi-Query and Grouped-Query Attention"
SECTIONS = [
    # 1. Hook
    "Attention diagrams show a grid of weights. But where does each number come from? Let's compute one attention "
    "head of the real GPT-2 by hand, entry by entry, and check every step against the library.",
    # 2. Q, K, V
    "Our sentence: the cat sat on the mat because it was tired. Ten tokens. We look at layer four, head three, "
    "counting from zero. Each token's vector of seven hundred and sixty-eight numbers is normalized, then multiplied "
    "by one matrix that gives its query, key and value. Each is split into twelve heads of sixty-four numbers.",
    # 3. One entry
    "One entry: the query of it, against the key of cat. Multiply the sixty-four pairs of numbers, and add them up: "
    "four point five five. Then divide by eight, the square root of sixty-four: zero point five seven. That is one "
    "score.",
    # 4. The row
    "Now do it for every token up to it. The scores range from minus four point three to zero point five seven. Mask "
    "the future, apply softmax, and eighty-four percent of the attention goes to cat. In this head, it points at the "
    "cat.",
    # 5. The matrix
    "Repeat for every query, and you get the full matrix: ten by ten, a hundred entries, forty-five of them masked. Our "
    "numbers match the library's exactly: a difference of zero. In this head, most words look back at cat, the "
    "subject of the sentence.",
    # 6. Why divide by 8
    "Why divide by eight? The dot product of two random vectors of sixty-four numbers spreads out by about eight. "
    "Without the division, softmax becomes nearly all or nothing: the largest weight in each row averages zero point "
    "nine eight, instead of zero point six eight. Almost one-hot, and much harder to train.",
    # 7. Values and heads
    "Then the weights mix the values: the output for it is a weighted average of sixty-four numbers, mostly cat's. "
    "The twelve heads' outputs are joined back into seven hundred and sixty-eight numbers, and mixed by one more "
    "matrix. Again, identical to the library.",
    # 8. Other heads
    "Other heads have other habits. In layer four, head eleven, every token puts all of its attention on the token "
    "just before it. And in ninety-two of the hundred and forty-four heads, more than half the attention goes to the "
    "very first token. We'll see why in the episode on attention sinks.",
    # 9. Scale
    "GPT-2 builds a hundred and forty-four of these matrices on every forward pass. At a thousand and twenty-four "
    "tokens, that is a hundred and fifty-one million weights. Keeping that memory in check is a story of its own.",
    # 10. Code
    "In code, a head is a few lines: queries times keys, divided by the square root of the head size, masked, "
    "softmaxed, then times the values.",
    # 11. Outro
    "That's one head, entry by entry. Next: why many heads can share their keys and values.",
]
