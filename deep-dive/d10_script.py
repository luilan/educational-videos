"""How LLMs Work: Deep Dive, episode 10 — FlashAttention: Same Math, Less Memory. One entry per narration section."""
TITLE = "FlashAttention"
TAGLINE = "Same math, less memory"
NEXT = "Sliding Windows and Sparse Attention"
SECTIONS = [
    # 1. Hook
    "Standard attention builds the full matrix of scores: every token against every token. For four thousand tokens "
    "and twelve heads, that's one and a half gigabytes, measured, for a single layer. At a hundred and thirty-one "
    "thousand tokens, it would be seven hundred and sixty-eight gigabytes. FlashAttention computes exactly the same result, "
    "without ever storing that matrix.",
    # 2. The problem
    "The obstacle is softmax. To turn a row of scores into weights, you need the largest score and the total, over "
    "the whole row, before you know any single weight. So it seems you must keep the whole row.",
    # 3. Online softmax
    "The trick is an online softmax. Read the row in chunks, and keep three running numbers: the largest score so "
    "far, the sum so far, and the output so far. Here are eight scores in two chunks. After the first, the max is "
    "three. The second chunk brings a four, so we rescale what we had, and add the new terms. The result: thirty-six "
    "point two seven four two, exactly what the full softmax gives.",
    # 4. Tiling
    "FlashAttention applies this to the whole matrix, in tiles. Take a block of sixty-four queries and a block of "
    "sixty-four keys, compute their scores, update the running numbers, and move on. Blocks entirely in the future "
    "are skipped, and only the diagonal blocks need the mask. The biggest piece ever stored is sixty-four by "
    "sixty-four: sixteen kilobytes instead of four megabytes.",
    # 5. Same math
    "This is not an approximation. Our twenty-line version matches standard attention to within five ten-millionths, "
    "just rounding.",
    # 6. Measured
    "Now with twelve heads, against PyTorch's fused attention kernel, which also works block by block. At a thousand "
    "tokens, ninety-nine megabytes against eight. At four thousand: one and a half gigabytes against eleven "
    "megabytes. And it is twelve times faster.",
    # 7. Why faster
    "Why faster, when the math is the same? On a GPU, the slow part is moving data, not multiplying it. The small "
    "tiles stay in fast on-chip memory, while the big matrix would travel back and forth to the much slower main "
    "memory. Fewer trips, faster attention.",
    # 8. What doesn't change
    "What FlashAttention does not change: every query is still compared with every earlier key. The work still grows "
    "with the square of the length. To cut that, models must skip some of the pairs.",
    # 9. Code
    "In code, the heart of it is the inner loop: find the new max, rescale the old sum and output, and add the new "
    "block. In PyTorch, you get the fused version by calling scaled dot product attention.",
    # 10. Outro
    "That's FlashAttention. Next: sliding windows and sparse attention, which skip pairs on purpose.",
]
