"""How LLMs Work: Deep Dive, episode 14 — LayerNorm vs RMSNorm. One entry per narration section."""
TITLE = "LayerNorm vs RMSNorm"
TAGLINE = "Reading the stream at the right scale"
NEXT = "Activations: ReLU, GELU, SwiGLU"
SECTIONS = [
    # 1. Hook
    "Every block reads the residual stream through a normalization. GPT-2 uses LayerNorm. Llama, Mistral and Kwen use "
    "RMSNorm. What's the difference, and does it matter?",
    # 2. Why normalize
    "First, why normalize at all? In GPT-2, the stream grows from a size of five to two hundred and fifty-three across "
    "the layers. But after normalization, every block reads a vector of exactly the same size: twenty-seven point "
    "seven, the square root of seven hundred and sixty-eight, at every layer.",
    # 3. LayerNorm
    "LayerNorm, by hand. Take the token sat, entering layer six: seven hundred and sixty-eight numbers, with a mean of "
    "zero point zero eight eight, and a standard deviation of three point four six five. Subtract the mean, divide by "
    "the standard deviation, then multiply by a learned scale and add a learned shift. Seven hundred and sixty-eight "
    "numbers each. It matches GPT-2's own LayerNorm to within five ten-millionths.",
    # 4. RMSNorm
    "RMSNorm skips the mean. Divide by the root mean square, multiply by a learned scale, and that's it: no centering, "
    "no shift. In Kwen two point five, the mean of the stream is zero point zero two two, almost zero anyway. Again, "
    "our version matches the model's.",
    # 5. Same results
    "Does it matter? We train the same tiny GPT, eight layers, three ways. LayerNorm: one point six nine. RMSNorm: one "
    "point six nine. And with no normalization at all: one point seven one. At this learning rate, barely any "
    "difference.",
    # 6. Stability
    "Now raise the learning rate ten times. LayerNorm and RMSNorm both reach one point eight one. Without "
    "normalization, the loss becomes not a number: the training blows up. Normalization is what keeps it stable.",
    # 7. Why RMSNorm
    "So why have modern models switched? RMSNorm is simpler: one statistic instead of two, and no shift. Centering "
    "turned out to matter little. Speed, though, depends on the implementation: on this CPU, PyTorch's built-in "
    "LayerNorm ran three times faster than its RMSNorm. With equally optimized code, RMSNorm does slightly less "
    "work.",
    # 8. Code
    "In code, RMSNorm is two lines: divide by the square root of the mean of the squares, plus a small epsilon, and "
    "multiply by the learned weight.",
    # 9. Outro
    "That's normalization. Next: the activation inside the M L P, from ReLU to GELU to SwiGLU.",
]
