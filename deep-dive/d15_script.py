"""How LLMs Work: Deep Dive, episode 15 — Activations: ReLU, GELU, SwiGLU. One entry per narration section."""
TITLE = "Activations: ReLU, GELU, SwiGLU"
TAGLINE = "The bend inside every MLP"
NEXT = "Pre-Norm, Post-Norm, and Stability"
SECTIONS = [
    # 1. Hook
    "Every M L P in a transformer has the same shape: widen, bend, narrow. The bend is the activation function. "
    "Without it, the two matrices would collapse into one, and the M L P could only compute straight lines. GPT-2 "
    "uses GELU. Llama and Kwen use SwiGLU. Let's see why.",
    # 2. ReLU and GELU
    "ReLU is the simplest bend: negative numbers become zero, positive ones pass. GELU is a smooth version: large "
    "positive numbers pass, large negative ones fade to zero, but in between it dips below zero, down to minus zero "
    "point one seven.",
    # 3. GPT-2 uses the dip
    "Does that dip matter? In GPT-2's sixth layer, eighty-six percent of the hidden values are negative before GELU. "
    "Most of them don't become zero: only four percent end up within a hundredth of zero. Swap every GELU for ReLU, "
    "without retraining, and the loss jumps from four point one three to seven point two three. The model relies on "
    "those small negative values.",
    # 4. SwiGLU
    "SwiGLU adds a gate. Instead of one matrix going up, there are two: a gate and an up projection. The gate passes "
    "through SiLU, a smooth bend like GELU, and multiplies the other one, number by number. Then a third matrix goes "
    "down. In Kwen two point five, that's eight hundred and ninety-six, to four thousand eight hundred and sixty-four, "
    "and back. Our version matches the model's exactly.",
    # 5. Same budget
    "Three matrices instead of two would be unfair, so SwiGLU uses a narrower middle: two thirds of the width, for "
    "about the same number of parameters.",
    # 6. Results
    "We train the same tiny GPT three times with each activation, three seeds each. ReLU: one point six five. GELU: "
    "one point six two. SwiGLU: one point five nine. The same order in every run. A small gain, but for free.",
    # 7. Dead neurons
    "A classic worry with ReLU is dead neurons: units that never fire again. In our small model, all two thousand and "
    "forty-eight fired at least once. At this scale, that's not what made ReLU worse.",
    # 8. Why gates help
    "Why do gates help? One common explanation: each hidden unit becomes a product of two learned signals, so the M L P can switch features on "
    "and off depending on the input, not just bend one signal.",
    # 9. Code
    "In code, SwiGLU is one line: down of SiLU of gate of x, times up of x.",
    # 10. Outro
    "That's the bend inside every M L P. Next: where to put the normalization, before or after each block, and why it "
    "decides whether training stays stable.",
]
