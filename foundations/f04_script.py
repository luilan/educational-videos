"""Foundations F4 — Straight Lines and Bends."""
TITLE = "Straight Lines and Bends"
TAGLINE = "Linear functions, and why networks need a bend"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F4"
NEXT = "F5 · Exponentials and Logarithms"
USED_IN = [(8, "The MLP: Where Facts Live"), (9, "The Transformer Block")]
SECTIONS = [
    "A function takes an input and gives an output. The simplest kind is linear: double the input, "
    "and the output doubles too. Its graph is a straight line through the origin.",
    "Multiplying by a matrix is linear too. Scale the input vector, and the output scales the same way. "
    "Add two inputs, and their outputs add.",
    "Here's the catch. Apply one matrix, then another, and the result is just a single matrix: their product. "
    "Stack a hundred linear layers, and they still collapse into one.",
    "And a single linear map can only do so much. It can stretch and rotate, but it can't bend. "
    "Try to separate the points inside this circle from the points outside it with a straight line, and you can't.",
    "The fix is to add a bend between the layers: a nonlinear function, applied to every number separately.",
    "The simplest one is called ReLU. It keeps positive numbers, and replaces negative ones with zero. "
    "A tiny change, but now stacked layers can't collapse.",
    "Transformers usually use GELU, a smooth version of ReLU. Large positive inputs pass through, "
    "large negative ones fade to zero, with a gentle curve in between.",
    "With bends between the layers, a network can build curved shapes out of many simple pieces, "
    "and approximate almost any function.",
    "In code, ReLU is one line. A layer is a matrix multiplication, then the bend, then the next matrix.",
    "You'll see this in episode eight, the MLP: expand, bend with GELU, and project back. "
    "And in episode nine, where the blocks are stacked. Next: exponentials and logarithms.",
]
