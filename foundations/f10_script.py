"""Foundations F10 — Slopes and Gradients."""
TITLE = "Slopes and Gradients"
TAGLINE = "How a model knows which way to adjust"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F10"
NEXT = "F11 · NumPy in Three Minutes"
USED_IN = [(11, "Training: Learning from Mistakes"), (12, "Build a Tiny GPT")]
SECTIONS = [
    "Training means adjusting weights to reduce a loss. To know which way to adjust them, "
    "we need one idea from calculus: the slope.",
    "The slope of a curve tells you how much the output changes when you nudge the input a little. "
    "A steep slope means a big change. A flat one, almost no change.",
    "Take the function x squared. At x equals three, nudge x by a tiny amount, and the output grows about six times as fast. "
    "That rate, six, is the derivative. In general, the derivative of x squared is two x.",
    "The slope also tells us which way is downhill. At x equals three, the slope is positive, "
    "so to make the output smaller, we step to the left: against the slope.",
    "That's gradient descent, in one dimension. The new x is the old x, minus the learning rate times the slope. "
    "With a learning rate of zero point one, three becomes two point four. Repeat, and x slides toward the minimum at zero.",
    "A real model has millions of weights, not one. The gradient collects one slope for each weight, into a single vector. "
    "It points uphill, so we step the opposite way.",
    "Layers are functions inside functions. The chain rule says their slopes multiply. If y changes three times as fast as x, "
    "and z changes twice as fast as y, then z changes six times as fast as x.",
    "Backpropagation applies the chain rule layer by layer, from the loss back to every weight, "
    "so one backward pass gives the whole gradient.",
    "In code, gradient descent is a short loop: compute the slope, and step against it.",
    "You'll see all of this in episode eleven: the landscape of the loss, gradients, the learning rate, and backpropagation. "
    "Next: a quick tour of NumPy, so you can read every snippet in the series.",
]
