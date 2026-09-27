"""Foundations F13 (extra) — PyTorch and Autograd."""
TITLE = "PyTorch and Autograd"
TAGLINE = "How one line computes every gradient"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F13  ·  EXTRA"
NEXT = "F14 · Train vs Validation Data"
USED_IN = [(11, "Training: Learning from Mistakes"), (12, "Build a Tiny GPT")]
SECTIONS = [
    "In episode eleven, one line did all the calculus: loss, dot backward. Let's see what that line actually does.",
    "PyTorch works with tensors: arrays of numbers, just like NumPy arrays. "
    "The difference is that a tensor can remember how it was computed.",
    "Mark a tensor with requires grad equals true, and PyTorch starts recording every operation that uses it.",
    "Take x equals three, and compute y equals x squared, plus two x. As it computes, PyTorch builds a graph: "
    "x goes into a square, and into a times two, and the two results are added. Y comes out as fifteen.",
    "Call y dot backward, and PyTorch walks that graph in reverse, applying the chain rule at every step. "
    "The result lands in x dot grad.",
    "The slope of x squared plus two x is two x plus two. At x equals three, that's eight. "
    "And x dot grad says exactly eight.",
    "A real model is the same thing, just bigger. The graph has millions of steps, "
    "and the loss depends on millions of weights, but one backward call fills in the gradient of every weight.",
    "Then an optimizer, like Adam, reads those gradients and updates the weights. Zero grad clears the old gradients first, "
    "because PyTorch adds new gradients to whatever is already there. Forget it, and eight plus eight becomes sixteen.",
    "In code: create a tensor that requires gradients, compute with it, call backward, and read the gradient.",
    "You'll see this in the training loop of episode eleven, and in the tiny GPT of episode twelve. "
    "Next: why we always keep some data aside, and what the validation curve tells us.",
]
