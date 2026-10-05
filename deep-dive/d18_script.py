"""How LLMs Work: Deep Dive, episode 18 — Backprop Through a Transformer. One entry per narration section."""
TITLE = "Backprop Through a Transformer"
TAGLINE = "One loss number, 124 million gradients"
NEXT = "Adam and AdamW"
SECTIONS = [
    # 1. Hook
    "Training needs, for every one of GPT-2's hundred and twenty-four million weights, one number: how the loss would "
    "change if that weight moved a little. One backward pass computes all of them at once. Let's see how, and check "
    "that it's right.",
    # 2. Chain rule
    "Backprop is the chain rule, applied step by step from the loss back to the weights. A tiny example: tanh of w "
    "times x, minus three, squared. Multiply the local derivatives of each step: minus three point seven six zero "
    "three. Autograd gets the same number.",
    # 3. Checking on GPT-2
    "Now a real weight inside GPT-2, in the M L P of layer five. Autograd says its gradient is minus two times ten to "
    "the minus five. Check it the slow way: nudge the weight up and down by a ten-thousandth, and measure the loss "
    "each time. The slope matches autograd to seven digits.",
    # 4. A trap
    "One trap on the way: the first time, the check failed. The library's built-in loss quietly rounds to lower "
    "precision, which swamps changes this small. Computing the loss ourselves, in full precision, fixed it.",
    # 5. Every weight
    "One backward pass gives every one of the hundred and twenty-four million parameters its gradient, from a single "
    "loss number. Even all fifty thousand rows of the token embedding, because the same matrix also produces the "
    "output scores.",
    # 6. Gradient sizes
    "How big are the gradients, layer by layer? In this model, the early and middle layers get the largest ones, "
    "between about two and four, and the last two layers the smallest, under one. Nothing vanishes on the way back: "
    "even layer zero gets a large gradient.",
    # 7. Cost in time
    "The cost. On this CPU, a forward pass over a thousand and twenty-four tokens takes zero point five six seconds. "
    "The backward pass takes one point two one: about twice as long, because it computes gradients for both the "
    "activations and the weights.",
    # 8. Cost in memory
    "And memory. To run backward, the forward pass must keep its intermediate results. For one sequence of a thousand "
    "and twenty-four tokens, that's one thousand four hundred and forty-three megabytes of activations: three times "
    "the size of the model's weights.",
    # 9. Code
    "In code, it's one call: loss dot backward. Every parameter then holds its gradient, ready for the optimizer.",
    # 10. Outro
    "That's backprop. Next: the optimizer that turns those gradients into updates, Adam and AdamW.",
]
