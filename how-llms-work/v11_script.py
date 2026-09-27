"""Video 11 — Training: Learning from Mistakes. One entry per narration section."""
TITLE = "Training: Learning from Mistakes"
TAGLINE = "Loss, gradients, and millions of tiny nudges"
NEXT = "Build a Tiny GPT"
SECTIONS = [
    # 1. Random start
    "A freshly created model is pure noise. Every weight is random, and its predictions are gibberish. "
    "Training is how those numbers become useful.",
    # 2. Self-supervised data
    "The recipe is simple. Take a huge amount of text. Hide the next token, ask the model to predict it, "
    "and compare its guess with the truth. The text itself provides the answers, so nobody has to label anything.",
    # 3. Cross-entropy
    "We score each prediction with cross-entropy loss: the negative log of the probability "
    "the model gave to the correct token. If it gave mat a probability of ninety percent, the loss is about zero point one. "
    "If it gave it just one percent, the loss is about four point six.",
    # 4. Averaging
    "We average this loss over every position in a batch of text. Lower is better. "
    "Training is simply the search for weights that make the loss small.",
    # 5. Loss landscape
    "Picture the loss as a landscape, where every point is one setting of all the weights. "
    "We want to find a deep valley. But there are millions of dimensions, so we can't just look around. "
    "We have to feel our way downhill.",
    # 6. Gradient descent
    "The gradient tells us, for every single weight, which direction makes the loss go up, and how steeply. "
    "So we step the other way. Every weight moves a tiny bit against its gradient. That's gradient descent.",
    # 7. Learning rate
    "The size of that step is the learning rate. Too small, and training takes forever. "
    "Too large, and we overshoot the valley and bounce around.",
    # 8. Backpropagation
    "But how do we get the gradient for millions of weights at once? With backpropagation. "
    "It applies the chain rule from calculus, starting at the loss and working backwards through every layer, "
    "reusing results along the way. A backward pass costs only about twice as much as a forward pass.",
    # 9. The loop and Adam
    "Then we repeat. A batch of text, a forward pass, the loss, a backward pass, a small step. "
    "Over and over, across billions of tokens. In practice, we use a smarter update rule called Adam, "
    "which adapts the step size for each weight.",
    # 10. Code
    "In code, a library like PyTorch does the calculus for us. Forward pass. Loss. Backward pass. Step.",
    # 11. A real loss curve
    "Here's a real loss curve, from the tiny model we'll build next time. It starts at about four point four: "
    "no better than guessing. It falls fast at first, then slower, and ends near one point six, "
    "measured on text it never trained on.",
    # 12. Outro
    "We now have every piece. Next time, we'll put them all together, write a tiny GPT from scratch, "
    "train it, and watch it learn to write.",
]
