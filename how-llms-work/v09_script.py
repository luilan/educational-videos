"""Video 9 — The Transformer Block. One entry per narration section."""
TITLE = "The Transformer Block"
TAGLINE = "Attention and MLP, wired together and stacked"
NEXT = "From Vectors Back to Words"
SECTIONS = [
    # 1. Intro
    "We now have both halves: attention, where tokens share information, and the MLP, "
    "where each token processes it. Let's wire them together.",
    # 2. The residual stream
    "The central idea is the residual stream. Picture each token's vector flowing along a stream, "
    "from its embedding at the start, to the prediction at the end.",
    # 3. Add, don't replace
    "Attention and the MLP don't replace the vector. They read from the stream, compute something, "
    "and add their result back. X plus attention of X. Then, X plus MLP of X.",
    # 4. Why adding matters
    "This matters. Adding keeps the original information around, and it gives training a direct path "
    "back through the network. It's a big part of what makes deep stacks of layers trainable.",
    # 5. Layer norm
    "Before each half, there's a layer norm. It rescales the vector so its numbers have a mean of zero "
    "and a standard deviation of one, then applies a learned scale and shift. "
    "It keeps the numbers in a healthy range as they flow through many layers.",
    # 6. The block
    "And that's a transformer block. Layer norm, attention, add. Layer norm, MLP, add.",
    # 7. Stacking
    "Now stack it. GPT two small has twelve blocks. The largest GPT two has forty-eight. "
    "Every block has its own weights, and each one refines the vectors a little more.",
    # 8. What the layers do
    "Researchers see a rough pattern. Earlier layers tend to handle local things, like grammar and nearby words. "
    "Later layers deal with more abstract meaning, and with predicting the next token.",
    # 9. Code
    "In code, a block is two lines: add attention of the normalized input, then add the MLP. "
    "The layer norm is a few more lines. And the whole model is just a loop over the blocks.",
    # 10. Outro
    "After the last block, each token's vector holds the model's view of what comes next. "
    "But it's still a vector. Next time: turning it back into words.",
]
