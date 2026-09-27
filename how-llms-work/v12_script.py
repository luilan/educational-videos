"""Video 12 — Build a Tiny GPT. One entry per narration section."""
TITLE = "Build a Tiny GPT"
TAGLINE = "Every piece, in about a hundred lines of code"
NEXT = "Making It Fast: the KV Cache"
SECTIONS = [
    # 1. Intro
    "Over eleven videos, we've built every piece of a GPT. Now let's put them all together, "
    "in about a hundred lines of Python, and train one on an ordinary computer.",
    # 2. Data
    "Our training text is about a million characters of Shakespeare's plays. To keep things tiny, "
    "every character is a token. The whole vocabulary is just sixty-five symbols.",
    # 3. Tokenizer
    "So the tokenizer is two small dictionaries: from characters to IDs, and back again.",
    # 4. Embeddings
    "The model starts with two embedding tables, one for tokens and one for positions, added together. "
    "Just like videos three and four.",
    # 5. Attention
    "Then attention: queries, keys and values from one linear layer, split across four heads, "
    "with the causal mask, and the softmax.",
    # 6. MLP and block
    "The MLP expands to four times the width, applies GELU, and projects back. "
    "And the block wires them onto the residual stream, with layer norms. We stack four blocks.",
    # 7. Output
    "Finally, one last layer norm, and a linear layer that turns each vector into sixty-five logits, "
    "one per character.",
    # 8. Size
    "That's the whole model: eight hundred eighteen thousand parameters. "
    "GPT two small has a hundred and twenty-four million. The largest models today have hundreds of billions.",
    # 9. Training
    "Training is the loop from last time. Grab random chunks of text, predict every next character, "
    "compute the loss, backpropagate, and step with Adam. Five thousand steps took under an hour, on an ordinary CPU.",
    # 10. Step 0
    "Before training, here's what it writes. Random characters. Pure noise.",
    # 11. Step 250
    "After just two hundred fifty steps, it has learned which letters are common, where the spaces go, "
    "and that lines are short. The words are still made up.",
    # 12. Step 5000
    "And after five thousand steps: real words, character names, and the shape of a play. "
    "It isn't Shakespeare, but it learned all of this from nothing but predicting the next character.",
    # 13. Outro
    "That's a GPT, built from scratch. Real models use the same recipe, with far more data, far more layers, "
    "and far more compute. In two bonus videos, we'll see how they generate text quickly, "
    "and how a raw GPT becomes a helpful chatbot.",
]
