"""How LLMs Work: Deep Dive, episode 13 — The Residual Stream. One entry per narration section."""
TITLE = "The Residual Stream"
TAGLINE = "Every layer adds; nothing is replaced"
NEXT = "LayerNorm vs RMSNorm"
SECTIONS = [
    # 1. Hook
    "Inside a transformer, no layer replaces what came before. Each one adds: x plus attention, then x plus the M L P. "
    "That running sum is called the residual stream, and it's the model's shared workspace.",
    # 2. Picture
    "Picture one vector per token, seven hundred and sixty-eight numbers in GPT-2, flowing from the embedding to the "
    "output. Twenty-four sub-blocks sit along the way. Each one reads the stream, and writes a correction back into "
    "it.",
    # 3. Size
    "Let's measure it on GPT-2. The stream starts small: a size of four point six, just the embedding. After the first "
    "layer, about fifty. Then it grows steadily, to two hundred and sixteen entering the last layer. One token is "
    "different: the first. Its stream grows past three thousand. That's the attention sink from the last episode: a "
    "huge vector every head can find.",
    # 4. Small edits
    "Now compare each update with the stream it's added to. From layer one to layer nine, every attention and M L P "
    "update is between fifteen and thirty-four percent of the stream's size. Edits, not rewrites. Only the first "
    "layer writes more than the embedding itself, and the last layers make bigger changes to prepare the output.",
    # 5. Delete a layer
    "So what if we delete a whole layer? The baseline loss is four point one three. Remove any middle layer, and the "
    "loss stays between three point seven four and four point three three. On this text, some deletions even help. "
    "Remove the first layer: seven point nine five. The last: six point one two.",
    # 6. Why
    "Because each layer only adds, deleting one just removes one edit. The stream still flows to the end. In a plain "
    "chain without the addition, every layer would depend on the exact output of the one before.",
    # 7. Training without
    "Now train without it. The same tiny GPT, eight layers, with x equals x plus block of x, or just x equals block of "
    "x. With the residual: two point four six after a hundred steps, one point six nine at the end. Without it: stuck "
    "at three point three six from the start. That's almost exactly what you get from letter frequencies alone. It learned "
    "nothing else.",
    # 8. Gradient highway
    "The reason is the gradient. The derivative of x plus f of x is one plus the derivative of f. That one carries the "
    "learning signal straight back to the early layers, through all eight blocks, undamaged.",
    # 9. Code
    "In code, the residual stream is just two plus signs: x equals x plus attention, and x equals x plus M L P. "
    "Removing them was the whole experiment.",
    # 10. Outro
    "That's the residual stream. Next: LayerNorm and RMSNorm, how each block reads the stream at the right scale.",
]
