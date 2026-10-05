"""How LLMs Work: Deep Dive, episode 11 — Sliding Windows and Sparse Attention. One entry per narration section."""
TITLE = "Sliding Windows and Sparse Attention"
TAGLINE = "Skipping pairs on purpose, and what it costs"
NEXT = "Attention Sinks"
SECTIONS = [
    # 1. Hook
    "Full attention compares every token with every earlier one, so the work grows with the square of the length. At "
    "a hundred and thirty-one thousand tokens, that's eight point six billion pairs, per head, per layer. A sliding "
    "window cuts this: each token only looks at the last few.",
    # 2. The mask
    "Here is a window of four on eight tokens. Instead of a full triangle, the mask is a band along the diagonal. With "
    "a window of four thousand and ninety-six on a hundred and thirty-one thousand tokens, only six percent of the "
    "pairs remain.",
    # 3. Stacking
    "But information can hop. In one layer, a token reaches fifteen tokens back with a window of sixteen. The next "
    "layer reaches through those: thirty, then forty-five, then sixty. We measured it with gradients, layer by layer. "
    "Mistral seven B uses a window of four thousand and ninety-six with thirty-two layers: in theory, a reach of "
    "about a hundred and thirty-one thousand tokens.",
    # 4. Shakespeare
    "Does it hurt? We train the same tiny GPT on texts of two hundred and fifty-six characters. Full attention: a "
    "validation loss of one point seven eight. A window of sixty-four: one point seven five. A window of sixteen: one "
    "point six nine. The windows did better, with twelve percent of the pairs. For spelling out Shakespeare, nearby "
    "characters matter most, and the window is a helpful shortcut.",
    # 5. Copy task
    "Now a task that needs long range: a hundred and twenty-eight random symbols, then the same ones again. Full "
    "attention learns to copy perfectly: a loss of zero. Both windows are stuck at two point seven seven: pure "
    "guessing. Even the window of sixty-four, whose reach on paper is two hundred and fifty-two tokens, never learned "
    "to relay the symbols.",
    # 6. Mixing
    "So windows are cheap and fine for local patterns, but lose exact recall far back. That's why models mix them. "
    "Gemma 2, for example, alternates sliding-window layers with full-attention layers.",
    # 7. Sparse
    "Sliding windows are one kind of sparse attention. Others add a few global tokens that see, and are seen by, "
    "everything, or skip with a stride. The idea is always the same: choose which pairs are worth computing.",
    # 8. KV cache
    "There's a bonus at inference. With a window, the K V cache only needs the last W tokens: a fixed-size rolling "
    "buffer, however long the text grows.",
    # 9. Code
    "In code, the window is one line of the mask: a key is allowed if it is not in the future, and less than W "
    "tokens back. Fast kernels then skip every block outside the band.",
    # 10. Outro
    "That's sliding windows. Next: attention sinks, and why so many heads stare at the very first token.",
]
