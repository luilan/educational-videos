"""How LLMs Work: Deep Dive, episode 12 — Attention Sinks. One entry per narration section."""
TITLE = "Attention Sinks"
TAGLINE = "Why so many heads stare at the first token"
NEXT = "The Residual Stream"
SECTIONS = [
    # 1. Hook
    "In episode eight, we noticed that many of GPT-2's heads stare at the very first token. Today: how much, why, and "
    "why it matters when a model reads a long stream of text.",
    # 2. Measure
    "We feed GPT-2 two hundred and fifty-six tokens of Shakespeare and look at all a hundred and forty-four heads. On "
    "average, thirty-nine percent of the attention goes to the first token. Fifty-nine heads give it more than half. "
    "The first layers barely do it; from the sixth to the eleventh layer, more than half.",
    # 3. Not the content
    "Is the first token special? Replace it with a line break, the word zebra, the word the, or a comma. Every time, "
    "about thirty-nine percent. Cut the text mid-sentence: the same. It's not the word. It's the position.",
    # 4. Why
    "Why? Softmax must add up to one. A head that has nothing useful to look at still has to put its attention "
    "somewhere. The first token is the one every query can see, so the model learns to park attention there: an "
    "attention sink.",
    # 5. No-op
    "And parking there is almost free. The first token's values are small: an average size of one point four, "
    "against six point four for the other tokens. Attending to the sink adds almost nothing.",
    # 6. Streaming
    "Here's why it matters. To read an endless stream, a model can keep a sliding window: the last two hundred and "
    "fifty-six tokens, dropping the oldest, including the first. With full attention, GPT-2's perplexity is forty-"
    "eight. With the window, it jumps to over two thousand. The model collapses.",
    # 7. The fix
    "The fix: always keep the first four tokens, plus the window. Perplexity: forty-three. Four tokens bring the model "
    "back. This trick is known as streaming with attention sinks.",
    # 8. Design
    "Some newer designs give heads an explicit way to attend to nothing, like a learned sink score for each head, so "
    "they don't have to borrow the first token.",
    # 9. Code
    "In code, it's one more term in the mask: a key is allowed if it is in the window, or if it is one of the first "
    "four tokens.",
    # 10. Outro
    "That completes attention. Next, part four: the rest of the block, starting with the residual stream.",
]
