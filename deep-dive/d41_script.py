"""How LLMs Work: Deep Dive, episode 41 — The Logit Lens. One entry per narration section."""
TITLE = "The Logit Lens"
TAGLINE = "Watching a prediction form, layer by layer"
NEXT = "Induction Heads"
SECTIONS = [
    # 1. Hook
    "A transformer builds its prediction layer by layer. What does it think halfway through? The logit lens lets us "
    "peek: read every layer's state as if it were the last one.",
    # 2. Method
    "After each layer, the residual stream holds one vector per token. Normally only the last one is turned into "
    "words. The logit lens applies the same final norm and unembedding to every intermediate vector. No training, "
    "just a peek.",
    # 3. Paris
    "GPT-2 reads: the Eiffel Tower is in the city of. For nine layers, its top guess is filler, mostly the word "
    "the. At layer nine: Rome. Layer ten: London. Layer eleven: Paris. It narrows down: a city, a European capital, "
    "then the right one.",
    # 4. Shakespeare
    "Romeo and Juliet was written by William. The early layers just echo the last word, William. At layer eight, "
    "Shakespeare takes over, and by layer ten it's at a hundred percent. Then the last layer pulls it back to twenty-"
    "one percent. The final layer hedges.",
    # 5. Over text
    "Across a thousand tokens of Shakespeare, how often does each layer's guess match the final prediction? Seven "
    "percent at layer five, thirty at layer eight, fifty-six at layer eleven. Most of the decision is made late.",
    # 6. Qwen
    "Kwen tells a different story. For most of its twenty-four layers, the lens shows junk: code fragments and "
    "random pieces of words, agreeing with the final answer less than three percent of the time until layer sixteen. "
    "Paris only appears at layer twenty-two.",
    # 7. Caveat
    "That doesn't mean Kwen knows nothing until then. Its middle layers use directions the unembedding can't read. "
    "The tuned lens fixes this by training a small translator for each layer. The plain lens is a quick, imperfect "
    "window.",
    # 8. Code
    "In code, the lens is one line: take the hidden state after any layer, apply the final norm, then the output "
    "matrix.",
    # 9. Outro
    "Next: one of the first circuits found inside transformers, the heads that let a model copy from context. "
    "Induction heads.",
]
