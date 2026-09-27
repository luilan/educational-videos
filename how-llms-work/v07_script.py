"""Video 7 — Multi-Head Attention. One entry per narration section."""
TITLE = "Multi-Head Attention"
TAGLINE = "Many questions, asked at the same time"
NEXT = "The MLP: Where Facts Live"
SECTIONS = [
    # 1. One head is not enough
    "One attention head asks one kind of question. But when the model reads the word sat, "
    "it might want to know several things at once. Who is sitting? Where? What just happened before?",
    # 2. Several heads
    "So instead of one big attention, we run several smaller ones, side by side. "
    "Each one is called a head, and each has its own query, key and value matrices.",
    # 3. Splitting the vector
    "Here's the trick: the heads share out the vector. GPT two small uses twelve heads "
    "on its seven hundred sixty-eight numbers, so each head works with sixty-four of them.",
    # 4. Different patterns
    "Because each head learns its own matrices, each can learn a different pattern. In trained models, "
    "researchers have found heads that look at the previous token, heads that connect a verb to its object, "
    "and heads that spot a repeated phrase and predict how it continues. Many others are much harder to interpret.",
    # 5. An illustration
    "Here's an illustration. On the word sat, one head looks at cat, the one doing the sitting. "
    "Another looks just one step back. A third spreads its attention widely, over the whole sentence.",
    # 6. Concatenate
    "Each head produces its own sixty-four numbers. We glue them back together, "
    "into one vector of seven hundred sixty-eight.",
    # 7. Output projection
    "Then one more learned matrix, the output projection, mixes what all the heads found, "
    "and the result is added back to the token's vector.",
    # 8. Same cost
    "The nice part: twelve heads of size sixty-four cost about the same as one head of size seven sixty-eight. "
    "We get many points of view, for the price of one.",
    # 9. Code
    "In code, we split the vectors into heads, run attention on every head at once, "
    "then merge the heads and apply the output projection.",
    # 10. Outro
    "Attention lets tokens share information. But once a token has gathered its context, "
    "it needs to work with it on its own. That's the job of the other half of every layer: the MLP. "
    "That's next.",
]
