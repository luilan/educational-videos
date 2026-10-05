"""How LLMs Work: Deep Dive, episode 9 — Multi-Query and Grouped-Query Attention. One entry per narration section."""
TITLE = "Multi-Query and Grouped-Query Attention"
TAGLINE = "Sharing keys and values to shrink the KV cache"
NEXT = "FlashAttention: Same Math, Less Memory"
SECTIONS = [
    # 1. Hook
    "When a model writes, it keeps a key and a value for every token, in every layer and every head: the K V cache. "
    "On long texts, that cache gets huge, and reading it slows every step. The fix used by most modern models: let "
    "heads share their keys and values.",
    # 2. The idea
    "In classic multi-head attention, every query head has its own keys and values. In multi-query attention, all "
    "the query heads share a single key and value head. Grouped-query attention sits in between: query heads are "
    "split into groups, and each group shares one.",
    # 3. Qwen
    "Kwen two point five, half a billion parameters, really does this. Twenty-four layers, fourteen query heads, but "
    "only two key and value heads: groups of seven. Its cached keys have two heads, not fourteen. That's twelve "
    "thousand bytes per token, measured. With a key and value per query head, it would be eighty-six thousand: seven "
    "times more.",
    # 4. At scale
    "At thirty-two thousand tokens, the cache takes three hundred and eighty-four megabytes, instead of more than "
    "two and a half gigabytes. At a hundred and thirty-one thousand tokens: one and a half gigabytes, instead of more "
    "than ten.",
    # 5. Why it works
    "Why does this work? Each query head still asks its own question. Only the keys and values it looks up are "
    "shared. Seven different questions, asked of the same index.",
    # 6. From scratch
    "Let's measure the cost. We train the same tiny GPT three times, with eight query heads and eight, two, or one "
    "key and value heads. The validation loss: one point six seven three, one point six eight five, one point seven "
    "zero four. The cache shrinks from one thousand and twenty-four numbers per token to two hundred and fifty-six, "
    "then one hundred and twenty-eight. A small price for an eight times smaller cache.",
    # 7. After training
    "But sharing has to be learned. Take GPT-2, and simply average its keys and values within each group, with no "
    "retraining. The loss jumps from three point eight to above six point four. Each head was trained to expect its "
    "own keys. The grouped-query paper fixes this with a short extra round of training after the conversion.",
    # 8. Speed
    "Why it matters for speed: generating a token means reading the whole cache from memory. A smaller cache means "
    "faster steps, and room for more users on the same GPU.",
    # 9. Code
    "In code, the key and value projections simply output fewer heads, and each one is repeated to match its group "
    "of queries. The rest of attention is unchanged.",
    # 10. Outro
    "That's grouped-query attention. Next: FlashAttention, the same math with far less memory.",
]
