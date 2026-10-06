"""How LLMs Work: Deep Dive, episode 31 — The KV Cache, Deeper. One entry per narration section."""
TITLE = "The KV Cache, Deeper"
TAGLINE = "Prefill, decode, and what the cache really saves"
NEXT = "Batching and Paged Attention"
SECTIONS = [
    # 1. Hook
    "Part eight of the deep dive is about inference: running a trained model. To write each new token, the model needs "
    "the keys and values of every token before it. The K V cache keeps them, instead of recomputing them. Let's measure "
    "what that buys, and what it costs.",
    # 2. Same output
    "GPT-2 writes two hundred tokens, greedily, twice: with the cache and without it. The output is identical, token "
    "for token. But with the cache it takes seven point four seconds, and without it, eighteen. Two point four times "
    "slower.",
    # 3. Per step
    "Look at each step. With the cache, every new token costs about thirty-six milliseconds, however long the text "
    "is. Without it, the cost keeps growing: forty-three milliseconds at the start, eighty-eight at a hundred tokens, a "
    "hundred and thirty-eight at two hundred. It's recomputing the whole text, every time.",
    # 4. Contents
    "What's inside? For each of the twelve layers, the keys and the values: twelve heads, sixty-four numbers each, for "
    "every token. That's seventy-three thousand bytes per token in thirty-two bit floats. Seventy-two megabytes at a "
    "thousand and twenty-four tokens, for one conversation.",
    # 5. Prefill vs decode
    "Now the prompt. Reading a five hundred and twelve token prompt in one pass, called prefill, takes zero point two "
    "nine seconds: seventeen hundred tokens per second. Feeding the same prompt one token at a time takes nearly "
    "nineteen seconds. Sixty-four times slower.",
    # 6. Why
    "Prefill processes all the prompt's tokens in parallel: big matrix multiplications that keep the hardware busy. "
    "Decoding can only produce one token at a time, and each step must read all the weights and the whole cache for "
    "very little math. Here, twenty-seven tokens per second.",
    # 7. Consequences
    "That's why a chatbot pauses before its answer, and then streams it word by word. And it's why the cache's size "
    "limits how long a context can be and how many users fit on one machine. The tricks from earlier episodes, shared "
    "keys and values, and sliding windows, all shrink it.",
    # 8. Code
    "In code, the cache is what the model returns as past key values. Pass it back in, and feed only the newest token.",
    # 9. Outro
    "That's the K V cache. Next: serving many users at once, with batching and paged attention.",
]
