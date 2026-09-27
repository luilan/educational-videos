"""Video 13 — Making It Fast: the KV Cache. One entry per narration section."""
TITLE = "Making It Fast: the KV Cache"
TAGLINE = "Why generation doesn't start over for every token"
NEXT = "From GPT to Chatbot"
SECTIONS = [
    # 1. The problem
    "Generating text means running the model once for every new token. Done naively, each step "
    "processes the whole text again, from the very first token. For a long answer, that's a lot of repeated work.",
    # 2. The observation
    "But look at attention with the causal mask. Earlier tokens never look at later ones. "
    "So when a new token arrives, nothing about the old tokens changes. "
    "Their keys and values are exactly the same as last time.",
    # 3. The cache
    "So we save them. That's the KV cache: every time we compute a token's key and value, "
    "in every layer, we store them.",
    # 4. One new row
    "When the next token arrives, we only compute its own query, key and value. "
    "Its query is compared with all the cached keys, and the weights mix the cached values. "
    "One new row, instead of redoing the whole grid.",
    # 5. Why it's fast
    "Each step now processes one token, instead of the entire text. "
    "That's a big part of why chatbots can stream their answers to you so quickly.",
    # 6. The cost: memory
    "The price is memory. The cache holds a key and a value for every token, in every layer. "
    "For GPT two small, that's about eighteen thousand numbers per token. "
    "For big models and long conversations, the cache can take up gigabytes.",
    # 7. Prefill and decode
    "That's why generation has two phases. Pre-fill: read the whole prompt at once, and fill the cache. "
    "Then decode: one token at a time, reusing the cache.",
    # 8. Shrinking the cache
    "Because the cache gets so large, modern models shrink it. One common trick lets several query heads "
    "share the same keys and values. It's called grouped-query attention.",
    # 9. Code
    "In code: compute the new token's query, key and value. Append the key and value to the cache. "
    "Then attend over everything cached so far. No mask is needed, because the cache only holds the past.",
    # 10. Outro
    "One more bonus video to go. A GPT trained on internet text is a brilliant autocomplete, "
    "but it isn't a helpful assistant. Next time: from GPT to chatbot.",
]
