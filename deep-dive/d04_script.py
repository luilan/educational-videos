"""How LLMs Work: Deep Dive, episode 4 — Why Attention Needs Position. One entry per narration section."""
TITLE = "Why Attention Needs Position"
TAGLINE = "Proof, with real GPT-2 weights, that attention is blind to order"
NEXT = "RoPE: Rotating Vectors to Encode Order"
SECTIONS = [
    # 1. Hook
    "Dog bites man. Man bites dog. Same three words, opposite news. Word order is everything. "
    "But attention, the heart of the transformer, has no idea what order its inputs are in. "
    "Let's prove it, with real GPT-2 weights.",
    # 2. Why attention is order-blind
    "Here's why. Each token's output is a weighted average of value vectors, and the weights come from dot products "
    "between queries and keys. Shuffle the input tokens, and you get the same dot products, just shuffled. "
    "A weighted sum doesn't care about order. Mathematicians call this permutation equivariance.",
    # 3. The experiment
    "The experiment. Take GPT-2 small, all twelve layers. Switch off its position embeddings, and let every token see "
    "every other. Feed in both sentences. Now compare the final vector for dog in the first sentence with dog in the "
    "second. Cosine similarity: one point zero zero zero zero zero zero. Identical. The same for man. "
    "The biggest difference anywhere is two ten-thousandths: just rounding. The model cannot tell who bit whom.",
    # 4. Positions back on
    "Switch the position embeddings back on, and dog in position one is no longer the same as dog in position three. "
    "The similarity drops to zero point nine six. Now order can matter.",
    # 5. The causal twist
    "There's a twist. Models like GPT use a causal mask: each token sees only the tokens before it. "
    "That alone leaks some order. The first token sees only itself, the last sees everything. "
    "With the mask on and no position embeddings, the two dogs are no longer identical: zero point nine eight seven. "
    "Some models are trained this way, without explicit positions. But it's an indirect signal.",
    # 6. What GPT-2 learned
    "So what did GPT-2 actually learn? One vector for each position, one thousand and twenty-four of them, "
    "each with seven hundred and sixty-eight numbers, learned from scratch. And a smooth pattern emerges. "
    "Position one hundred is almost identical to one hundred and one. Still very close to one hundred and ten. "
    "Less so to one hundred and fifty. And at three hundred, the similarity is negative.",
    # 7. The limit
    "But a learned table has a hard limit. GPT-2 has no vector for position one thousand and twenty-five, "
    "so it can't read anything longer. Modern models use a cleverer idea: they encode position by rotating the "
    "queries and keys, so attention sees relative distance, and the range can be stretched.",
    # 8. Code
    "In code, the whole experiment is a switch: add the position embeddings to the token embeddings, or don't, "
    "run the blocks, and compare.",
    # 9. Outro
    "Next up: Rope, and how rotating vectors encodes order.",
]
