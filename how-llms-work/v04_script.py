"""Video 4 — Where Am I? Position. One entry per narration section."""
TITLE = "Where Am I? Position"
TAGLINE = "How a model knows the order of words"
SECTIONS = [
    # 1. The problem
    "The cat sat on the mat. The mat sat on the cat. Same tokens, completely different meaning. "
    "So far, our token vectors can't tell these apart.",
    # 2. Why it matters
    "And it matters. Attention, the heart of the transformer that we'll meet next, "
    "compares every token with every other token. Nothing in that comparison says which one came first. "
    "We have to put the order into the vectors themselves.",
    # 3. Raw position numbers
    "The most obvious idea: just add the position number. One, two, three, and so on. "
    "But these numbers grow without limit. By token five thousand, the position would drown out the meaning.",
    # 4. Learned positions
    "Another option, used by GPT two, is to learn it. Keep a second table, with one vector for each position, "
    "and add it to the token's embedding. Simple, but the model can't handle positions it never saw in training.",
    # 5. Sinusoidal waves
    "The original transformer paper used a neat trick instead: waves. "
    "Each pair of dimensions follows a sine and a cosine wave, and every pair has its own frequency. "
    "To encode a position, you read off where each wave is at that point.",
    # 6. Clock analogy
    "It works like a clock. The second hand moves fast, the minute hand slower, the hour hand slowest. "
    "Each hand alone is ambiguous, but read them together and you know the exact time. "
    "Fast waves tell neighbours apart; slow waves track where you are in the long run.",
    # 7. Adding position to meaning
    "This position vector is simply added to the token's embedding. "
    "So cat at position two, and cat at position six, start out as different vectors: "
    "the same meaning, in a different place.",
    # 8. Code
    "In code, it's a few lines. For each position and each pair of dimensions, compute an angle. "
    "Put the sine in the even dimensions, and the cosine in the odd ones. "
    "Then add the result to the embeddings.",
    # 9. RoPE
    "Most modern models, like Llama, use a newer method called rotary position embedding, or rope. "
    "Instead of adding a vector, it rotates pairs of numbers by an angle that grows with position. "
    "When two tokens are compared, only the difference between their angles matters, "
    "so the model directly sees how far apart they are.",
    # 10. Outro
    "Now every token vector carries two things: what it is, and where it is. "
    "We're finally ready for the heart of the transformer. Next time: attention.",
]
