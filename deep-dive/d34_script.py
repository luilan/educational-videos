"""How LLMs Work: Deep Dive, episode 34 — Speculative Decoding. One entry per narration section."""
TITLE = "Speculative Decoding"
TAGLINE = "A small model guesses, a big model checks"
NEXT = "Decoding Strategies Compared"
SECTIONS = [
    # 1. Hook
    "A big model writes one token per pass, and every pass is slow. Speculative decoding lets a small model guess "
    "ahead, and the big model check the guesses. The output is exactly the same. Only faster.",
    # 2. Why checking is cheap
    "It works because checking is cheap. Our target is Kwen one point five B. One pass for one new token: three "
    "hundred and thirty milliseconds on this CPU. One pass over five tokens: two hundred. Nine tokens: two hundred and "
    "forty. Here, checking a handful of tokens costs no more than writing one.",
    # 3. The algorithm
    "So the small Kwen guesses four tokens, one by one. The big model reads all four in a single pass, and at every "
    "position it says which token it would have picked. Keep the guesses up to the first disagreement, then add the "
    "big model's own token. Every pass gives at least one token, and up to five.",
    # 4. Code results
    "On Python code, the guesses are easy. Plain decoding: forty-four seconds for a hundred and twenty-eight tokens. "
    "With four guesses: ninety-six percent accepted, almost five tokens per pass, nineteen seconds. Two point three "
    "times faster, and the output is identical, token for token.",
    # 5. Prose results
    "Prose is harder to guess. With two guesses, sixty-three percent are accepted, and it's one and a half times "
    "faster. With six guesses, only thirty-seven percent, and it's actually slower than plain decoding.",
    # 6. Trade-off
    "Because the guesses aren't free. Here the draft takes a hundred and thirty milliseconds per token, forty percent "
    "of a big step. Every rejected guess is wasted time. The best number of guesses depends on how predictable the "
    "text is, and how cheap the draft is.",
    # 7. Sampling
    "With sampling, a simple rule keeps the big model's distribution exactly. Accept the guess with probability p "
    "over q, the big model's probability over the draft's. If it's rejected, sample from what's left over. A million "
    "simulated draws: the result matches the target, and seventy-five percent of the guesses are accepted.",
    # 8. Code
    "In code, the check is a loop: run the target once over the guesses, count how many match its own choices, and "
    "keep those, plus its next token.",
    # 9. Outro
    "Greedy, sampling, beams: which decoding strategy should you use? Next: decoding strategies compared.",
]
