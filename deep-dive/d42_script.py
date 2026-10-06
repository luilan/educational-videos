"""How LLMs Work: Deep Dive, episode 42 — Induction Heads. One entry per narration section."""
TITLE = "Induction Heads"
TAGLINE = "The circuit that copies from context"
NEXT = "Superposition"
SECTIONS = [
    # 1. Hook
    "Language models are good at copying: a name mentioned once gets repeated correctly later. One of the first "
    "circuits found inside transformers does exactly that. They're called induction heads.",
    # 2. Test
    "Here's a clean test. Give GPT-2 fifty random tokens. It can't predict them: the loss is twelve point eight. Now "
    "repeat the same fifty tokens. On the second copy, the loss drops to zero point two five. The model copies almost "
    "perfectly from its context.",
    # 3. Rule
    "The rule it uses is simple. The current token is A. Find where A appeared before. Look at the token that came "
    "right after it, B. Predict B.",
    # 4. Find heads
    "An induction head is an attention head that does that lookup. So we score all one hundred and forty-four heads "
    "by how much they attend to the token right after the earlier occurrence. Most score near zero. Five score above "
    "point eight, all in layers five to seven.",
    # 5. Circuit
    "Why not in layer zero? To find the token after A, a head must know, at every position, what the previous token "
    "was. In the original analysis, an earlier head writes that in, and the induction head reads it. It's a circuit "
    "of two heads in different layers.",
    # 6. Ablation
    "Now remove them. Remove the top two induction heads, and the second-copy loss rises from zero point two five to "
    "zero point three five. The top four: one point seven eight. The top six: five point eight. Removing six random "
    "heads instead: only zero point six six.",
    # 7. Why it matters
    "Induction heads are a building block of in-context learning: using patterns from earlier in the prompt. Research "
    "on small models found that they form suddenly during training, at the same moment in-context learning improves.",
    # 8. Code
    "In code, the induction score is one line: in the second copy, average the attention from each token to the "
    "position fifty minus one tokens back.",
    # 9. Outro
    "A head does one clear job here. But most neurons don't. Next: superposition.",
]
