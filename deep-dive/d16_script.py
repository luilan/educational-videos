"""How LLMs Work: Deep Dive, episode 16 — Pre-Norm, Post-Norm, and Stability. One entry per narration section."""
TITLE = "Pre-Norm, Post-Norm, and Stability"
TAGLINE = "Same parts, different order"
NEXT = "Cross-Entropy, Deeper"
SECTIONS = [
    # 1. Hook
    "Where should the normalization go? The original transformer and BERT put it after each block. GPT-2, Llama and "
    "Kwen put it before. Same parts, different order. It turns out to decide whether training survives a high learning "
    "rate.",
    # 2. The two
    "Post-norm: add the block's output to the stream, then normalize the stream itself. Pre-norm: normalize only what "
    "the block reads, and add its output to the stream untouched. Pre-norm then needs one final norm before the output.",
    # 3. The highway
    "In pre-norm, the residual stream from episode thirteen is a clean highway: nothing but additions from the "
    "embedding to the end. In post-norm, every layer's sum passes through a norm, so the direct path is interrupted "
    "twenty-four times in a twelve-layer model.",
    # 4. Low learning rate
    "Let's train a twelve-layer tiny GPT both ways. At a learning rate of zero point zero zero one: pre-norm reaches "
    "one point six seven, post-norm one point six eight. No difference.",
    # 5. High learning rate
    "Now triple the learning rate. Pre-norm: one point six seven again. Post-norm: stuck at three point three six, from "
    "step one hundred to the end. That's the letter-frequency loss we met in episode thirteen. It learned nothing else.",
    # 6. Warmup
    "The classic fix is warmup: start the learning rate near zero, and raise it over the first three hundred steps. "
    "With warmup, post-norm trains: one point seven two. Pre-norm reaches one point six six. Post-norm transformers "
    "were famously hard to train without warmup.",
    # 7. Where it goes wrong
    "Where does it go wrong? At initialization, in our small model, both versions get similar gradients at every layer. "
    "The trouble comes with the first updates: within ten steps, post-norm's loss stalls at three point three eight and "
    "never recovers, while pre-norm keeps falling. Warmup makes those first updates small.",
    # 8. Why pre-norm won
    "That's why modern models use pre-norm: it trains stably at higher learning rates and with less careful tuning, "
    "even when the network gets very deep.",
    # 9. Code
    "In code, the whole difference is where the norm sits. Pre-norm: x plus block of norm of x. Post-norm: norm of x "
    "plus block of x.",
    # 10. Outro
    "That completes the transformer block. Next, part five: training, starting with a closer look at the loss.",
]
