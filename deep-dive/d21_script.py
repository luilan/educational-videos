"""How LLMs Work: Deep Dive, episode 21 — Batch Size and Gradient Accumulation. One entry per narration section."""
TITLE = "Batch Size and Gradient Accumulation"
TAGLINE = "How many examples per step, and what if they don't fit"
NEXT = "Mixed Precision"
SECTIONS = [
    # 1. Hook
    "Every training step averages the gradient over a batch of examples. How big should that batch be? And what if "
    "the batch you want doesn't fit in memory?",
    # 2. Noise
    "A small batch gives a noisy estimate of the true gradient. We compare each batch's gradient with the average over "
    "four thousand and ninety-six sequences. With one sequence, the similarity is zero point one seven. Sixteen: zero "
    "point five. Sixty-four: zero point seven two. Two hundred and fifty-six: zero point nine. Bigger batches point "
    "more reliably downhill.",
    # 3. Same data
    "But a bigger batch means fewer steps for the same data. We train on forty-eight thousand sequences every time. "
    "Batches of eight: six thousand steps, and a loss of one point six six. Thirty-two: fifteen hundred steps, one "
    "point seven one. A hundred and twenty-eight: three hundred and seventy-five steps, one point nine three.",
    # 4. Learning rate
    "Doubling the learning rate for the big batch helps, to one point eight one, but it still trails. For a fixed "
    "amount of data, more, noisier steps won here.",
    # 5. Speed
    "So why do real models use huge batches? Speed. On this CPU, batches of eight cost two point eight two "
    "milliseconds per sequence. Batches of thirty-two: one point five six. The hardware does more work in parallel. "
    "Here it was already saturated at thirty-two; GPUs keep gaining up to much larger batches.",
    # 6. Trade-off
    "That's the trade-off. Small batches use the data best. Big batches use the hardware best. Large language models "
    "are trained with batches of millions of tokens, and their learning rates are tuned to match.",
    # 7. Accumulation
    "And when the batch doesn't fit in memory, there's a simple trick: gradient accumulation. Run several small "
    "micro-batches, divide each loss by their number, and call backward each time without clearing the gradients. "
    "They add up.",
    # 8. Check
    "Four micro-batches of eight give the same gradient as one batch of thirty-two, to within two hundred-millionths. "
    "The memory of a batch of eight, the gradient of thirty-two. Just no speed-up.",
    # 9. Code
    "In code: a loop over micro-batches, loss divided by their number, backward, and only then one optimizer step.",
    # 10. Outro
    "That's batch size. Next: training in sixteen-bit numbers, and how to do it without losing small updates.",
]
