"""How LLMs Work: Deep Dive, episode 35 — Decoding Strategies Compared. One entry per narration section."""
TITLE = "Decoding Strategies Compared"
TAGLINE = "Greedy, beams, temperature, top-k, top-p, min-p"
NEXT = "Supervised Fine-Tuning"
SECTIONS = [
    # 1. Hook
    "A model gives probabilities for the next token. How you pick from them, the decoding strategy, can turn the same "
    "model into a broken record, a poet, or a random word generator. Let's compare them on the same model.",
    # 2. Setup
    "GPT-2 continues four story openings for a hundred and twenty tokens. We measure two things: repetition, the share "
    "of four-token phrases it has already used, and how surprising a bigger model, Kwen one point five B, finds the "
    "text.",
    # 3. Greedy
    "Greedy decoding always picks the most likely token. The text starts fine, then gets stuck: he was wearing a black "
    "hat with a black belt, again and again. Sixty-nine percent of its phrases are repeats.",
    # 4. Beam
    "Beam search keeps the four best partial texts and returns the most likely overall. It's even more repetitive: "
    "seventy-six percent. Yet the judge finds these texts the least surprising of all. The most likely text is not "
    "the best text.",
    # 5. Temperature
    "Sampling picks randomly, in proportion to the probabilities. Temperature reshapes them first. At zero point "
    "seven, the text is varied and mostly sensible. At one, it wanders into nonsense. At one point five, it's word "
    "salad, and the judge's surprise jumps to nine.",
    # 6. Truncation
    "The problem is the long tail of unlikely tokens. Truncation methods cut it off before sampling. Top-k keeps the "
    "forty most likely tokens. Top-p keeps the smallest set that covers ninety percent of the probability. Min-p "
    "keeps tokens at least a tenth as likely as the top one.",
    # 7. Results
    "All three stay varied: between zero and ten percent repetition. Min-p gets the lowest surprise of the sampling "
    "methods. Its cut adapts: strict when the model is confident, loose when it isn't.",
    # 8. Code
    "In code, each strategy is a few lines that edit the logits before one random draw. Min-p is one line: drop "
    "every token below a tenth of the top probability.",
    # 9. Outro
    "Decoding shapes what a model says. Training shapes what it wants to say. Next: supervised fine-tuning.",
]
