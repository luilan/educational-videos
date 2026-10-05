"""How LLMs Work: Deep Dive, episode 6 — Long Context: Stretching RoPE. One entry per narration section."""
TITLE = "Long Context"
TAGLINE = "Stretching RoPE to read past the training length"
NEXT = "Causal Masking, in Detail"
SECTIONS = [
    # 1. Hook
    "Every model is trained on texts up to some length. For Kwen two point five, that's thirty-two thousand, seven "
    "hundred and sixty-eight tokens. Yet models are used on far longer texts. How can a model read further than it "
    "was ever trained? With Rope, by stretching the rotations. Let's test it, at a scale you can run on a laptop.",
    # 2. Setup
    "We train a tiny GPT on Shakespeare, with Rope and no position table: eight hundred and ten thousand parameters, "
    "trained on only sixty-four characters at a time. Then we make it read two hundred and fifty-six, four times "
    "longer, and measure its loss at every position.",
    # 3. No change
    "First, no change. In the first sixty-four positions, the loss is one point six. Then it climbs: two point one "
    "five, then three point two three. Past about ninety-six, the model falls apart. Those rotation angles were "
    "never seen in training.",
    # 4. Position interpolation
    "Idea one: position interpolation. Squeeze every position by four, so all the angles fall back inside the trained "
    "range. Without retraining, the result is bad: a loss of about three point four everywhere, even in the first "
    "sixty-four positions. Neighboring tokens are now a quarter of a step apart, and the fast pairs can no longer tell "
    "them apart.",
    # 5. NTK scaling
    "Idea two: N T K aware scaling. Instead of squeezing every pair, raise Rope's base. The slow pairs stretch a lot, "
    "and the fast pairs hardly change. With no training at all, the loss stays at one point six up to position one "
    "hundred and twenty-seven, twice the training length, and only then starts to rise.",
    # 6. Fine-tune
    "Idea three: interpolate, then train briefly on long texts. Just two hundred steps at two hundred and fifty-six "
    "tokens. Now the loss is flat across the whole text: one point six two, one point five six, one point five eight. "
    "The model quickly learns the finer grid of angles.",
    # 7. Real models
    "Real models use the same recipe at scale. Kwen two point five's documentation describes Yarn, a refined version "
    "of NTK scaling, to stretch thirty-two thousand tokens to one hundred and thirty-one thousand. Other model "
    "families use their own variants, usually with some long-text training.",
    # 8. Caveats
    "But a longer window isn't free. The KV cache grows with every token, and models still tend to use the middle of "
    "a long text less well. Stretching the window is one thing. Using it well is another.",
    # 9. Code
    "In code, each method is one number. Position interpolation multiplies the positions by a scale. NTK scaling "
    "raises the base. Everything else stays the same.",
    # 10. Outro
    "That's position. Next, part three of the deep dive: attention, inside out, starting with causal masking.",
]
