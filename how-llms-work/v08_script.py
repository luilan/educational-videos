"""Video 8 — The MLP: Where Facts Live. One entry per narration section."""
TITLE = "The MLP: Where Facts Live"
TAGLINE = "The part of the model that thinks on its own"
NEXT = "The Transformer Block"
SECTIONS = [
    # 1. Recap
    "Attention moves information between tokens. The MLP works on each token by itself, "
    "one at a time, with the same weights at every position. "
    "And it's where much of the model's factual knowledge seems to live.",
    # 2. Three steps
    "MLP stands for multi-layer perceptron. Here, it's just three steps. "
    "Expand the vector to four times its size. Apply a simple nonlinear function. Then project it back down.",
    # 3. Sizes
    "In GPT two small, that's seven hundred sixty-eight numbers, expanded to three thousand and seventy-two, "
    "and back down to seven hundred sixty-eight.",
    # 4. Neurons as detectors
    "Each of those three thousand hidden numbers is a neuron. A neuron takes a dot product with the token's vector, "
    "so it measures how much the vector points in one particular direction. "
    "You can think of it as a question. Is this about animals? Is this the end of a sentence?",
    # 5. GELU
    "The nonlinear function is usually GELU. Large positive inputs pass through almost unchanged, "
    "and negative ones are squashed close to zero. So a neuron stays quiet, unless its pattern really shows up.",
    # 6. Why the bend matters
    "Without that bend, the two matrix multiplications would collapse into a single one, "
    "and stacking layers would add nothing. The nonlinearity is what lets the network learn more than straight lines.",
    # 7. Recalling facts
    "Then the down projection turns each active neuron into a push in some direction. "
    "A simplified picture: if a neuron that detects Eiffel Tower fires, it might add a direction that means Paris. "
    "Researchers have found that facts like this are largely recalled in the MLP layers.",
    # 8. Where the weights are
    "The MLP also holds most of each layer's weights. In GPT two small, every layer's MLP has about "
    "four point seven million weights, twice as many as its attention.",
    # 9. Code
    "In code, it's two matrix multiplications with GELU in between: expand and bend, then project back.",
    # 10. Outro
    "So each layer has two halves: attention, where tokens talk, and the MLP, where each token thinks. "
    "Next time, we'll wire them together into a transformer block, and stack it.",
]
