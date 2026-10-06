"""How LLMs Work: Deep Dive, episode 43 — Superposition. One entry per narration section."""
TITLE = "Superposition"
TAGLINE = "More features than dimensions"
NEXT = "Sparse Autoencoders"
SECTIONS = [
    # 1. Hook
    "Look inside a language model, and single neurons often respond to many unrelated things. Why would a network "
    "mix things up like that? One answer: superposition. The model stores more features than it has dimensions.",
    # 2. Toy model
    "A toy model shows it. Five features go in. The model squeezes them into just two numbers, then tries to read all "
    "five back out, through a ReLU. Each feature gets a direction in the two-dimensional space. The first features "
    "matter most.",
    # 3. Dense
    "When features are active all the time, there's no room to share. The model keeps the two most important, at "
    "right angles, and drops the other three entirely.",
    # 4. Sparse
    "Now make the features rare: each one is zero most of the time. At seventy percent sparsity, the model fits four "
    "features into two dimensions. At ninety-seven percent, all five, spread about seventy-two degrees apart: a "
    "pentagon.",
    # 5. Why it works
    "Why does that work? The directions overlap, so reading one feature picks up a bit of the others. But if features "
    "are rarely active together, that interference rarely happens, and the ReLU and bias filter out the small leftovers.",
    # 6. Bigger
    "Scale it up: a hundred equally important features into twenty dimensions. Dense: only fourteen represented. Half "
    "the time zero: forty, in pairs pointing in opposite directions. Ninety percent: ninety-five. Ninety-five percent "
    "and up: all one hundred, five per dimension.",
    # 7. Consequence
    "Real features, like a topic, a language, or a grammar rule, are sparse. So large models likely pack many more "
    "features than neurons, and each neuron ends up shared: polysemantic. To read the features, we need to unpack them.",
    # 8. Code
    "In code, the toy model is one line: multiply by W, multiply back by W transposed, add a bias, and apply a ReLU.",
    # 9. Outro
    "How do we unpack features from a real model? Next: sparse autoencoders.",
]
