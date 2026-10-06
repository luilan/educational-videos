"""How LLMs Work: Deep Dive, episode 30 — Multimodal: Images as Tokens. One entry per narration section."""
TITLE = "Multimodal: Images as Tokens"
TAGLINE = "Teaching a language model to read pictures"
NEXT = "The KV Cache, Deeper"
SECTIONS = [
    # 1. Hook
    "Many language models can now read images. But a transformer only reads sequences of vectors. So how does a "
    "picture become something it can read? The answer: turn the image into tokens. Let's build a tiny one that does "
    "it.",
    # 2. Patches
    "Cut a thirty-two by thirty-two color image into sixteen patches, eight by eight pixels each. A patch is a hundred "
    "and ninety-two numbers. One linear layer turns each patch into a vector of a hundred and twenty-eight numbers, "
    "exactly the size of a text token. Here, that layer is the whole vision encoder.",
    # 3. One sequence
    "Then image and text become one sequence: sixteen image tokens, followed by the characters of a caption. One causal "
    "transformer reads it all, and learns to write the caption: a green circle, top left.",
    # 4. Training
    "We train it on random pictures of circles, squares, triangles and crosses, in three colors and four positions. The "
    "caption loss falls from three point one six to zero point zero four in two hundred and fifty steps.",
    # 5. Test
    "Now two hundred new images, never seen. Color: right every time. Position: every time. Shape: ninety-one percent.",
    # 6. Mistakes
    "Most mistakes are squares called circles, and circles called squares. These shapes are only eight to twelve "
    "pixels wide, and at that size, a corner is a subtle thing.",
    # 7. Real models
    "Real vision-language models follow the same plan at scale. An image is cut into patches of around fourteen or "
    "sixteen pixels, passed through a vision transformer, and projected into the language model's space. A two hundred "
    "and twenty-four pixel image in sixteen-pixel patches is a hundred and ninety-six tokens.",
    # 8. Cost
    "That's the catch: images cost tokens. A detailed, high-resolution image can take hundreds or thousands of them, "
    "competing with text for the context window.",
    # 9. Code
    "In code, it's three steps: cut the image into patches, project each patch with one linear layer, and concatenate "
    "with the text embeddings.",
    # 10. Outro
    "That completes the architecture variants. Next, part eight: inference, starting with a deeper look at the K V "
    "cache.",
]
