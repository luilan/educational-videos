"""How LLMs Work: Deep Dive, episode 3 — Tokenizer Quirks That Shape Model Behavior. One entry per narration section."""
TITLE = "Tokenizer Quirks"
TAGLINE = "Small details in tokenization, big effects on the model"
NEXT = "Why Attention Needs Position"
SECTIONS = [
    # 1. Hook
    "The tokenizer decides what the model sees. Small quirks in that step show up later as strange behavior. "
    "Here are four of them, each tested for real.",
    # 2. Case and spaces
    "Quirk one: the same word, many tokens. In GPT-2, hello, hello with a space in front, and hello with a capital "
    "letter, with and without a space, are four different tokens, with unrelated numbers. The model has to learn, "
    "separately, that they mean the same thing. And hello in capitals isn't even one token: it's three pieces.",
    # 3. Numbers
    "Quirk two: numbers. GPT-2 cuts one, two, three, four, five, six, seven into one two three, four five, and six "
    "seven. Twenty twenty-four becomes twenty, and twenty-four. The pieces don't line up with place value, which makes "
    "arithmetic harder to learn. Kwen two point five splits every number into single digits, so each digit keeps "
    "its place. Many newer models do the same.",
    # 4. Trailing space
    "Quirk three: a trailing space. Give a small Kwen model the text: the capital of France is. Its top guess is "
    "Paris, at thirty percent. Now add one space at the end. Paris drops to twenty-eighth place, below one percent, "
    "and the top guess is the digit one. Words carry their space at the start, so a prompt ending in a lone space "
    "is unusual text, and the model is thrown off.",
    # 5. Glitch tokens
    "Quirk four: glitch tokens. GPT-2's vocabulary contains strange single tokens, like Solid Gold Magikarp, a "
    "user name from a Reddit forum. The vocabulary was built from one collection of text, and the model was trained "
    "on another, where these strings almost never appear. In early models of that family, asking about them "
    "famously produced bizarre answers.",
    # 6. Finding them
    "A token that is never seen in training gets almost no updates, so its embedding stays generic. One rough test: "
    "find the embeddings closest to the average of all of them. In GPT-2, that finds control characters, broken "
    "pieces of bytes, and odd strings like external to EVA, and quick ship, themselves known glitch tokens. "
    "But Solid Gold Magikarp isn't flagged by this test. No single check finds them all.",
    # 7. Lessons
    "So: never end a prompt with a trailing space. Keep spelling and capitalization consistent. For math, prefer "
    "models that split digits. And always look at how your own data is tokenized.",
    # 8. Code
    "In code, every one of these quirks is one line away: tokenize a string, and look at the pieces.",
    # 9. Outro
    "That's tokenization. Next, part two of the deep dive: why attention needs position.",
]
