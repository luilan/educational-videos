"""Video 2 — Tokens: Chopping Text into Pieces. One entry per narration section."""
TITLE = "Tokens: Chopping Text into Pieces"
TAGLINE = "How text becomes numbers a model can read"
SECTIONS = [
    # 1. Recap
    "Last time, we said an LLM predicts the next word. That was a small lie. "
    "It actually predicts the next token. So what is a token? And why not just use words?",
    # 2. Text must become numbers
    "A neural network only understands numbers. So before anything else, we need a way "
    "to turn text into a list of numbers, and back again. That's the tokenizer's job.",
    # 3. Characters
    "The simplest idea: one number per character. The vocabulary is tiny, just a few hundred symbols. "
    "But sequences get long, and a single letter carries almost no meaning. "
    "The model would spend its effort just spelling.",
    # 4. Words
    "The other extreme: one number per word. Now sequences are short, but the vocabulary explodes. "
    "Think of every name, every typo, every new bit of slang. "
    "And a word the model never saw during training? It simply can't read it.",
    # 5. Subwords
    "Modern LLMs take the middle road: subword tokens. Common words get a single token. "
    "Rarer words are built from pieces. In GPT two's tokenizer, tokenization becomes two pieces: token, and ization. "
    "And catnap becomes three: cat... N... and ap.",
    # 6. Byte pair encoding
    "The most popular way to choose these pieces is called byte pair, encoding. "
    "Start with single characters. Count every pair of neighbours in a pile of text. "
    "Then merge the most frequent pair into a brand new token.",
    # 7. Worked example
    "Here's a tiny example. E and S appear side by side three times, as often as any pair. "
    "So we merge them into a single token. Now that new token and T are the top pair. Merge again. "
    "Then L and O. And then that pair and W. "
    "Already, the tokenizer has discovered the word low, and the ending E S T.",
    # 8. Code
    "In code, training a tokenizer is a short loop. Count the pairs. Pick the most frequent one. "
    "Merge it everywhere. And repeat, tens of thousands of times.",
    # 9. Vocabulary and IDs
    "All those merges give us a vocabulary. GPT two has about fifty thousand tokens; "
    "newer models use a hundred thousand or more. Each token gets an ID, "
    "and our sentence becomes a short list of numbers. Notice that spaces are part of the tokens: "
    "cat with a space in front is a different token from cat without one.",
    # 10. A quirk
    "Tokens explain some odd behaviour. Ask a model how many R's are in strawberry, and it may stumble. "
    "That's because it never sees the letters. To GPT two, strawberry, with a space in front, "
    "is one single token.",
    # 11. Outro
    "So now our text is a list of numbers. But an ID like thirty-seven ninety-seven is just a label. "
    "It says nothing about what a cat is. Next time, we'll turn each token into something much richer: "
    "a vector. That's embeddings.",
]
