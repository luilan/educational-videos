"""How LLMs Work: Deep Dive, episode 1 — Byte-Pair Encoding, Step by Step. One entry per narration section."""
TITLE = "Byte-Pair Encoding"
TAGLINE = "How a tokenizer learns its vocabulary, step by step"
NEXT = "Bytes, Unicode, and Why Strawberry Is Hard"
SECTIONS = [
    # 1. Hook
    "In How LLMs Work, we said text is chopped into tokens by something called byte pair, encoding. "
    "In this deep dive, we build it from scratch, and watch it learn, one merge at a time.",
    # 2. Bytes
    "Start with bytes. Any text, in any language, can be written as bytes, and a byte has only two hundred and "
    "fifty-six possible values. So we begin with a vocabulary of exactly two hundred and fifty-six tokens. "
    "Nothing is ever unknown. But it's long: our two hundred thousand characters of Shakespeare are two hundred "
    "thousand tokens.",
    # 3. The algorithm
    "The algorithm fits in one sentence. Count every pair of neighboring tokens. Take the most frequent pair, "
    "give it a new token number, and replace it everywhere. Then do it again.",
    # 4. First merges
    "Here are the real first merges. Number one: the letter e, followed by a space, seen five thousand two hundred "
    "and forty-nine times. It becomes token two hundred and fifty-six. Then t, h. Then t and a space, s and a space, "
    "o, u. Each merge makes the text a little shorter.",
    # 5. After 500 merges
    "After five hundred merges, the text has shrunk to eighty-one thousand tokens. To encode new text, we replay the "
    "merges in the order they were learned. To be, or not to be, that is the question, becomes sixteen tokens. "
    "Question is split into q u, e s, and tion.",
    # 6. The problem
    "But look at the longest tokens it learned: a period, two new lines, the name Coriolanus, a colon, and another "
    "new line. All glued into one token. The merges ignore word boundaries, so punctuation, spacing and names "
    "fuse together.",
    # 7. Pre-tokenization
    "The fix is to split the text into words first, keeping each space at the start of the word, and only merge "
    "inside a word. Now the first merges are: space t, h e, o u. The longest tokens are whole words, like senator. "
    "And our line becomes fifteen tokens: space be, space not, space the. Just like a real tokenizer.",
    # 8. GPT-2
    "GPT-2 uses exactly this recipe, with fifty thousand merges learned from forty gigabytes of web text. "
    "Its vocabulary: two hundred and fifty-six bytes, fifty thousand merges, plus one end of text token. "
    "Our line is thirteen tokens, and question is a single one. On Shakespeare it hasn't seen, GPT-2 needs about "
    "a third fewer tokens than our small tokenizer.",
    # 9. Trade-off
    "So why not merge forever? A bigger vocabulary makes sequences shorter, which means less attention work. "
    "But every token needs its own embedding, and rare tokens get little training. Modern models settle on about "
    "one hundred to two hundred and fifty thousand tokens.",
    # 10. Code
    "In code, training is a loop: count the pairs, find the most common one, and merge it. Encoding replays the "
    "merges. Decoding just joins the bytes back together.",
    # 11. Outro
    "Next up: what bytes really are, and why counting the letters in strawberry is hard for a model.",
]
