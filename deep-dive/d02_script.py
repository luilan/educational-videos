"""How LLMs Work: Deep Dive, episode 2 — Bytes, Unicode, and Why Strawberry Is Hard. One entry per narration section."""
TITLE = "Bytes, Unicode, and Strawberry"
TAGLINE = "Why a model can't see the letters it writes"
NEXT = "Tokenizer Quirks That Shape Model Behavior"
SECTIONS = [
    # 1. Hook
    "Ask a model how many times the letter r appears in strawberry. For a while, this was a famous way to trip up "
    "language models. Why would counting letters be hard for something that reads text? Because it doesn't read letters.",
    # 2. Unicode
    "Start with characters. Unicode gives every character a number, called a code point. The letter a is ninety-seven. "
    "E with an accent, Russian letters, Chinese characters, and emoji all have their own numbers. "
    "More than a million are possible.",
    # 3. UTF-8
    "UTF-8 writes each code point as one to four bytes. Plain English letters take one byte. "
    "Accented and Russian letters take two. Chinese characters take three, and emoji take four. "
    "The first bits of each byte say how many bytes belong together. "
    "That's why a byte-level tokenizer, with just two hundred and fifty-six starting tokens, can read any language.",
    # 4. Cost per language
    "But languages don't cost the same. Here's one sentence about a cat, in five languages. "
    "For GPT-2, English is ten tokens. Russian is thirty-eight, and Chinese is twenty-five, for the same meaning. "
    "GPT-2 learned its merges mostly from English. Kwen two point five, trained on far more languages, "
    "needs sixteen tokens for the Russian, and only eight for the Chinese. Same meaning, very different price, "
    "and very different use of the context window.",
    # 5. Strawberry tokens
    "Now, strawberry. With a space in front, as it usually appears in a sentence, it's a single token. "
    "Without the space, three: s t r, a w, and berry. Either way, the model never sees the ten letters. "
    "It sees one or three token numbers, and has to have learned how each one is spelled.",
    # 6. The test
    "So we asked Kwen two point five, one and a half billion parameters, directly. How many r's in strawberry? "
    "Three. Correct. But this question is famous, and the answer may simply have been learned. "
    "Bookkeeper and Mississippi were right too. But nevertheless: three e's instead of four. "
    "To the model, nevertheless is just two tokens: never, and theless.",
    # 7. Spelling it out
    "Then we asked it to spell strawberry first. It wrote: s, t, r, o, w, a, b, e. It misspelled the word, "
    "and counted one r. And when we gave it the letters with spaces, so every letter is its own token, "
    "it said four. Seeing the letters is necessary, but a small model still struggles to count them.",
    # 8. Lessons
    "Because tokens hide letters, tasks like spelling, counting characters, rhyming, or reversing a word are "
    "surprisingly hard. Numbers can be split into odd chunks too. The fixes: let the model call a tool, "
    "like a short piece of code, or use a larger model, and check.",
    # 9. Code
    "In code, it's all visible in a few lines: encode a character to see its bytes, run the tokenizer to see the "
    "pieces, and count them for each language.",
    # 10. Outro
    "Next up: more tokenizer quirks, and how they shape what models do.",
]
