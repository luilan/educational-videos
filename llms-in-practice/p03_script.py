"""LLMs in Practice, episode 3 — Sampling: Temperature, Top-p, and Why Answers Vary. One entry per narration section."""
TITLE = "Sampling"
TAGLINE = "Temperature, top-p, and why answers vary"
NEXT = "Embeddings for Search: Meaning as Distance"
SECTIONS = [
    # 1. Hook
    "Ask a model the same question twice, and you can get two different answers. "
    "That's not a bug. It's a choice, made at every single token. It's called sampling.",
    # 2. Real probabilities
    "At each step, the model scores every token in its vocabulary: more than a hundred and fifty thousand of them. "
    "Softmax turns the scores into probabilities. Here are the real ones for: the cat sat on the. "
    "Couch, eight percent. Bed, six. Window, five. And mat? Just two percent, in ninth place. "
    "Real models are rarely as sure as our tidy examples.",
    # 3. Greedy
    "The simplest rule is to always pick the top token. That's called greedy decoding. "
    "Same input, same output, every time. But greedy text tends to be dull, and it can get stuck repeating itself.",
    # 4. Sampling
    "Sampling draws a token at random, weighted by its probability: couch eight percent of the time, "
    "bed six percent, and so on. Run it three times, and you get three different stories.",
    # 5. Temperature
    "Temperature reshapes the odds. We divide the scores by the temperature before the softmax. "
    "At zero point five, the distribution gets sharper: couch jumps to twenty-nine percent. "
    "At two, it flattens out: now couch is under one percent, and every token is a long shot. "
    "Low temperature is focused and predictable. High is creative, and then chaotic.",
    # 6. Top-p
    "There's another problem: the long tail. Thousands of unlikely tokens, each tiny, "
    "but together they add up, and some of them are nonsense. "
    "Top P sampling keeps only the smallest set of top tokens whose probabilities add up to P. "
    "Here, top P of zero point nine keeps three hundred and seventy-eight tokens. Zero point five keeps just nineteen. "
    "Everything else is cut, and we sample from what's left.",
    # 7. Code
    "In code: divide the logits by the temperature, and take the softmax. Sort the probabilities, "
    "add them up, and keep tokens until you reach P. Then draw one at random.",
    # 8. Practical settings
    "So which settings should you use? For facts, code, and extracting data: low temperature, close to greedy. "
    "For brainstorming and stories: a higher temperature, with top P to trim the nonsense. "
    "And for repeatable tests, fix the random seed.",
    # 9. Outro
    "Next up: embeddings, and how meaning becomes a distance you can search.",
]
