"""LLMs in Practice, episode 2 — Context Windows, and Why They Run Out. One entry per narration section."""
TITLE = "Context Windows"
TAGLINE = "Why a model can only read so much at once"
NEXT = "Sampling: Temperature, Top-p, and Why Answers Vary"
SECTIONS = [
    # 1. Hook
    "Last time we saw that every message resends the whole chat, so the text the model reads keeps growing. "
    "But a model can only read so much at once. That limit is called the context window.",
    # 2. What the window is
    "The context window is the maximum number of tokens the model can handle in one request: "
    "the prompt, plus the reply it writes. For the small model in our code, Kwen two point five, "
    "that's thirty-two thousand, seven hundred and sixty-eight tokens.",
    # 3. How big is that
    "That sounds like a lot. But Tiny Shakespeare, the collection of plays our tiny GPT learned from, "
    "is about three hundred thousand tokens. More than nine full windows.",
    # 4. Why there's a limit: position and attention
    "Why is there a limit at all? First, position. The model was trained on sequences up to a certain length, "
    "and positions beyond that are unfamiliar. Second, attention. Every new token looks back at every token before it, "
    "so double the length, and the attention work roughly quadruples.",
    # 5. Memory
    "Third, memory. For each token in the window, the model keeps its keys and values: the KV cache. "
    "In this small model that's twelve kilobytes per token, so one full window takes about three hundred and "
    "eighty-four megabytes, for a single conversation.",
    # 6. Overflow
    "So what happens when a chat outgrows the window? Something has to go. The usual fix: keep the system prompt, "
    "and drop the oldest turns. The model simply forgets how the conversation started.",
    # 7. Other strategies
    "Smarter apps summarize the old turns into a short note, or store them, and bring back only the pieces "
    "that matter. That's retrieval, and we'll build it in episode five.",
    # 8. Code
    "In code: count the tokens. While the chat is over budget, drop the oldest exchange, but never the system prompt. "
    "Here, a chat of a hundred and thirty-three tokens is trimmed to fit a budget of one hundred.",
    # 9. Practical lessons
    "Bigger windows help, but they aren't free. Long requests are slower and cost more, "
    "and models tend to use information at the start and end of a long context better than in the middle. "
    "So keep the context lean, and put what matters most where the model will notice it.",
    # 10. Outro
    "The context decides what the model reads. Next up: how it chooses what to write, with sampling.",
]
