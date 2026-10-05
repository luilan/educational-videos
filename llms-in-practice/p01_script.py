"""LLMs in Practice, episode 1 — Prompts Are Just Context. One entry per narration section."""
TITLE = "Prompts Are Just Context"
TAGLINE = "What the model really sees when you chat"
NEXT = "Context Windows, and Why They Run Out"
SECTIONS = [
    # 1. Hook
    "When you chat with an assistant, it looks like a conversation: speech bubbles, turns, "
    "and a memory of what you said. But the model never sees any of that. It sees one long piece of text.",
    # 2. The chat template
    "Here's a real example. A system message sets the role. You ask how long to boil an egg. "
    "The assistant answers, and you ask a follow-up. Before the model sees it, the app flattens "
    "everything into a single document, with special tokens that mark who is speaking.",
    # 3. The open turn
    "Look at the very end. The document stops right after a new assistant marker. "
    "The most natural way to continue this document is the assistant's reply. "
    "That's all answering is: continuing the text.",
    # 4. Tokens
    "Then the document becomes tokens. Fifty-five of them, and each one is just a number. "
    "That list of numbers is the model's entire world.",
    # 5. No memory
    "And here's the surprise: the model has no memory between messages. Every time you send one, "
    "the app sends the whole conversation again, from the top. Your first question was twenty-eight tokens. "
    "The follow-up, fifty-five. A chat gets longer to read on every single turn.",
    # 6. Everything is context
    "So everything the model knows about you, right now, lives in that text: the system prompt, "
    "earlier turns, documents you pasted, results from tools. "
    "If it isn't in the context, the model can't see it.",
    # 7. Code
    "In code, it's three steps. Start with a list of messages. Apply the chat template to get one string. "
    "Then tokenize it, and count. The code for this episode is in the repository, linked below.",
    # 8. What this means for prompting
    "That's why prompting works the way it does. You're not giving orders to a mind. "
    "You're writing the beginning of a document, so that the answer you want is its most likely continuation. "
    "Clear instructions and a few good examples make that continuation easy to predict.",
    # 9. Outro
    "But that window of text has a limit. Next up: context windows, and why they run out.",
]
