"""Video 1 — What is an LLM? One entry per narration section."""
TITLE = "What is an LLM?"
TAGLINE = "How a language model writes, one word at a time"
NEXT = "Tokens: Chopping Text into Pieces"
SECTIONS = [
    # 1. Hook
    "You type a question, and a large language model writes back an answer, word by word. "
    "It can feel like magic, or like someone is in there typing. "
    "But at its core, an LLM does one surprisingly simple thing, over and over.",
    # 2. Next-word prediction
    "It predicts the next word. Give it the words: the cat sat on the... "
    "and it answers just one question. What comes next?",
    # 3. Probabilities
    "Actually, it doesn't pick a single word. It gives a probability to every word it knows. "
    "Mat gets a high score. Floor, a little less. Sofa, less still. "
    "And banana? Almost nothing.",
    # 4. Pick and append
    "Then we pick one of them, usually one of the likely ones, and add it to the end of the text.",
    # 5. The loop
    "And now, the trick. We feed the whole thing back in. "
    "The model reads: the cat sat on the mat, and predicts the next word again. Maybe a period. "
    "Then another word, and another. That's all text generation is: "
    "predict, pick, append, repeat.",
    # 6. Code
    "In code, the whole loop fits in a few lines. We start with some tokens. "
    "We ask the model for probabilities, sample one, and append it. "
    "Then we go around again, until we reach a limit, or the model says it's done.",
    # 7. The mystery
    "So everything interesting hides inside this one function: model. "
    "Underneath, it's just a huge pile of numbers. So how can a pile of numbers "
    "read: the cat sat on the, and know that mat is likely? That's what this series is about.",
    # 8. Series map
    "First, we'll chop text into tokens. Each token becomes a vector, called an embedding, "
    "and we'll add information about its position. Then comes the heart of it all: attention, "
    "where words look at each other. Next, the MLP, where facts are stored. "
    "Together, they form a transformer block, stacked many times. "
    "Finally, we turn vectors back into probabilities, see how training sets all those numbers, "
    "and build a tiny GPT of our own.",
    # 9. Outro
    "Next up: tokens. How do you turn text into something a computer can actually work with? "
    "See you there.",
]
