"""Video 5 — Attention I: Tokens Talking to Each Other. One entry per narration section."""
TITLE = "Attention I: Tokens Talking to Each Other"
TAGLINE = "How words borrow meaning from their neighbours"
NEXT = "Attention II: The Math"
SECTIONS = [
    # 1. Recap
    "Each token now has a vector that says what it is, and where it is. "
    "But each vector was made on its own, without looking at any other word. And words don't work like that.",
    # 2. Context changes meaning
    "Take the word bank. In: I sat on the river bank, it means the edge of a river. "
    "In: I paid money into the bank, it's a place that keeps your money. "
    "Same token, same starting vector, but two very different meanings.",
    # 3. The idea
    "Attention lets every token look at the other tokens, and pull in information from the ones that matter. "
    "Bank looks around, notices river, and updates its own vector to mean: riverbank.",
    # 4. Attention weights
    "Each token decides how much to listen to every other token. These attention weights add up to one, "
    "like a budget. Bank might spend most of its attention on river, and very little on the, or on.",
    # 5. Query, key, value
    "How does a token decide? Every token produces three new vectors. A query: what am I looking for? "
    "A key: what do I contain? And a value: what will I share, if someone listens to me?",
    # 6. Library analogy
    "Think of a library. Your query is the question you bring. Each book has a key, like the title on its spine. "
    "You compare your question with every title, and the better the match, "
    "the more you read from that book's contents: its value.",
    # 7. Matching with dot products
    "The match is measured with the dot product we met in the embeddings video. "
    "If bank's query points in the same direction as river's key, the score is high, "
    "and bank pays attention to river.",
    # 8. Updating the vector
    "Then bank collects a weighted mix of all the values, mostly river's, and adds it to its own vector. "
    "Now its vector doesn't just mean bank. It means bank, next to a river.",
    # 9. Only looking back
    "One rule for language models: a token can only look backwards. When predicting the next word, "
    "the future isn't written yet. So cat can look at the, but never at sat.",
    # 10. Learned
    "Where do the queries, keys and values come from? Each one is made by multiplying the token's vector "
    "by a matrix, and those matrices are learned in training. Nobody tells the model that river explains bank. "
    "It figures that out from data.",
    # 11. Outro
    "That's the idea of attention: tokens asking questions, and borrowing meaning from each other. "
    "Next time, we'll open it up and do the actual math, step by step.",
]
