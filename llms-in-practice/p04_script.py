"""LLMs in Practice, episode 4 — Embeddings for Search: Meaning as Distance. One entry per narration section."""
TITLE = "Embeddings for Search"
TAGLINE = "Meaning as distance"
NEXT = "RAG: Giving the Model a Library"
SECTIONS = [
    # 1. Hook
    "You search your notes for: kitten. Nothing. But you wrote about your cat twice. "
    "Keyword search matches letters, not meaning. Embeddings fix that.",
    # 2. What an embedding is
    "An embedding model reads a whole piece of text, and returns a single vector. In our code, three hundred and "
    "eighty-four numbers. Texts with similar meanings get vectors that point in similar directions. "
    "It's the same idea as token embeddings from How LLMs Work, but for whole sentences.",
    # 3. How it is made
    "Under the hood, it's a small transformer. The text goes in, each token comes out as a vector, "
    "and we average them into one. Then we scale it to length one, so only the direction matters.",
    # 4. The map
    "To look at them, we can squash three hundred and eighty-four dimensions down to two. These are the real positions "
    "of our eight sentences. The three about eggs land together. The two about cats sit side by side. "
    "And the train to Milan is off on its own.",
    # 5. Search
    "Now, a question becomes a vector too. Where does my kitten like to sleep? It lands right next to the cats. "
    "To rank the documents, we take the cosine similarity: for length-one vectors, just a dot product. "
    "The cat on the sofa scores zero point six five. Cats in boxes, zero point six. Dogs, only zero point two one. "
    "And not one word in common.",
    # 6. More queries
    "Ask how long to cook a hard-boiled egg, and the best match is: boil eggs for ten minutes. "
    "Ask about the first departure to Milan, and the train wins easily. Everything else scores close to zero.",
    # 7. Code
    "In code: embed your documents once, and store the vectors. When a question comes in, embed it, "
    "take its dot product with every stored vector, and return the top few.",
    # 8. At scale, and limits
    "With millions of documents, a vector database finds the nearest neighbors fast, by searching approximately. "
    "But close in meaning isn't the same as correct. And for exact things, like names, codes, or numbers, "
    "plain keyword search still wins. Many systems combine both.",
    # 9. Outro
    "Next up: we hand these search results to the model. That's RAG.",
]
