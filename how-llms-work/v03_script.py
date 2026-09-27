"""Video 3 — Embeddings: Words as Vectors. One entry per narration section."""
TITLE = "Embeddings: Words as Vectors"
TAGLINE = "How numbers start to carry meaning"
SECTIONS = [
    # 1. IDs are just labels
    "Our sentence is now a list of token IDs. But an ID is just a label. "
    "Cat is token thirty-seven ninety-seven, and its neighbours are rief and esc. "
    "The numbers mean nothing. We need numbers that actually carry meaning.",
    # 2. Vectors as points
    "The idea: give every token a vector, a list of numbers, like coordinates. "
    "Then each token becomes a point in space.",
    # 3. Lookup table
    "These vectors are stored in a big table called the embedding matrix, "
    "with one row for every token in the vocabulary. Embedding a token is just a lookup: "
    "token thirty-seven ninety-seven grabs row thirty-seven ninety-seven.",
    # 4. Real sizes
    "Real vectors are long. The smallest GPT two uses seven hundred sixty-eight numbers per token, "
    "and the largest models use many thousands. We'll draw just two or three, so we can see them.",
    # 5. Meaning as position
    "Here's the key idea. After training, tokens that are used in similar ways end up close together. "
    "Cat lands near dog and kitten. Mat, near rug and carpet. "
    "Numbers gather in one place, verbs in another.",
    # 6. Dot product and cosine similarity
    "How do we measure close? With the dot product: multiply matching coordinates, then add them up. "
    "Vectors pointing the same way score high. Unrelated ones score near zero, "
    "and opposite ones go negative. Divide by their lengths, and you get cosine similarity: "
    "the cosine of the angle between them.",
    # 7. Directions carry meaning
    "Directions can carry meaning too. In classic word embeddings like word two vec, "
    "the step from man to woman is roughly the same as the step from king to queen. "
    "So king, minus man, plus woman, lands close to queen.",
    # 8. Learned, not designed
    "Nobody writes these numbers by hand. They start out random. "
    "Then training nudges them, a tiny bit at a time, until the geometry reflects how words are actually used.",
    # 9. Code
    "In code, it's a single line. The embedding matrix has one row per token. "
    "Index it with our token IDs, and we get one vector per token: "
    "a five by seven sixty-eight array for our sentence.",
    # 10. Outro
    "So every token is now a point in a space of meaning. But there's a catch. "
    "The cat sat on the mat, and the mat sat on the cat, contain exactly the same tokens, "
    "so they get exactly the same vectors. Nothing says which came first. Next time: position.",
]
