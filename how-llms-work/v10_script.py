"""Video 10 — From Vectors Back to Words. One entry per narration section."""
TITLE = "From Vectors Back to Words"
TAGLINE = "Logits, softmax, and choosing the next token"
NEXT = "Training: Learning from Mistakes"
SECTIONS = [
    # 1. The last vector
    "After the last transformer block, we have one vector per token. To predict what comes after: "
    "the cat sat on the, we only need the last one, the vector sitting on the word: the.",
    # 2. Unembedding
    "First, one last layer norm. Then we multiply by the unembedding matrix, which has one column "
    "for every token in the vocabulary. The result is one score per token: "
    "fifty thousand two hundred fifty-seven numbers, called logits. In GPT two, this matrix is simply "
    "the embedding matrix again, reused.",
    # 3. Logits as dot products
    "Each logit is a dot product: how well the final vector lines up with that token's direction. "
    "The better the match, the higher the score.",
    # 4. Softmax
    "Logits can be any number, positive or negative. Softmax turns them into probabilities. "
    "Exponentiate each one, then divide by the total. Now every token has a probability, and they add up to one.",
    # 5. Greedy
    "The simplest choice is greedy: always take the most likely token. "
    "But that tends to be repetitive and dull.",
    # 6. Temperature
    "Instead, we sample, and we control how adventurous that is with temperature: "
    "divide the logits by a number before the softmax. A low temperature sharpens the distribution "
    "toward the top choice. A high temperature flattens it, and rarer words get a chance.",
    # 7. Top-k and top-p
    "Two more tricks keep sampling sensible. Top k keeps only the k most likely tokens. "
    "Top p keeps the smallest set whose probabilities add up to p, say ninety percent. "
    "Everything else is cut before we sample.",
    # 8. Training uses every position
    "During training, every position predicts its own next token at the same time, "
    "and each prediction is checked against the real next token. "
    "So one sentence gives the model many lessons at once.",
    # 9. Code
    "In code: compute the logits and divide by the temperature. Keep the top k. "
    "Softmax. And sample one token.",
    # 10. Outro
    "And that closes the loop from video one: predict, pick, append, repeat. "
    "We've now seen the whole forward pass. But all those matrices started out as random noise. "
    "How do they learn? Next time: training.",
]
