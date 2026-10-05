"""How LLMs Work: Deep Dive, episode 7 — Causal Masking, in Detail. One entry per narration section."""
TITLE = "Causal Masking, in Detail"
TAGLINE = "How a model is kept from reading the answer"
NEXT = "The Attention Matrix, Entry by Entry"
SECTIONS = [
    # 1. Hook
    "A GPT is trained to predict the next token. But during training, it sees the whole text at once, including every "
    "answer. So what stops it from simply peeking at the next word? One small triangle of numbers: the causal mask.",
    # 2. The mask by hand
    "Here are the attention scores for four tokens. Each row is a query, each column a key. Above the diagonal are "
    "the future tokens. The mask sets those scores to minus infinity. Then softmax turns each row into weights. "
    "Since e to the minus infinity is zero, the future gets exactly zero weight, and each row still sums to one.",
    # 3. Why minus infinity
    "Why minus infinity, and not just zero? A score of zero still counts: e to the zero is one. Without the mask, "
    "the first token would put sixty-three percent of its attention on token four, a word that hasn't happened yet.",
    # 4. GPT-2 check
    "Let's check it on the real GPT-2. The cat sat on the mat. Now change the last word to moon. At every earlier "
    "position, the outputs are identical: a difference of exactly zero. Only the last position changes. And here is "
    "a real attention pattern from the first layer: everything above the diagonal is zero, and the first token can only "
    "attend to itself.",
    # 5. The experiment
    "What if we drop the mask? We train the same tiny GPT twice, on Shakespeare: eight hundred and eighteen thousand "
    "parameters, one thousand five hundred steps. The only difference is one flag: the mask on, or off.",
    # 6. Results
    "Without the mask, the training loss falls to zero point zero four. Almost perfect. Too perfect: the model just "
    "reads the next character. Then we test it honestly, giving it only the past. The loss is seven point three six, "
    "worse than guessing at random. With the mask, training gives one point five two, and the honest test one point "
    "seven three.",
    # 7. Generation
    "Ask both to write. The masked model writes something like Shakespeare: names, line breaks, almost words. The "
    "unmasked one has learned to copy, not to predict. Left with nothing to copy, it outputs a long run of empty "
    "lines, then gibberish.",
    # 8. Efficiency
    "The mask also makes training efficient. Every position gets its own prediction, using only what came before. So "
    "one text of sixty-four characters gives sixty-four training examples, all in a single pass.",
    # 9. Inference
    "At inference, the new token is always the last one, so it may see everything before it. With a K V cache, one "
    "query against all the cached keys needs no mask at all. Models like BERT drop the mask on purpose: they see both "
    "sides, but cannot write left to right.",
    # 10. Code
    "In code, the mask is one line: fill the upper triangle with minus infinity before the softmax. In PyTorch's "
    "attention function, it is a single flag: is causal equals true.",
    # 11. Outro
    "That's the mask. Next, we open up the attention matrix itself, entry by entry.",
]
