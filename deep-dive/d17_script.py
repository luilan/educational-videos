"""How LLMs Work: Deep Dive, episode 17 — Cross-Entropy, Deeper. One entry per narration section."""
TITLE = "Cross-Entropy, Deeper"
TAGLINE = "What the loss number really measures"
NEXT = "Backprop Through a Transformer"
SECTIONS = [
    # 1. Hook
    "Every training run in this series has printed one number: the loss. Part five of the deep dive is about "
    "training, so let's start by taking that number apart.",
    # 2. Token by token
    "The loss of one token is minus the log of the probability the model gave to the right token. Here's GPT-2 on: "
    "the capital of France is Paris, and the capital of Italy is Rome. After the capital of France is, Paris gets "
    "three percent: a loss of three point four. By the second sentence, the model has caught on: Rome gets fifty-"
    "eight percent, a loss of zero point five five.",
    # 3. Average
    "The loss we print is the average over all tokens: two point three four for this sentence.",
    # 4. Perplexity and bits
    "Two other names for the same number. Perplexity is e to the loss: ten point four. As if the model were choosing "
    "evenly among ten tokens at each step. And dividing by the log of two gives bits: three point three eight bits "
    "per token. On Shakespeare, GPT-2 needs one point eight seven bits per character.",
    # 5. Confident mistakes
    "The log punishes confident mistakes hard. Give the right token ninety percent: a loss of zero point one one. Ten "
    "percent: two point three. One percent: four point six. One in a thousand: six point nine. On Shakespeare, the "
    "worst ten percent of tokens make up a third of the total loss.",
    # 6. Calibration
    "A good loss rewards honest probabilities. When GPT-2's top guess gets about thirty percent, it's right thirty-"
    "three percent of the time. Around seventy percent, it's right sixty-seven percent. Well calibrated.",
    # 7. Except
    "Except at the top. When it's ninety-six percent sure, it's right only sixty-seven percent of the time. Almost "
    "all of those confident mistakes are the same one: GPT-2 expects a blank line after each line of Shakespeare, "
    "and this file has none.",
    # 8. Gradient
    "Finally, the gradient. For each logit, the gradient of the loss is simply the predicted probability, minus one "
    "for the right token. Softmax minus one-hot. Autograd agrees, to every digit shown. That's the signal that starts "
    "every backward pass.",
    # 9. Code
    "In code: take the log softmax of the logits, pick the right token's value, negate, and average. PyTorch's cross "
    "entropy does exactly that.",
    # 10. Outro
    "That's the loss. Next: how that signal travels backward through every layer of a transformer.",
]
