"""How LLMs Work: Deep Dive, episode 44 — Sparse Autoencoders. One entry per narration section."""
TITLE = "Sparse Autoencoders"
TAGLINE = "Unpacking features from superposition"
NEXT = "Circuits"
SECTIONS = [
    # 1. Hook
    "Last episode showed why single neurons are hard to read: a model stores more features than it has dimensions, "
    "overlapping. A sparse autoencoder tries to unpack them into separate, readable features.",
    # 2. Architecture
    "Take GPT-2's residual stream after layer six: seven hundred and sixty-eight numbers per token. Encode it into six "
    "thousand one hundred and forty-four features, eight times more, then decode it back. The loss: rebuild the input, "
    "plus a penalty on the features that are on.",
    # 3. Trade-off
    "The penalty sets the trade-off. Too weak, and over nine hundred features fire on every token: nothing is "
    "readable. Too strong, and the autoencoder gives up and explains almost nothing. In between: about thirty active "
    "features per token.",
    # 4. Reconstruction
    "Trained on GPT-2's activations over Tiny Shakespeare, it explains eighty percent of the variance on held-out "
    "text, with thirty of six thousand features active per token, and no dead features.",
    # 5. Splice
    "The real test: put the reconstruction back into GPT-2 and let it continue. Next-token loss rises from four point "
    "six nine to four point nine. Replacing the layer with its average gives seven point eight six. So the features "
    "keep ninety-three percent of what the layer contributes.",
    # 6. Features
    "What do the features mean? Pick some at random and look where they fire most. One fires on the word chief. One "
    "on Mess, the start of Messenger, as a speaker's name. One on learned words: arithmetic, history, mechanics, "
    "calendar, scripture. Not every feature is this clean.",
    # 7. Neurons
    "Compare with the raw dimensions. Of a feature's top twenty activations, on average half are the same token; for "
    "a raw dimension, a third. Fourteen percent of features fire on one token in all twenty; one percent of raw "
    "dimensions do.",
    # 8. Code
    "In code: encode with a ReLU, decode with a matrix, and add an L1 penalty on the features, weighted by the length "
    "of each feature's decoder column.",
    # 9. Outro
    "Features are the model's vocabulary. Next, in the final episode: circuits, how parts of the model combine into "
    "an algorithm.",
]
