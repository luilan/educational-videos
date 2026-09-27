"""Foundations F14 (extra) — Train vs Validation Data."""
TITLE = "Train vs Validation Data"
TAGLINE = "How we know a model really learned"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F14  ·  EXTRA"
NEXT = "Episode 1 · What is an LLM?"
USED_IN = [(11, "Training: Learning from Mistakes"), (12, "Build a Tiny GPT")]
SECTIONS = [
    "Episode eleven showed two loss curves: one for training, and one for validation. Why two? "
    "Because a low training loss, on its own, can fool us.",
    "A model could simply memorize its training text, word for word. "
    "It would score perfectly on that text, and be useless on anything new.",
    "So before training, we split the data. Most of it, the training set, is used for learning. "
    "A slice we never train on, the validation set, is kept aside to test with.",
    "For our tiny GPT, ninety percent of the Shakespeare text, about a million characters, was for training. "
    "The last ten percent, about a hundred and eleven thousand characters, was only ever used to measure.",
    "During training, we measure the loss on both. The training loss says how well the model fits what it has seen. "
    "The validation loss says how well it does on text it has never seen.",
    "Our tiny GPT ended at about one point three on training text, but about one point six on validation text. "
    "That gap is normal: a model usually does a little better on what it studied.",
    "But if the gap keeps growing, while the validation loss stops falling, or even rises, the model is overfitting: "
    "memorizing instead of learning. That's the time to stop, or to find more data.",
    "Careful projects keep a third slice, the test set, untouched until the very end, for one final, honest score.",
    "In code, the split is two lines: the first ninety percent for training, the rest for validation.",
    "You'll see both curves in episode eleven, and this exact split in the tiny GPT of episode twelve. "
    "That completes the foundations. Next up: episode one, what is an LLM?",
]
