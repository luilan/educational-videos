"""How LLMs Work: Deep Dive, episode 36 — Supervised Fine-Tuning. One entry per narration section."""
TITLE = "Supervised Fine-Tuning"
TAGLINE = "From text predictor to assistant"
NEXT = "Reward Models"
SECTIONS = [
    # 1. Hook
    "A base model only continues text. Ask it a question, and it may repeat the question, or ramble forever. "
    "Supervised fine-tuning turns it into an assistant that answers, and stops. Let's do it.",
    # 2. Base behaviour
    "Here's the base Kwen half B model, asked sixty questions in a chat format. Zero answers in the right form, and it "
    "never stops: it repeats the question and emits junk. Yet with a plain question and answer prompt, the same model "
    "gets the right answer in seventy-five percent of cases. The knowledge is there. What's missing is the behaviour.",
    # 3. Data
    "So we write examples of the behaviour we want: two hundred and eighty short conversations, with additions, "
    "capitals, and capital letters. Special tokens mark where the user speaks, where the assistant speaks, and where "
    "the answer ends.",
    # 4. Masking
    "The training is ordinary next-token prediction, with one twist: the loss counts only the assistant's tokens. "
    "Here that's thirty-one percent. The model isn't taught to write questions, only to answer them. And the end "
    "token is part of the answer, so it learns to stop.",
    # 5. Training
    "Seventy-five steps of eight examples. The answer loss drops from four point two four to zero point one four, in "
    "a few minutes on this CPU.",
    # 6. Results
    "Now the sixty held-out questions, with numbers, countries and words it never saw in training. Ninety-five "
    "percent exact answers. It stops every time, after seven tokens on average.",
    # 7. Failures
    "The three misses are telling. Forty-eight plus forty-five, it says a hundred and thirteen. Kiev instead of "
    "Kyiv. Flower in capitals becomes floral. Fine-tuning taught the format; the knowledge and skills still come from "
    "pre-training.",
    # 8. Code
    "In code, the mask is one line: set every label outside the answer to minus one hundred, and cross-entropy "
    "ignores it.",
    # 9. Outro
    "Fine-tuning copies examples. But how do you teach a model what people prefer? Next: reward models.",
]
