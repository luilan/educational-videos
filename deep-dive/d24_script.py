"""How LLMs Work: Deep Dive, episode 24 — Where Training Data Comes From. One entry per narration section."""
TITLE = "Where Training Data Comes From"
TAGLINE = "Quantity, quality, and copies"
NEXT = "Data Parallelism"
SECTIONS = [
    # 1. Hook
    "Every model in this series learned from data. Our Tiny Shakespeare is three hundred and thirty-eight thousand "
    "GPT-2 tokens. Meta says Llama 3 was trained on more than fifteen trillion. Where does that much text come from, "
    "and does its quality matter?",
    # 2. Sources
    "Mostly from the public web, crawled at enormous scale, plus books, code and papers. Then it's filtered hard: by "
    "language, by quality, by removing personal information, and by removing duplicates.",
    # 3. Duplicates
    "Duplicates are everywhere. Even in our small file, twenty-three percent of the lines appear more than once: "
    "Gloucester, with a colon, two hundred and twenty-nine times. On the web, it's boilerplate, mirrors, and copies of "
    "copies.",
    # 4. Quality
    "Does quality matter? We replace part of every training batch with junk: random characters. Clean data: a loss of "
    "one point six four four on clean validation text. A quarter junk: one point seven one nine. Half junk: one point "
    "seven seven six. Junk wastes compute, and it hurts.",
    # 5. Duplication experiment
    "Now copies. We take one passage from the validation text, which the model normally never sees, and slip it into "
    "training again and again.",
    # 6. Results
    "Never seen: a loss of one point five six on that passage, and the next character right forty-eight percent of "
    "the time. Two hundred copies: zero point four seven, ninety-four percent right. But that's not quite memorized: "
    "started from its first twenty characters, the model drifts off after one mistake.",
    # 7. Memorized
    "A thousand copies: one hundred percent. Given the first twenty characters, it writes the next forty-five exactly. "
    "Meanwhile the loss on everything else barely moves. The model has memorized one passage word for word.",
    # 8. Why it matters
    "That's why data is deduplicated. Repeated text wastes training, and what a model memorizes it can repeat: private "
    "details, or copyrighted pages.",
    # 9. Code
    "In code, the experiment is one line: every few steps, put the passage into the batch.",
    # 10. Outro
    "That's where the data comes from. Next, part six: training at scale, starting with data parallelism.",
]
