"""LLMs in Practice, episode 11 — Evaluating LLMs: How We Know It's Better. One entry per narration section."""
TITLE = "Evaluating LLMs"
TAGLINE = "How we know it's better"
NEXT = "Hallucinations and Safety: Where Things Go Wrong"
SECTIONS = [
    # 1. Hook
    "You changed a prompt. Swapped a model. Quantized it. Is it better now? "
    "It looked fine on a few tries is not an answer. You need an evaluation: a test you can run again and again.",
    # 2. Test set
    "Start with questions that have known answers. Ours: twenty easy questions about the bakery handbook, "
    "with the handbook in the prompt, like in episode five. And three models to compare: zero point five, one point five, "
    "and three billion parameters.",
    # 3. Exact match
    "First, the simplest check: is the answer exactly right? The one point five billion model scores one out of "
    "twenty. But look at its answers: the bakery opens at seven thirty AM on weekdays. Correct, just not in the exact "
    "words we expected. The test was wrong, not the model.",
    # 4. Contains
    "So we check whether the answer contains the expected fact. Now every model scores twenty out of twenty. "
    "That's a problem too. A test everyone passes can't tell models apart. It's too easy.",
    # 5. Harder set
    "So we add ten harder questions, each combining two facts. A cake for twelve, delivered on Sunday: what's the total? "
    "Can I order a birthday cake on Thursday for Saturday? Is the bakery open at eight thirty on a Saturday? "
    "Now the scores separate: three, three, and six out of ten. The bigger model pulls ahead.",
    # 6. Read the outputs
    "But read the answers. One model said forty-two euros, and still scored, because thirty-nine appears in its working. "
    "Another said no, because the bakery is closed on Saturdays. Right word, wrong reason. "
    "And one gave the right facts, but started with yes, and was marked wrong. "
    "Checked by hand, the real scores are two, one, and five.",
    # 7. Better checks
    "So make checking easier. Ask for a structured answer, like a single number, or yes or no. "
    "Use several checks. A stronger model can grade answers against a rubric, but check the grader too. "
    "And always read a sample by hand.",
    # 8. Noise
    "Mind the noise. With only ten questions, one answer is ten percent. With sampling, run each question several times. "
    "Small differences on small test sets often mean nothing.",
    # 9. Code
    "In code, an evaluation is a loop. For each question, ask the model, check the answer, and count. "
    "Then print the failures, and read them.",
    # 10. Keep it
    "Keep your test set, and run it after every change: a new prompt, a new model, a quantized version. "
    "That's how you know it's better, and not just different.",
    # 11. Outro
    "Next up, the final episode: hallucinations and safety, and where things go wrong.",
]
