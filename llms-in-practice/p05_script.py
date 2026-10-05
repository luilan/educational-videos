"""LLMs in Practice, episode 5 — RAG: Giving the Model a Library. One entry per narration section."""
TITLE = "RAG: Giving the Model a Library"
TAGLINE = "Search first, then answer"
NEXT = "Chunking and Retrieval: Why RAG Fails, and Fixes"
SECTIONS = [
    # 1. Hook
    "A model only knows what was in its training data, up to a cutoff date. "
    "It has never seen your company's documents, your notes, or today's news. "
    "So how do you get it to answer questions about them?",
    # 2. Without RAG
    "Here's a made-up bakery. We ask: can I buy gluten-free bread at Bella's Bakery on Wednesday? "
    "Without help, the model can only say it doesn't know, or worse, invent something that sounds right.",
    # 3. The idea
    "The fix is called RAG: retrieval-augmented generation. Two steps. First, search a library of documents "
    "for the parts that match the question. Then paste them into the prompt, and let the model answer from them.",
    # 4. Retrieval
    "The search is exactly last episode's embeddings. Our library has six facts about the bakery. "
    "The question becomes a vector, and the closest two are: gluten-free bread is only available on Fridays, "
    "with a score of zero point seven, and the opening hours, at zero point six two.",
    # 5. The prompt
    "Then we build the prompt: an instruction to answer only from the information below, "
    "and to say so if the answer isn't there. Then the retrieved facts, and finally the question. "
    "It's one hundred and one tokens, all of it context, just like episode one.",
    # 6. The answer
    "A small model, Kwen two point five, one point five billion, now answers: "
    "No, you cannot buy gluten-free bread on Wednesday, because it is only available on Fridays. Correct, "
    "and based on a fact it never saw in training.",
    # 7. The twist
    "But here's a warning. We gave the exact same prompt to the even smaller zero point five billion model. "
    "It said: yes, you can, because the bakery is open on weekdays. It had the right fact, and still got it wrong. "
    "Retrieval only helps if the model reads carefully. So check answers, and ask for sources.",
    # 8. Code
    "In code: embed the library once. For each question, retrieve the top matches, "
    "join them into the prompt with the instructions, and generate.",
    # 9. Why it matters
    "RAG is how most assistants answer from private or fresh information, without retraining the model. "
    "Update a document, and the next answer uses it. But everything depends on retrieving the right pieces.",
    # 10. Outro
    "And that's harder than it looks. Next up: chunking and retrieval, why RAG fails, and how to fix it.",
]
