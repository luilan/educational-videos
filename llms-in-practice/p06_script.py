"""LLMs in Practice, episode 6 — Chunking and Retrieval: Why RAG Fails, and Fixes. One entry per narration section."""
TITLE = "Chunking and Retrieval"
TAGLINE = "Why RAG fails, and how to fix it"
NEXT = "Tool Use: How a Model Calls a Function"
SECTIONS = [
    # 1. Hook
    "Last time, RAG worked. But real documents are long: handbooks, manuals, contracts. "
    "You can't paste a whole library into the prompt, so you retrieve pieces. "
    "And how you cut those pieces decides whether RAG works at all.",
    # 2. The experiment
    "Here's a test. A small library of three made-up handbooks: a bakery, a bike shop, and a gym. "
    "Ten questions about the bakery, each with a known answer. For each way of cutting the text, we check two things: "
    "is the answer in the top chunk, and how many words go into the prompt.",
    # 3. Whole documents
    "First, no cutting: one vector per handbook. It found the right handbook every time. "
    "But it sends all two hundred and fifty-five words for every question, about nine times more than needed. "
    "And there's a hidden problem. Our embedding model reads at most two hundred and fifty-six tokens. "
    "The bakery handbook is three hundred and twenty-seven. The student discount, the discount code, the job ad: "
    "they never make it into the vector.",
    # 4. Fixed-size chunks
    "Next, cut every thirty words, wherever the cut falls. Small chunks, but only eight out of ten. Look at the cuts. "
    "One chunk ends: students get a ten. And the word percent is in the next chunk. "
    "Another ends: we are hiring a morning baker. The. And the start time, four o'clock, is in the next one. "
    "The facts were cut in half.",
    # 5. Natural boundaries
    "Now cut along the text's own structure: one chunk per section, with its heading. Ten out of ten, "
    "with about thirty-eight words each. Or each sentence with its neighbors, so the chunks overlap. "
    "Also ten out of ten, with twenty-eight words.",
    # 6. Lessons
    "So: cut at natural boundaries. Keep the headings, they carry the topic. Overlap a little, "
    "so no fact falls between two chunks. And size the chunks to hold one answer, small enough to stay precise.",
    # 7. Other failures
    "Chunking isn't the only way retrieval fails. The question may use different words than the document: "
    "rewrite the question, or add keyword search. The answer may need two chunks: retrieve more of them. "
    "And retrieve too much, and the key fact gets lost in the middle, like in episode two.",
    # 8. Code
    "In code, the good chunkers are short. Split on blank lines for sections. "
    "Or split into sentences, and join each one with its neighbors.",
    # 9. Measure
    "Most of all: measure. Write a few questions with known answers, like we did, "
    "and check what retrieval returns. Change one thing at a time, and keep what works.",
    # 10. Outro
    "So far, the model only reads. Next up: tool use, and how a model can do things.",
]
