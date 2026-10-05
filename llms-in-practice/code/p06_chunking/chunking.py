"""LLMs in Practice, episode 6: Chunking and Retrieval — why RAG fails, and fixes.

A small library of three made-up handbooks (a bakery, a bike shop, a gym). Cut it into chunks in four ways,
embed every chunk (episode 4), and measure two things for ten questions about the bakery: how often the top
chunk really contains the answer, and how many words that chunk puts into the prompt.

    pip install -r requirements.txt
    python chunking.py
Downloads sentence-transformers/all-MiniLM-L6-v2 (about 90 MB, CPU).
"""
import re
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).parent
DOCS = {name: (HERE / f"{name}.txt").read_text() for name in ("bakery", "bikes", "gym")}
# Each question, and a phrase that appears only in the passage that answers it.
QUESTIONS = [
    ("When can I get gluten-free bread?", "only available on Fridays"),
    ("How much is a cake for 12 people?", "costs 39 euros"),
    ("Is bakery delivery free on Sundays?", "free on Sundays"),
    ("Do your pastries contain nuts?", "traces of nuts"),
    ("Do students get a discount at the bakery?", "Students get a 10 percent"),
    ("What time does the morning baker start?", "starts at 4:00"),
    ("How early must I order a birthday cake?", "three days in advance"),
    ("Is the bakery open on public holidays?", "On public holidays we open at 9:00"),
    ("Is BELLA10 still valid?", "BELLA10"),
    ("When is rye bread baked?", "Tuesdays and Thursdays"),
]

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()


def embed(texts):
    batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    mask = batch.attention_mask.unsqueeze(-1)
    return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)


# ---- four ways to cut the same library (each chunker works per document)
def sections_of(text):
    return [s.strip() for s in text.split("\n\n")]


def sentences_of(text):
    return [s for sec in sections_of(text) for s in re.split(r"(?<=[.!?])\s+|\n", sec) if s]


def whole(text):                  # 1 chunk per document
    return [text]


def fixed(text, n=30):            # every 30 words, cut wherever that falls
    words = text.split()
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)]


def by_section(text):             # one chunk per section, with its heading
    return sections_of(text)


def sentence_window(text):        # each sentence with its neighbours (overlapping chunks)
    s = sentences_of(text)
    return [" ".join(s[max(0, i - 1):i + 2]) for i in range(len(s))]


def chunk_library(chunker):
    return [c for text in DOCS.values() for c in chunker(text)]


def evaluate(chunks, show_misses=False):
    vectors = embed(chunks)
    hits, words_sent = 0, 0
    for question, answer in QUESTIONS:
        best = chunks[int((vectors @ embed([question])[0]).argmax())]
        hits += answer in best
        words_sent += len(best.split())
        if show_misses and answer not in best:
            print(f"    miss: {question!r} -> {' '.join(best.split()[:12])} ...")
    return hits, words_sent / len(QUESTIONS)


print(f"library: {len(DOCS)} documents, {sum(len(t.split()) for t in DOCS.values())} words")
# The embedding model reads at most 256 tokens: anything after that is silently dropped from the vector.
for name, text in DOCS.items():
    ids = tok(text)["input_ids"]
    lost = f", the vector ignores everything after {tok.decode(ids[250:256])!r}" if len(ids) > 256 else ""
    print(f"  {name}: {len(ids)} tokens{lost}")
print()
print(f"{'strategy':<24}{'chunks':>7}{'answer in top chunk':>21}{'words sent':>12}")
for name, chunker in [("whole document", whole), ("fixed 30 words", fixed), ("one per section", by_section),
                      ("sentence + neighbours", sentence_window)]:
    chunks = chunk_library(chunker)
    hits, sent = evaluate(chunks)
    print(f"{name:<24}{len(chunks):>7}{hits:>15} / {len(QUESTIONS)}{sent:>12.0f}")
    evaluate(chunks, show_misses=True)
