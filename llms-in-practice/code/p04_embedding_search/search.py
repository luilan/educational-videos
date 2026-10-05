"""LLMs in Practice, episode 4: Embeddings for Search — meaning as distance.

Embed a few short documents with a small sentence-embedding model, then find the ones closest in meaning
to a question, even when they share no words with it.

    pip install -r requirements.txt
    python search.py
Downloads sentence-transformers/all-MiniLM-L6-v2 (about 90 MB, runs on a CPU).
"""
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DOCS = [
    "Boil eggs for 10 minutes for firm yolks.",
    "Simmer a poached egg for about 3 minutes.",
    "Cats love sleeping in cardboard boxes.",
    "Dogs need a walk at least twice a day.",
    "Preheat the oven to 200 degrees before baking bread.",
    "The train to Milan leaves at 8 in the morning.",
    "Store cooked eggs in the fridge for up to a week.",
    "Our cat naps on the sofa all afternoon.",
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).eval()


def embed(texts):
    """One vector per text: average the token vectors, then scale to length 1."""
    batch = tok(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state              # (texts, tokens, 384)
    mask = batch.attention_mask.unsqueeze(-1)
    vectors = (hidden * mask).sum(1) / mask.sum(1)            # mean over real tokens
    return torch.nn.functional.normalize(vectors, dim=-1)


docs = embed(DOCS)
print(f"{len(DOCS)} documents -> vectors of {docs.shape[1]} numbers")


def search(question, k=3):
    q = embed([question])[0]
    scores = docs @ q                                          # cosine similarity (vectors have length 1)
    best = torch.topk(scores, k)
    print(f"\n{question!r}")
    for s, i in zip(best.values, best.indices):
        print(f"  {s:.2f}  {DOCS[i]}")


search("How long should I cook a hard-boiled egg?")
search("Where does my kitten like to sleep?")          # shares no keyword with the cat documents
search("When is the first departure to Milan?")

# Keyword search for comparison: count shared words.
q = set("where does my kitten like to sleep".split())
print("\nkeyword overlap for 'Where does my kitten like to sleep?':",
      {d: len(q & set(d.lower().strip('.').split())) for d in DOCS[2:3] + DOCS[7:8]})

# 2-D picture of the vectors (PCA), as drawn in the episode.
mean = docs.mean(0)
_, _, vt = torch.linalg.svd(docs - mean, full_matrices=False)    # exact PCA: top 2 directions
axes = vt[:2].T
axes = axes * torch.sign(axes.sum(0))                             # fix the arbitrary sign of each direction
print("\n2-D coordinates (PCA):")
for (x, y), d in zip(((docs - mean) @ axes).tolist(), DOCS):
    print(f"  ({x:+.2f}, {y:+.2f})  {d}")
for question in ["Where does my kitten like to sleep?", "How long should I cook a hard-boiled egg?"]:
    x, y = ((embed([question])[0] - mean) @ axes).tolist()
    print(f"  ({x:+.2f}, {y:+.2f})  question: {question}")
