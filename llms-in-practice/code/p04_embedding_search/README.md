# Episode 4 · Embeddings for Search: Meaning as Distance

Search by meaning: embed short documents with `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB, CPU) and rank
them against a question with cosine similarity.

```bash
pip install -r requirements.txt
python search.py
```

It prints:

1. each document as a **384-number vector** (token vectors averaged, then scaled to length 1);
2. the top matches for three questions, e.g. *"Where does my kitten like to sleep?"* → the two cat sentences
   (0.65, 0.60) although they share **no words** with the question (keyword overlap 0);
3. a **2-D map** of the vectors (exact PCA), as drawn in the episode, with the questions projected onto it.

Try it: add your own documents and questions, or the exercise in the study guide (a boiled egg from five days ago,
a puppy, and "I love my cat" vs "I love my car").
