# Episode 5 · RAG: Giving the Model a Library

Retrieval-augmented generation in about 40 lines, about a made-up bakery the models have never seen.

```bash
pip install -r requirements.txt
python rag.py
```

Downloads `all-MiniLM-L6-v2` (retrieval, 90 MB) and two readers, `Qwen2.5-1.5B-Instruct` (about 3 GB) and
`Qwen2.5-0.5B-Instruct` (about 1 GB). Runs on a CPU; the 1.5B model needs a minute or two and about 8 GB of RAM.

It prints:

1. the 1.5B model's answer **without RAG** (it cannot know about Bella's Bakery);
2. the two **retrieved** facts with their similarity scores (0.70: gluten-free bread only on Fridays; 0.62: opening hours);
3. the answer **with RAG** from both readers: the 1.5B model answers correctly; the 0.5B model, given the same prompt,
   answers wrongly. Retrieval only helps if the model reads carefully.

Greedy decoding is used, so the answers are repeatable on the same library versions.
