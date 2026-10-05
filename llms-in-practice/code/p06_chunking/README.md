# Episode 6 · Chunking and Retrieval: Why RAG Fails, and Fixes

A measured comparison of four ways to cut documents for retrieval, on a small library of three made-up handbooks
(`bakery.txt`, `bikes.txt`, `gym.txt`) and ten questions about the bakery with known answers.

```bash
pip install -r requirements.txt
python chunking.py
```

Uses `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB, CPU). It prints the token length of each handbook (the
bakery's 327 tokens exceed the model's 256-token input, so the end of the whole-document vector is silently lost),
then a table:

| strategy | chunks | answer in top chunk | words sent |
|---|---|---|---|
| whole document | 3 | 10 / 10 | 255 |
| fixed 30 words | 17 | 8 / 10 | 30 |
| one per section | 17 | 10 / 10 | 38 |
| sentence + neighbours | 63 | 10 / 10 | 28 |

with every miss and the chunk that was ranked first instead. Try other chunk sizes or your own documents and questions.
