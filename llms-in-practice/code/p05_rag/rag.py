"""LLMs in Practice, episode 5: RAG — giving the model a library.

Retrieval-augmented generation in about 40 lines: find the documents that match a question (episode 4's
embeddings), paste them into the prompt, and let the model answer from them. Bella's Bakery is made up,
so the model cannot know these facts from training.

    pip install -r requirements.txt
    python rag.py
Downloads all-MiniLM-L6-v2 (90 MB) and two readers, Qwen2.5-1.5B-Instruct (about 3 GB) and
Qwen2.5-0.5B-Instruct (about 1 GB); runs on a CPU (the 1.5B model takes a minute or two).
"""
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

LIBRARY = [
    "Bella's Bakery opens at 7:30 on weekdays and at 9:00 on weekends.",
    "Bella's Bakery is closed on Mondays.",
    "Our sourdough loaf costs 6 euros and is baked every morning.",
    "Gluten-free bread is only available on Fridays.",
    "We deliver within 5 km for orders over 30 euros.",
    "Our croissants are made with French butter.",
]
QUESTION = "Can I buy gluten-free bread at Bella's Bakery on Wednesday?"

# 1. Retrieval: the embedding search from episode 4.
emb_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
emb = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()


def embed(texts):
    batch = emb_tok(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        hidden = emb(**batch).last_hidden_state
    mask = batch.attention_mask.unsqueeze(-1)
    return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)


library = embed(LIBRARY)                      # done once, stored


def retrieve(question, k=2):
    scores = library @ embed([question])[0]
    best = torch.topk(scores, k)
    return [(float(s), LIBRARY[i]) for s, i in zip(best.values, best.indices)]


# 2. Generation: chat models from episodes 1–3, greedy so the output is repeatable.
READERS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]


def ask(model_name, prompt):
    tok = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt")
    out = llm.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=60, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), ids.shape[1]


# 3. Without RAG: the model has never seen Bella's Bakery.
answer, n = ask(READERS[0], QUESTION)
print(f"WITHOUT RAG, 1.5B ({n} prompt tokens):\n  {answer}\n")

# 4. With RAG: retrieve, then put the documents in the prompt with clear instructions.
hits = retrieve(QUESTION)
print("retrieved:")
for s, doc in hits:
    print(f"  {s:.2f}  {doc}")
context = "\n".join(f"- {doc}" for _, doc in hits)
prompt = (f"Answer the question using only the information below. "
          f"If the answer is not there, say you don't know.\n\nInformation:\n{context}\n\nQuestion: {QUESTION}")
for name in READERS:
    answer, n = ask(name, prompt)
    print(f"\nWITH RAG, {name.split('-')[1]} ({n} prompt tokens):\n  {answer}")
