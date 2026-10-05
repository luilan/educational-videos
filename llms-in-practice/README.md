# LLMs in Practice

A follow-on to [How LLMs Work](../how-llms-work/): how real products are built on top of a language model.
Prerequisite: How LLMs Work (especially episodes 3, 10 and 14). Status: **in production** (first; the [Deep Dive](../deep-dive/) follows).

Every episode ships three things, all public in this repository:

1. **The video source**: `pNN_script.py` (narration) and `pNN_scene.py` (Manim scene).
2. **The study guide**: `study/pNN_study.pdf`, in the same format as the other series (concepts in order,
   each with an explanation, a video frame, the key idea and check questions; answers at the end).
3. **The episode's code**: `code/pNN_<topic>/`, small runnable Python that does exactly what the episode shows,
   with its own README and pinned requirements.

| # | Title (working) | Code |
|---|---|---|
| 01 | Prompts Are Just Context: what the model really sees | chat template → raw token stream |
| 02 | Context Windows, and Why They Run Out | token counting and truncation |
| 03 | Sampling: Temperature, Top-p, and Why Answers Vary | sampling from real logits |
| 04 | Embeddings for Search: Meaning as Distance | cosine-similarity search |
| 05 | RAG: Giving the Model a Library | minimal retrieval-augmented generation |
| 06 | Chunking and Retrieval: Why RAG Fails, and Fixes | chunking strategies compared |
| 07 | Tool Use: How a Model Calls a Function | function-calling loop |
| 08 | Agents: Think, Act, Observe | a tiny agent loop |
| 09 | Fine-Tuning vs Prompting, and LoRA in One Picture | LoRA on the tiny GPT |
| 10 | Quantization: Shrinking a Model to Fit a Laptop | int8 quantization of the tiny GPT |
| 11 | Evaluating LLMs: How We Know It's Better | a small eval harness |
| 12 | Hallucinations and Safety: Where Things Go Wrong | grounding check |
