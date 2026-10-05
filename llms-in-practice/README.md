# LLMs in Practice

A follow-on to [How LLMs Work](../how-llms-work/): how real products are built on top of a language model.
Prerequisite: How LLMs Work (especially episodes 3, 10 and 14). Status: **complete** (12 episodes, about 30 minutes). The [Deep Dive](../deep-dive/) follows.

Every episode ships three things, all public in this repository:

1. **The video source**: `pNN_script.py` (narration) and `pNN_scene.py` (Manim scene).
2. **The study guide**: `study/pNN_study.pdf`, in the same format as the other series (concepts in order,
   each with an explanation, a video frame, the key idea and check questions; answers at the end).
3. **The episode's code**: `code/pNN_<topic>/`, small runnable Python that does exactly what the episode shows,
   with its own README and pinned requirements.

| # | Title | Length | Code |
|---|---|---|---|
| 01 | **Prompts Are Just Context**<br>What the model really sees when you chat | 2:06 | `p01_prompt_is_context`: the real chat template, tokens per turn |
| 02 | **Context Windows**<br>Why a model can only read so much at once | 2:21 | `p02_context_window`: window size, KV-cache memory, trimming a chat |
| 03 | **Sampling**<br>Temperature, top-p, and why answers vary | 2:24 | `p03_sampling`: real next-token probabilities, temperature, top-p |
| 04 | **Embeddings for Search**<br>Meaning as distance | 2:14 | `p04_embedding_search`: search by meaning with a sentence model |
| 05 | **RAG: Giving the Model a Library**<br>Search first, then answer | 2:26 | `p05_rag`: retrieval + generation, two model sizes compared |
| 06 | **Chunking and Retrieval**<br>Why RAG fails, and how to fix it | 2:39 | `p06_chunking`: four chunking strategies measured |
| 07 | **Tool Use**<br>How a model calls a function | 2:17 | `p07_tool_use`: a real tool call and the tool loop |
| 08 | **Agents**<br>Think, act, observe, repeat | 2:46 | `p08_agent`: a multi-step agent, failures and recovery |
| 09 | **Fine-Tuning and LoRA**<br>When prompting isn't enough | 2:37 | `p09_lora`: full fine-tuning vs hand-written LoRA on the tiny GPT |
| 10 | **Quantization**<br>Shrinking a model to fit a laptop | 2:30 | `p10_quantization`: 8, 6, 5, 4, 3 and 2 bits measured |
| 11 | **Evaluating LLMs**<br>How we know it's better | 2:37 | `p11_evaluation`: a small evaluation harness, three models |
| 12 | **Hallucinations and Safety**<br>Where things go wrong, and what to do | 2:56 | `p12_safety`: hallucinations, a grounding check, prompt injection |

All numbers on screen come from running the episode's code (Qwen2.5 Instruct models 0.5B–3B, all-MiniLM-L6-v2 and
the tiny GPT, on a CPU). Bella's Bakery and the other businesses are made up.
