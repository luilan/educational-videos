# How LLMs Work: Deep Dive

A long-running follow-on to [How LLMs Work](../how-llms-work/) that opens up every part of a modern LLM:
how each piece really works, why it is built that way, and how today's models differ from the tiny GPT.
Prerequisite: How LLMs Work (and [Foundations](../foundations/) for the math). Status: **in production**.

Every episode ships three things, all public in this repository:

1. **The video source**: `dNN_script.py` (narration) and `dNN_scene.py` (Manim scene).
2. **The study guide**: `study/dNN_study.pdf`, same format as the other series (concepts in order, each with an
   explanation, a video frame, the key idea and check questions; answers at the end).
3. **The episode's code**: `code/dNN_<topic>/`, small runnable Python (NumPy / PyTorch) that does exactly what the
   episode shows, with its own README and pinned requirements. Where it fits, the code upgrades the series' tiny GPT
   one component at a time, so by the end viewers have a small modern LLM they built themselves.

Release: up to six episodes a day, after How LLMs Work finishes (from 2026-10-08).

## Plan (working titles, in arcs)

**1. Tokenization**
d01 Byte-Pair Encoding, step by step · d02 Bytes, Unicode and why "strawberry" is hard · d03 Tokenizer quirks that shape model behaviour

**2. Position**
d04 Why attention needs position · d05 RoPE: rotating vectors to encode order · d06 Long context: stretching RoPE

**3. Attention, inside out**
d07 Causal masking, in detail · d08 The attention matrix, entry by entry · d09 Multi-query and grouped-query attention · d10 FlashAttention: same math, less memory · d11 Sliding windows and sparse attention · d12 Attention sinks

**4. The rest of the block**
d13 The residual stream · d14 LayerNorm vs RMSNorm · d15 Activations: ReLU, GELU, SwiGLU · d16 Pre-norm, post-norm and stability

**5. Training**
d17 Cross-entropy, deeper · d18 Backprop through a transformer · d19 Adam and AdamW · d20 Learning-rate warmup and schedules · d21 Batch size and gradient accumulation · d22 Mixed precision · d23 Scaling laws · d24 Where training data comes from

**6. Training at scale**
d25 Data parallelism · d26 Tensor and pipeline parallelism · d27 Sharding optimizer state

**7. Architecture variants**
d28 Mixture of experts · d29 State-space models · d30 Multimodal: images as tokens

**8. Inference**
d31 The KV cache, deeper · d32 Batching and paged attention · d33 Quantization · d34 Speculative decoding · d35 Decoding strategies compared

**9. Post-training**
d36 Supervised fine-tuning · d37 Reward models · d38 RLHF with PPO · d39 DPO · d40 Reinforcement learning for reasoning

**10. Interpretability**
d41 The logit lens · d42 Induction heads · d43 Superposition · d44 Sparse autoencoders · d45 Circuits
