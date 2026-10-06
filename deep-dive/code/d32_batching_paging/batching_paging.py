"""Deep Dive, episode 32: Batching and Paged Attention.

1. Batching: GPT-2 generating 64 tokens for 1, 4, 16 requests at once. Throughput in tokens per second.
2. Uneven requests: how much work is wasted on padding when short and long requests share a batch?
3. KV-cache memory: reserving the maximum length per request (contiguous) vs pages of 16 tokens (paged attention).
"""
import random
import time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_num_threads(8)
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
NEW = 64


@torch.no_grad()
def generate(batch_size):
    ids = tok(["The history of the printing press begins"] * batch_size, return_tensors="pt").input_ids
    past = None
    t0 = time.perf_counter()
    for _ in range(NEW):
        out = model(ids if past is None else ids[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        ids = torch.cat([ids, out.logits[:, -1].argmax(-1, keepdim=True)], 1)
    return time.perf_counter() - t0


generate(1)                                                    # warm-up
print(f"1. GPT-2 generating {NEW} tokens per request (this CPU)")
base = None
for bs in (1, 4, 16):
    sec = generate(bs)
    tps = bs * NEW / sec
    base = base or tps
    print(f"   {bs:>2} requests at once: {sec:5.2f} s, {tps:6.0f} tokens per second ({tps / base:.1f}x)")

# ---------------------------------------------------------------- 2. padding
random.seed(0)
lengths = [random.choice([20, 50, 100, 400, 1000]) for _ in range(16)]
padded = len(lengths) * max(lengths)
print(f"\n2. 16 requests wanting {sorted(lengths)} tokens")
print(f"   one fixed batch runs until the longest finishes: {padded:,} token slots, {sum(lengths):,} useful "
      f"({1 - sum(lengths) / padded:.0%} wasted)")
print("   continuous batching: a finished request leaves, a waiting one joins at the next step")

# ---------------------------------------------------------------- 3. paged KV cache
PAGE, MAX_LEN, PER_TOKEN = 16, 2048, 2 * 12 * 768 * 2          # bytes per token for GPT-2 small in float16
random.seed(1)
reqs = [random.randint(50, 1500) for _ in range(100)]
contiguous = len(reqs) * MAX_LEN * PER_TOKEN
paged = sum(-(-n // PAGE) * PAGE for n in reqs) * PER_TOKEN
used = sum(reqs) * PER_TOKEN
print(f"\n3. 100 requests of 50 to 1,500 tokens (average {sum(reqs) / len(reqs):.0f}), GPT-2 small, float16 cache")
print(f"   reserve {MAX_LEN:,} tokens each (contiguous): {contiguous / 2**30:.2f} GiB, {used / contiguous:.0%} actually used")
print(f"   pages of {PAGE} tokens, allocated as needed:    {paged / 2**30:.2f} GiB, {used / paged:.1%} actually used")
print(f"   the same memory holds {contiguous / paged:.1f}x more requests with paging")
