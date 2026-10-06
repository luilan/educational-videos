"""Deep Dive, episode 31: The KV Cache, Deeper.

GPT-2 small, greedy generation:
1. With and without the KV cache: the same tokens? How long does each take?
2. What the cache holds, and how big it is per token.
3. Time per generated token as the text grows, with and without the cache.
4. Prefill vs decode: reading a 512-token prompt in one pass vs one token at a time.
"""
import time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_num_threads(8)
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
prompt = tok("The history of the printing press begins", return_tensors="pt").input_ids
N = 200


@torch.no_grad()
def generate(use_cache):
    ids, past, step_times = prompt.clone(), None, []
    for _ in range(N):
        t0 = time.perf_counter()
        if use_cache:
            out = model(ids if past is None else ids[:, -1:], past_key_values=past, use_cache=True)
            past = out.past_key_values
        else:
            out = model(ids, use_cache=False)                     # recompute every token, every step
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
        step_times.append(time.perf_counter() - t0)
    return ids, step_times, past


generate(True)                                                     # warm-up
a, ta, past = generate(True)
b, tb, _ = generate(False)
print(f"1. {N} new tokens, greedy: identical output with and without the cache: {torch.equal(a, b)}")
print(f"   with the cache {sum(ta):.2f} s, without {sum(tb):.2f} s ({sum(tb) / sum(ta):.1f}x slower)")
print(f"   text: {tok.decode(a[0, prompt.shape[1]:prompt.shape[1] + 25])!r} ...")

# ---------------------------------------------------------------- 2. what is in the cache
k0 = past.layers[0].keys
n_tok = k0.shape[2]
total = sum(l.keys.numel() + l.values.numel() for l in past.layers) * 4
print(f"\n2. the cache: for each of {len(past.layers)} layers, keys and values of shape {tuple(k0.shape)} "
      f"(batch, heads, tokens, head size)")
print(f"   {total / n_tok:,.0f} bytes per token (float32) = 2 × 12 layers × 768 × 4; "
      f"{total / 2**20:.1f} MiB for these {n_tok} tokens; {total / n_tok * 1024 / 2**20:.0f} MiB at 1,024 tokens")

# ---------------------------------------------------------------- 3. time per step as the text grows
print("\n3. time for one new token (ms), at different text lengths")
for i in (0, 49, 99, 199):
    print(f"   token {i + 1:>3} (text length {prompt.shape[1] + i:>3}): with cache {ta[i] * 1000:6.1f}   "
          f"without {tb[i] * 1000:6.1f}")

# ---------------------------------------------------------------- 4. prefill vs decode
long = tok("The " * 600, return_tensors="pt").input_ids[:, :512]
with torch.no_grad():
    t0 = time.perf_counter()
    model(long, use_cache=True)
    prefill = time.perf_counter() - t0
    t0 = time.perf_counter()
    past = None
    for i in range(long.shape[1]):
        past = model(long[:, i:i + 1], past_key_values=past, use_cache=True).past_key_values
    one_by_one = time.perf_counter() - t0
print(f"\n4. reading a 512-token prompt: in one pass (prefill) {prefill:.2f} s, "
      f"one token at a time {one_by_one:.2f} s ({one_by_one / prefill:.0f}x slower)")
print(f"   prefill: {512 / prefill:,.0f} tokens per second; decoding: {1 / (sum(ta) / N):,.0f} tokens per second")
