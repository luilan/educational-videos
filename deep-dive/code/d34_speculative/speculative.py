"""Deep Dive, episode 34: Speculative Decoding.

A small draft model (Qwen2.5-0.5B) guesses k tokens; the big target model (Qwen2.5-1.5B) checks them all in one pass.
Greedy decoding, CPU, float32:
1. Why it can work: the target's time for 1 token vs k+1 tokens in one pass.
2. Speculative vs plain greedy decoding: same output? acceptance rate, tokens per target pass, speed, for k = 2..8.
3. Sampling: the accept/reject rule keeps exactly the target's distribution (a simulation on 4 tokens).
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

torch.manual_seed(0)
torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).eval()
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.float32).eval()
N = 128
PROMPTS = {
    "prose": "The water cycle describes how water moves through the environment. It begins when",
    "code": "def merge_sort(items):\n    \"\"\"Sort a list with merge sort.\"\"\"\n",
}


@torch.no_grad()
def plain(ids):
    cache, out = DynamicCache(), ids
    t0 = time.perf_counter()
    for _ in range(N):
        logits = target(out if cache.get_seq_length() == 0 else ids[:, -1:], past_key_values=cache, use_cache=True).logits
        ids = torch.cat([ids, logits[:, -1:].argmax(-1)], 1)
    return ids, time.perf_counter() - t0, N


@torch.no_grad()
def speculative(ids, k):
    """Caches hold every token except the newest; each round feeds whatever is missing."""
    tc, dc, start = DynamicCache(), DynamicCache(), ids.shape[1]
    passes = accepted = 0
    t0 = time.perf_counter()
    while ids.shape[1] - start < N:
        d_in, drafts = ids[:, dc.get_seq_length():], []
        for _ in range(k):                                         # the draft guesses k tokens, one at a time
            nxt = draft(d_in, past_key_values=dc, use_cache=True).logits[:, -1:].argmax(-1)
            drafts.append(nxt)
            d_in = nxt
        drafts = torch.cat(drafts, 1)
        t_in = torch.cat([ids[:, tc.get_seq_length():], drafts], 1)
        preds = target(t_in, past_key_values=tc, use_cache=True).logits[:, -(k + 1):].argmax(-1)  # one pass checks all
        passes += 1
        n = 0
        while n < k and drafts[0, n] == preds[0, n]:
            n += 1
        accepted += n
        old = ids.shape[1]
        ids = torch.cat([ids, drafts[:, :n], preds[:, n:n + 1]], 1)  # accepted guesses + the target's own next token
        tc.crop(old + n)
        dc.crop(min(dc.get_seq_length(), old + n))
    return ids[:, :start + N], time.perf_counter() - t0, passes, accepted


# ---------------------------------------------------------------- 1. one token vs several in one pass
print("1. time for one target pass, after a 100-token context (median of 7)")
ctx = tok("The " * 100, return_tensors="pt").input_ids[:, :100]
for name, model in (("draft 0.5B", draft), ("target 1.5B", target)):
    row = []
    for m in (1, 5, 9):
        ts = []
        for _ in range(7):
            c = DynamicCache()
            with torch.no_grad():
                model(ctx, past_key_values=c, use_cache=True)
                t0 = time.perf_counter()
                model(ctx[:, :m], past_key_values=c, use_cache=True)
            ts.append(time.perf_counter() - t0)
        row.append(sorted(ts)[3] * 1000)
    print(f"   {name}: 1 token {row[0]:.0f} ms, 5 tokens {row[1]:.0f} ms, 9 tokens {row[2]:.0f} ms")

# ---------------------------------------------------------------- 2. speculative decoding
print(f"\n2. {N} new tokens, greedy")
plain(tok("warm up", return_tensors="pt").input_ids)
for name, p in PROMPTS.items():
    ids = tok(p, return_tensors="pt").input_ids
    ref, t_ref, _ = plain(ids)
    print(f"   [{name}] plain target: {t_ref:.1f} s, {N / t_ref:.1f} tokens/s, {N} target passes")
    for k in (2, 4, 6, 8):
        out, t, passes, acc = speculative(ids, k)
        same = torch.equal(out, ref)
        print(f"   [{name}] k={k}: {t:.1f} s ({t_ref / t:.2f}x), {acc / (passes * k) * 100:.0f}% of guesses accepted, "
              f"{passes} target passes ({N / passes:.1f} tokens each), identical output: {same}")
    print(f"   [{name}] text: {tok.decode(ref[0, ids.shape[1]:ids.shape[1] + 40])!r}")

# ---------------------------------------------------------------- 3. sampling: accept / reject
print("\n3. sampling with a draft: accept the draft's token x with probability min(1, p(x)/q(x)); "
      "on rejection, sample from max(0, p - q), renormalised")
p = torch.tensor([0.50, 0.30, 0.15, 0.05])                         # target
q = torch.tensor([0.25, 0.50, 0.20, 0.05])                         # draft
M = 1_000_000
x = torch.multinomial(q, M, replacement=True)
keep = torch.rand(M) < (p[x] / q[x]).clamp(max=1)
resid = (p - q).clamp(min=0)
y = torch.where(keep, x, torch.multinomial(resid / resid.sum(), M, replacement=True))
emp = torch.bincount(y, minlength=4).float() / M
print(f"   target p      {[round(v, 3) for v in p.tolist()]}")
print(f"   draft q       {[round(v, 3) for v in q.tolist()]}")
print(f"   result ({M:,} samples) {[round(v, 3) for v in emp.tolist()]}")
print(f"   accepted {keep.float().mean() * 100:.1f}% (theory: sum of min(p, q) = {torch.minimum(p, q).sum() * 100:.0f}%)")
