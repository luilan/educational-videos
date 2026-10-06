"""Deep Dive, episode 39: DPO (Direct Preference Optimization).

The same goal as episode 38 (GPT-2 small writing positive continuations), without a reward model in the loop and without
sampling during training:
1. Build preference pairs once: sample continuations from GPT-2 and pair every sample that has a positive score
   (positive minus negative words) with a random lower-scoring sample of the same prompt. A stand-in for human raters.
2. Train with the DPO loss:  -log sigmoid(beta * [(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))])
3. Measure what PPO measured: positivity of fresh samples, KL from the original GPT-2, and the original model's loss on the
   text.
"""
import random
import re
import time

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(8)
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
PROMPTS = ["The movie was", "I think this restaurant", "My new phone is", "The weather today", "This book",
           "Our trip to the city", "The concert last night", "My neighbour's dog"]
POS = {"good", "great", "love", "loved", "wonderful", "amazing", "excellent", "best", "happy", "beautiful", "perfect",
       "fantastic", "nice", "enjoy", "enjoyed", "awesome", "brilliant", "delightful", "fun", "favorite"}
NEG = {"bad", "terrible", "awful", "worst", "hate", "boring", "poor", "horrible", "disappointing", "sad", "waste",
       "ugly", "annoying", "worse"}
N_NEW, SAMPLES, BETA, EPOCHS, B = 24, 192, 0.1, 3, 16


def reward(text):
    words = re.findall(r"[a-z']+", text.lower())
    return sum(w in POS for w in words) - sum(w in NEG for w in words)


def sample(lm, p, n, seed):
    torch.manual_seed(seed)
    prompt = tok(p, return_tensors="pt").input_ids.repeat(n, 1)
    with torch.no_grad():
        ids = lm.generate(prompt, do_sample=True, top_k=0, max_new_tokens=N_NEW, min_new_tokens=N_NEW,
                          suppress_tokens=[tok.eos_token_id], pad_token_id=tok.eos_token_id,
                          attention_mask=torch.ones_like(prompt))
    return ids, prompt.shape[1]


def seq_logprob(lm, ids, start):
    """Sum of log p over the continuation tokens, per sequence."""
    lp = F.log_softmax(lm(ids).logits[:, start - 1:-1], -1)
    return lp.gather(-1, ids[:, start:, None]).squeeze(-1).sum(1)


policy = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
ref = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()

# ---------------------------------------------------------------- 1. preference pairs
pairs = []                                                             # (prompt index, chosen ids, rejected ids)
for i, p in enumerate(PROMPTS):
    ids, P = sample(ref, p, SAMPLES, seed=i)
    rs = [reward(t) for t in tok.batch_decode(ids[:, P:])]
    for c in range(SAMPLES):
        lower = [r for r in range(SAMPLES) if rs[r] < rs[c]]
        if rs[c] > 0 and lower:
            r = random.choice(lower)
            pairs.append((i, P, ids[c], ids[r], rs[c], rs[r]))
print(f"1. {len(PROMPTS) * SAMPLES} samples from GPT-2 → {len(pairs)} preference pairs")
c0 = pairs[0]
print(f"   e.g. chosen   ({c0[4]:+d}): {tok.decode(c0[2])!r}\n        rejected ({c0[5]:+d}): {tok.decode(c0[3])!r}")

with torch.no_grad():                                                  # the reference never changes: score it once
    ref_c = torch.cat([seq_logprob(ref, c[None], P) for _, P, c, _, _, _ in pairs])
    ref_r = torch.cat([seq_logprob(ref, r[None], P) for _, P, _, r, _, _ in pairs])

# ---------------------------------------------------------------- 2. DPO training
print(f"\n2. DPO: beta {BETA}, {EPOCHS} epochs, batches of {B} pairs")
opt = torch.optim.Adam(policy.parameters(), lr=1e-5)
t0, step = time.time(), 0
for ep in range(EPOCHS):
    by_prompt = {}
    for j, pr in enumerate(pairs):
        by_prompt.setdefault(pr[0], []).append(j)
    batches = []
    for js in by_prompt.values():                                       # same prompt → same length → one tensor
        random.shuffle(js)
        batches += [js[k:k + B] for k in range(0, len(js), B)]
    random.shuffle(batches)
    for batch in batches:
        P = pairs[batch[0]][1]
        C = torch.stack([pairs[j][2] for j in batch])
        R = torch.stack([pairs[j][3] for j in batch])
        margin = BETA * ((seq_logprob(policy, C, P) - ref_c[batch]) - (seq_logprob(policy, R, P) - ref_r[batch]))
        loss = -F.logsigmoid(margin).mean()                            # implicit reward of chosen minus rejected
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"   step {step + 1:>3} (epoch {ep + 1}): loss {loss.item():.3f}, implicit reward prefers chosen in "
                  f"{(margin > 0).float().mean() * 100:.0f}% of the batch ({time.time() - t0:.0f} s)", flush=True)
        step += 1
print(f"   {step} steps")


# ---------------------------------------------------------------- 3. evaluation, as in episode 38
@torch.no_grad()
def evaluate(lm, label):
    rs, rl, kls, samples = [], [], [], []
    for i, p in enumerate(PROMPTS):
        ids, P = sample(lm, p, 8, seed=1000 + i)
        texts = tok.batch_decode(ids[:, P:])
        rs += [reward(t) for t in texts]
        lp_ref = F.log_softmax(ref(ids).logits[:, P - 1:-1], -1).gather(-1, ids[:, P:, None]).squeeze(-1)
        lp_pol = F.log_softmax(lm(ids).logits[:, P - 1:-1], -1).gather(-1, ids[:, P:, None]).squeeze(-1)
        rl.append(-lp_ref.mean().item())
        kls.append((lp_pol - lp_ref).sum(1).mean().item())
        samples.append(p + texts[0])
    print(f"   {label}: reward {sum(rs) / len(rs):.2f}, KL {sum(kls) / len(kls):.2f}, reference loss "
          f"{sum(rl) / len(rl):.2f} (64 samples)")
    for s in samples[:3]:
        print(f"      {s!r}")


print("\n3. fresh samples")
evaluate(ref, "GPT-2")
evaluate(policy, "after DPO")
