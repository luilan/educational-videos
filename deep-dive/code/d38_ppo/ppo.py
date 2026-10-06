"""Deep Dive, episode 38: RLHF with PPO.

GPT-2 small (the policy) learns to write positive continuations. The reward is a simple stand-in for a reward model:
positive words minus negative words in the continuation. PPO, written out in full: sample, score, compute advantages
with a value head, and take clipped policy-gradient steps. Two runs:
- beta = 0: reward only. The policy is free to drift from the original model (reward hacking).
- beta = 0.05 and 0.5: each token also pays beta × (log p_policy - log p_reference), a KL penalty that keeps it close
  to the original GPT-2.
Reported: task reward, KL from the reference, and the reference model's loss on the text (lower = more like normal
English to the original GPT-2).
"""
import copy
import re
import sys
import time

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_num_threads(8)
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
PROMPTS = ["The movie was", "I think this restaurant", "My new phone is", "The weather today", "This book",
           "Our trip to the city", "The concert last night", "My neighbour's dog"]
POS = {"good", "great", "love", "loved", "wonderful", "amazing", "excellent", "best", "happy", "beautiful", "perfect",
       "fantastic", "nice", "enjoy", "enjoyed", "awesome", "brilliant", "delightful", "fun", "favorite"}
NEG = {"bad", "terrible", "awful", "worst", "hate", "boring", "poor", "horrible", "disappointing", "sad", "waste",
       "ugly", "annoying", "worse"}
N_NEW, BATCH, ITERS, EPOCHS, CLIP = 24, 16, 200, 4, 0.2
BETAS = [float(b) for b in sys.argv[1:]] or [0.0, 0.05, 0.5]          # e.g. python ppo.py 0.5 to run one setting


def reward(text):
    words = re.findall(r"[a-z']+", text.lower())
    return sum(w in POS for w in words) - sum(w in NEG for w in words)


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        self.value = torch.nn.Linear(768, 1)

    def forward(self, ids):
        out = self.lm(ids, output_hidden_states=True)
        return out.logits, self.value(out.hidden_states[-1]).squeeze(-1)


def token_logprobs(logits, ids, start):
    """log p of each generated token (positions start..end), from logits at the position before it."""
    lp = F.log_softmax(logits[:, start - 1:-1], -1)
    return lp.gather(-1, ids[:, start:, None]).squeeze(-1)


def run(beta, seed=0):
    torch.manual_seed(seed)
    policy = Policy().eval()                                           # eval: no dropout in rollouts or updates
    ref = copy.deepcopy(policy.lm).eval()
    opt = torch.optim.Adam(policy.parameters(), lr=2e-5)
    history = []
    t0 = time.time()
    for it in range(ITERS):
        prompt = tok(PROMPTS[it % len(PROMPTS)], return_tensors="pt").input_ids.repeat(BATCH, 1)
        P = prompt.shape[1]
        with torch.no_grad():                                          # 1. sample continuations
            ids = policy.lm.generate(prompt, do_sample=True, top_k=0, max_new_tokens=N_NEW, min_new_tokens=N_NEW,
                                     suppress_tokens=[tok.eos_token_id], pad_token_id=tok.eos_token_id,
                                     attention_mask=torch.ones_like(prompt))
            texts = tok.batch_decode(ids[:, P:])
            task = torch.tensor([float(reward(t)) for t in texts])     # 2. score them
            logits, values = policy(ids)
            old_lp = token_logprobs(logits, ids, P)
            ref_lp = token_logprobs(ref(ids).logits, ids, P)
            values = values[:, P - 1:-1]
            kl = old_lp - ref_lp                                       # per-token KL estimate
            rewards = -beta * kl
            rewards[:, -1] += task
            adv, last = torch.zeros_like(rewards), torch.zeros(BATCH)  # 3. advantages (GAE, gamma 1, lambda 0.95)
            for t in reversed(range(N_NEW)):
                nxt = values[:, t + 1] if t + 1 < N_NEW else torch.zeros(BATCH)
                delta = rewards[:, t] + nxt - values[:, t]
                last = delta + 0.95 * last
                adv[:, t] = last
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(EPOCHS):                                        # 4. clipped policy-gradient steps
            logits, v = policy(ids)
            lp = token_logprobs(logits, ids, P)
            ratio = (lp - old_lp).exp()
            pg = -torch.min(ratio * adv, ratio.clamp(1 - CLIP, 1 + CLIP) * adv).mean()
            vloss = ((v[:, P - 1:-1] - returns) ** 2).mean()
            loss = pg + 0.1 * vloss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        ref_loss = -ref_lp.mean().item()
        history.append((task.mean().item(), kl.sum(1).mean().item(), ref_loss))
        if it % 10 == 0 or it == ITERS - 1:
            print(f"   iter {it + 1:>3}: reward {task.mean():5.2f}   KL {kl.sum(1).mean():6.2f}   "
                  f"reference loss {ref_loss:5.2f}   ({time.time() - t0:.0f} s)", flush=True)
    return policy, history


@torch.no_grad()
def evaluate(lm, label):
    torch.manual_seed(123)
    rs, rl, samples = [], [], []
    ref = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    for p in PROMPTS:
        prompt = tok(p, return_tensors="pt").input_ids.repeat(8, 1)
        ids = lm.generate(prompt, do_sample=True, top_k=0, max_new_tokens=N_NEW, min_new_tokens=N_NEW,
                          suppress_tokens=[tok.eos_token_id], pad_token_id=tok.eos_token_id,
                          attention_mask=torch.ones_like(prompt))
        texts = tok.batch_decode(ids[:, prompt.shape[1]:])
        rs += [reward(t) for t in texts]
        rl.append(-token_logprobs(ref(ids).logits, ids, prompt.shape[1]).mean().item())
        samples.append(p + texts[0])
    print(f"   {label}: reward {sum(rs) / len(rs):.2f}, reference loss {sum(rl) / len(rl):.2f} (64 samples, 8 prompts)")
    for s in samples[:3]:
        print(f"      {s!r}")


print(f"policy GPT-2 small; {ITERS} iterations of {BATCH} samples × {N_NEW} tokens; PPO {EPOCHS} epochs, clip {CLIP}")
print("\n0. before training")
evaluate(GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval(), "GPT-2")
for beta in BETAS:
    print(f"\nbeta = {beta}")
    policy, hist = run(beta)
    evaluate(policy.lm.eval(), f"after PPO, beta {beta}")
