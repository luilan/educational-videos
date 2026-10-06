"""Deep Dive, episode 40: Reinforcement Learning for Reasoning.

GRPO (group relative policy optimization) with a verifiable reward, on Qwen2.5-0.5B-Instruct:
- the task: arithmetic questions, answered step by step and ending with "Answer: <number>";
- the reward: 1 if the final number is right, else 0 (no reward model, no human labels);
- for each question, sample a group of G answers; each answer's advantage is its reward minus the group's mean, divided
  by the group's standard deviation; raise the log-probability of better-than-average answers, lower the others;
- a small KL penalty keeps the policy near the original model.
Memory (fits in ~8 GB): the token embeddings (shared with the output layer) stay frozen, the reference model is kept in
bfloat16, Adam updates one tensor at a time (foreach=False), and log-probabilities are computed 2 sequences at a time.
Measured on 100 held-out questions (greedy), before and after.
"""
import copy
import random
import re
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)
NAME = "Qwen/Qwen2.5-0.5B-Instruct"
STEPS, Q, G, MAX_NEW, LR, BETA, MICRO = int(sys.argv[1]) if len(sys.argv) > 1 else 30, 4, 8, 160, 2e-6, 0.04, 2
tok = AutoTokenizer.from_pretrained(NAME)
tok.padding_side = "left"
policy = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32)
policy.eval()                                                      # no dropout
policy.generation_config.repetition_penalty = 1.0                  # sample from the model itself (no default tweaks)
policy.generation_config.temperature, policy.generation_config.top_p, policy.generation_config.top_k = 1.0, 1.0, None
ref = copy.deepcopy(policy).to(torch.bfloat16).eval()              # frozen reference: half the memory
policy.get_input_embeddings().weight.requires_grad_(False)         # 136M of 494M parameters (tied with the output layer)
for p in ref.parameters():
    p.requires_grad_(False)


def question(rng):
    a, b, c = rng.randint(11, 49), rng.randint(2, 9), rng.randint(10, 99)
    return f"What is {a} × {b} + {c}?", a * b + c


def prompt(q):
    return tok.apply_chat_template([{"role": "user", "content": q + " Think step by step, then end with 'Answer: <number>'."}],
                                   tokenize=False, add_generation_prompt=True)


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*", text)
    return int(nums[-1].replace(",", "")) if nums else None


def reward(text, answer):
    m = re.findall(r"Answer:\s*\$?(-?[\d,]+)", text)
    return float(bool(m) and int(m[-1].replace(",", "")) == answer)


@torch.no_grad()
def evaluate(label, n=100):
    rng = random.Random(12345)
    qs = [question(rng) for _ in range(n)]
    ok, last_ok, finished, lengths = 0, 0, 0, []
    for i in range(0, n, 25):
        part = qs[i:i + 25]
        enc = tok([prompt(q) for q, _ in part], return_tensors="pt", padding=True)
        out = policy.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)[:, enc.input_ids.shape[1]:]
        for (q, a), o in zip(part, out):
            t = tok.decode(o, skip_special_tokens=True)
            ok += reward(t, a)
            last_ok += last_number(t) == a
            finished += (o == tok.convert_tokens_to_ids("<|im_end|>")).any().item()
            lengths.append(((o != tok.pad_token_id) & (o != tok.convert_tokens_to_ids("<|im_end|>"))).sum().item())
    print(f"   {label}: reward (right 'Answer: N') {ok}/{n}; last number in the text right {last_ok}/{n}; "
          f"finished within {MAX_NEW} tokens {finished}/{n}; mean length {sum(lengths) / n:.0f} tokens", flush=True)
    enc = tok([prompt(qs[0][0])], return_tensors="pt")
    out = policy.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)[0, enc.input_ids.shape[1]:]
    print(f"      e.g. {qs[0][0]!r} (= {qs[0][1]}) → {tok.decode(out, skip_special_tokens=True)!r}")
    return ok


def seq_logprobs(model, ids, attn, start):
    logits = model(ids, attention_mask=attn).logits[:, start - 1:-1]
    return F.log_softmax(logits.float(), -1).gather(-1, ids[:, start:, None]).squeeze(-1)


print(f"policy {NAME}; {STEPS} steps × {Q} questions × {G} samples; lr {LR}, KL beta {BETA}")
print("\n1. before training")
before = evaluate("greedy")

print("\n2. GRPO")
opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=LR, foreach=False)
rng = random.Random(0)
torch.manual_seed(0)
t0 = time.time()
history = []
for step in range(STEPS):
    qs = [question(rng) for _ in range(Q)]
    enc = tok([prompt(q) for q, _ in qs for _ in range(G)], return_tensors="pt", padding=True)
    P = enc.input_ids.shape[1]
    with torch.no_grad():
        ids = policy.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                              pad_token_id=tok.pad_token_id)
    gen = ids[:, P:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    r = torch.tensor([reward(t, qs[i // G][1]) for i, t in enumerate(texts)])
    grp = r.view(Q, G)
    adv = ((grp - grp.mean(1, keepdim=True)) / (grp.std(1, keepdim=True) + 1e-4)).view(-1)
    # mask: generated tokens up to and including the first end-of-turn token; padding never counts
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    valid = (gen != tok.pad_token_id).float()
    for i in range(len(gen)):
        stop = (gen[i] == end_id).nonzero()
        if len(stop):
            valid[i, stop[0, 0] + 1:] = 0
    attn = torch.cat([enc.attention_mask, torch.ones_like(gen)], 1)
    opt.zero_grad()
    kl_sum = 0.0
    for m in range(0, len(ids), MICRO):
        sl = slice(m, m + MICRO)
        lp = seq_logprobs(policy, ids[sl], attn[sl], P)
        with torch.no_grad():
            lr_ = seq_logprobs(ref, ids[sl], attn[sl], P)
        k3 = (lr_ - lp).exp() - (lr_ - lp) - 1                      # KL estimator (always >= 0)
        per_tok = -adv[sl, None] * lp + BETA * k3
        loss = ((per_tok * valid[sl]).sum(1) / valid[sl].sum(1)).sum() / len(ids)
        loss.backward()
        kl_sum += ((k3 * valid[sl]).sum(1) / valid[sl].sum(1)).sum().item()
    opt.step()
    history.append((r.mean().item(), kl_sum / len(ids)))
    print(f"   step {step + 1:>3}: reward {r.mean():.2f} (groups with mixed results: {((grp.sum(1) > 0) & (grp.sum(1) < G)).sum().item()}/{Q}), "
          f"KL {kl_sum / len(ids):.4f}, mean length {valid.sum(1).mean():.0f} ({time.time() - t0:.0f} s)", flush=True)

print("\n3. after training")
after = evaluate("greedy")
first, last = history[:5], history[-5:]
print(f"   training reward, first 5 steps {sum(h[0] for h in first) / 5:.2f}, last 5 steps {sum(h[0] for h in last) / 5:.2f}")
