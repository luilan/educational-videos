"""Deep Dive, episode 35: Decoding Strategies Compared.

GPT-2 small continues 4 open-ended prompts for 120 tokens with each strategy (3 seeded samples per prompt for
the random ones). Two measurements per text:
- repetition: the share of its 4-grams (of tokens) that already appeared earlier in the same continuation;
- judge loss: how surprising a bigger model (Qwen2.5-1.5B) finds the continuation (mean cross-entropy, lower = more
  predictable; note that predictable is not the same as good).
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, GenerationConfig

torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
gen = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", dtype=torch.float32).eval()
jtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
judge = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.float32).eval()
N, SAMPLES = 120, 3
PROMPTS = [
    "The old lighthouse keeper opened the door and",
    "In the year 2150, the first city on Mars",
    "My favourite thing about autumn is",
    "The scientist looked at the results and realised that",
]


def pick(logits, method, arg, g):
    """One next token from the logits of the last position."""
    if method == "greedy":
        return logits.argmax()
    if method == "temperature":
        logits = logits / arg
    elif method == "top-k":
        kth = logits.topk(arg).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    elif method == "top-p":
        p, idx = logits.softmax(-1).sort(descending=True)
        drop = p.cumsum(-1) - p > arg                              # keep the smallest set reaching probability arg
        logits = logits.clone()
        logits[idx[drop]] = float("-inf")
    elif method == "min-p":
        p = logits.softmax(-1)
        logits = logits.masked_fill(p < arg * p.max(), float("-inf"))
    return torch.multinomial(logits.softmax(-1), 1, generator=g)[0]


@torch.no_grad()
def generate(prompt, method, arg, seed):
    g = torch.Generator().manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").input_ids
    if method == "beam":
        cfg = GenerationConfig(max_new_tokens=N, min_new_tokens=N, num_beams=arg, do_sample=False, length_penalty=1.0,
                               repetition_penalty=1.0, eos_token_id=None, pad_token_id=tok.eos_token_id)
        return gen.generate(ids, attention_mask=torch.ones_like(ids), generation_config=cfg)[0, ids.shape[1]:]
    cache, inp, out = DynamicCache(), ids, []
    for _ in range(N):
        logits = gen(inp, past_key_values=cache, use_cache=True).logits[0, -1]
        nxt = pick(logits, method, arg, g)
        out.append(nxt.item())
        inp = nxt.view(1, 1)
    return torch.tensor(out)


def repetition(c):
    grams = [tuple(c[i:i + 4].tolist()) for i in range(len(c) - 3)]
    seen, rep = set(), 0
    for x in grams:
        rep += x in seen
        seen.add(x)
    return rep / len(grams)


@torch.no_grad()
def judge_loss(prompt, c):
    """The judge re-tokenizes the text with its own tokenizer and scores only the continuation's tokens."""
    p = jtok(prompt, return_tensors="pt").input_ids[0]
    full = jtok(prompt + tok.decode(c), return_tensors="pt").input_ids[0]
    logits = judge(full[None]).logits[0, len(p) - 1:-1]
    return F.cross_entropy(logits.double(), full[len(p):]).item()


METHODS = [("greedy", "greedy", None), ("beam search, 4 beams", "beam", 4),
           ("temperature 0.7", "temperature", 0.7), ("temperature 1.0", "temperature", 1.0),
           ("temperature 1.5", "temperature", 1.5), ("top-k 40", "top-k", 40), ("top-p 0.9", "top-p", 0.9),
           ("min-p 0.1", "min-p", 0.1)]
examples = {}
print(f"{N} new tokens, 4 prompts; random methods: {SAMPLES} samples per prompt")
print(f"{'method':>22}  repetition  judge loss")
for name, method, arg in METHODS:
    reps, losses = [], []
    runs = 1 if method in ("greedy", "beam") else SAMPLES
    for pi, p in enumerate(PROMPTS):
        for s in range(runs):
            c = generate(p, method, arg, seed=100 * pi + s)
            reps.append(repetition(c))
            losses.append(judge_loss(p, c))
            if pi == 0 and s == 0:
                examples[name] = tok.decode(c)
    print(f"{name:>22}  {sum(reps) / len(reps) * 100:8.0f}%  {sum(losses) / len(losses):9.2f}", flush=True)

print("\nexamples, prompt 1:", repr(PROMPTS[0]))
for name, text in examples.items():
    print(f"--- {name}\n{text[:400]!r}")

# ---------------------------------------------------------------- what each truncation keeps
print("\nhow many tokens each truncation keeps (top-k 40, top-p 0.9, min-p 0.1)")
for ctx in (PROMPTS[0], PROMPTS[0] + " saw the man standing there. He was wearing a black suit and a black"):
    with torch.no_grad():
        p = gen(tok(ctx, return_tensors="pt").input_ids).logits[0, -1].softmax(-1)
    ps = p.sort(descending=True).values
    print(f"   ...{ctx[-30:]!r}: top token {ps[0]:.3f}; top-k keeps 40 ({ps[:40].sum() * 100:.0f}% of the probability), "
          f"top-p keeps {int(((ps.cumsum(0) - ps) < 0.9).sum())}, min-p keeps {int((p >= 0.1 * p.max()).sum())}")
