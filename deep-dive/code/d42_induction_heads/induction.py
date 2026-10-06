"""Deep Dive, episode 42: Induction Heads.

GPT-2 small reads 50 random tokens, then the same 50 tokens again.
1. Loss on the first copy (unpredictable) vs the second (predictable, if the model copies from context).
2. Induction score of every head: in the second copy, the attention a token pays to the token right after its earlier
   occurrence ("A B ... A → look at B").
3. Ablation: remove the top induction heads (head_mask = 0) and measure the second-copy loss again; compare with removing
   the same number of random heads.
"""
import random

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

torch.set_num_threads(8)
torch.manual_seed(0)
random.seed(0)
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager").eval()
L, H, T, BATCH = 12, 12, 50, 8
seq = torch.randint(1000, 30000, (BATCH, T))
ids = torch.cat([seq, seq], 1)                                     # 100 tokens: random, then the same again


@torch.no_grad()
def halves(head_mask=None):
    out = model(ids, head_mask=head_mask, output_attentions=True)
    lp = F.cross_entropy(out.logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
    return lp[:, :T - 1].mean().item(), lp[:, T:].mean().item(), out.attentions


first, second, att = halves()
print(f"1. {BATCH} sequences of {T} random tokens, repeated")
print(f"   loss on the first copy {first:.2f}, on the second copy {second:.2f}")

score = torch.zeros(L, H)
for l in range(L):
    a = att[l]                                                      # batch, heads, query, key
    for t in range(T, 2 * T):
        score[l] += a[:, :, t, t - T + 1].mean(0)
    score[l] /= T
print("\n2. induction score (attention to the token after the earlier occurrence), layer × head")
for l in range(L):
    print(f"   layer {l:>2}: " + " ".join(f"{v:.2f}" for v in score[l].tolist()))
flat = [(score[l, h].item(), l, h) for l in range(L) for h in range(H)]
flat.sort(reverse=True)
top = flat[:6]
print("   strongest: " + ", ".join(f"{l}.{h} ({s:.2f})" for s, l, h in top))
print(f"   heads scoring above 0.3: {sum(s > 0.3 for s, _, _ in flat)} of {L * H}")


print("\n3. ablation: second-copy loss with heads removed")
for k in (2, 4, 6):
    mask = torch.ones(L, H)
    for _, l, h in flat[:k]:
        mask[l, h] = 0
    _, s_top, _ = halves(mask)
    rand = []
    for trial in range(5):
        m = torch.ones(L, H)
        for l, h in random.sample([(l, h) for l in range(L) for h in range(H)], k):
            m[l, h] = 0
        rand.append(halves(m)[1])
    print(f"   remove top {k} induction heads: {s_top:.2f}   remove {k} random heads (mean of 5): {sum(rand) / 5:.2f}")
