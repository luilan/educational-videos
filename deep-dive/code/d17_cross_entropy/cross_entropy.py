"""Deep Dive, episode 17: Cross-Entropy, Deeper.

1. The loss of one token is -log(probability of the right token): GPT-2 on one sentence, token by token.
2. Perplexity = e^loss; bits per token = loss / ln 2.
3. Confident mistakes are expensive: the loss of a token predicted at 50%, 10%, 1%, 0.1%.
4. Calibration: when GPT-2's top guess has 30% probability, is it right 30% of the time?
5. The gradient with respect to the logits is softmax minus one-hot, checked with autograd.
"""
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = Path(__file__).parent
torch.set_num_threads(4)
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tok.model_max_length = 10 ** 6
gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()

# ---------------------------------------------------------------- 1. token by token
sentence = "The capital of France is Paris, and the capital of Italy is Rome."
ids = tok(sentence, return_tensors="pt").input_ids
with torch.no_grad():
    probs = gpt2(ids).logits[0, :-1].softmax(-1)
p_right = probs[torch.arange(ids.shape[1] - 1), ids[0, 1:]]
print("1. GPT-2, token by token: probability of the actual next token, and its loss -ln(p)")
for t, p in zip(ids[0, 1:], p_right):
    print(f"   {tok.decode(t)!r:>10}  p = {p:.4f}   loss {-math.log(p):.2f}")
loss = -p_right.log().mean().item()
print(f"   average loss {loss:.3f}")

# ---------------------------------------------------------------- 2. perplexity and bits
print(f"\n2. perplexity e^{loss:.3f} = {math.exp(loss):.1f}; {loss / math.log(2):.2f} bits per token")
text = (HERE.parents[2] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()
sh = tok(text[100_000:106_000], return_tensors="pt").input_ids[:, :1024]
with torch.no_grad():
    sh_loss = gpt2(sh, labels=sh).loss.item()
n_chars = len(tok.decode(sh[0, 1:]))
print(f"   on 1,024 tokens of Shakespeare: loss {sh_loss:.3f}, perplexity {math.exp(sh_loss):.1f}, "
      f"{sh_loss / math.log(2) * 1023 / n_chars:.2f} bits per character")

# ---------------------------------------------------------------- 3. confident mistakes
print("\n3. loss of the right token at different probabilities")
for p in (0.9, 0.5, 0.1, 0.01, 0.001):
    print(f"   p = {p:<6} loss {-math.log(p):.2f}")
with torch.no_grad():
    logits = gpt2(sh).logits[0, :-1]
lp = logits.log_softmax(-1)[torch.arange(1023), sh[0, 1:]]
worst = (-lp).topk(5)
print(f"   share of the Shakespeare loss from the worst 10% of tokens: "
      f"{(-lp).topk(102).values.sum() / (-lp).sum():.0%}")

# ---------------------------------------------------------------- 4. calibration
top_p, top_i = logits.softmax(-1).max(-1)
right = (top_i == sh[0, 1:]).float()
print("\n4. calibration on 1,023 Shakespeare tokens: top guess's probability vs how often it is right")
for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
    m = (top_p >= lo) & (top_p < hi)
    print(f"   {lo:.1f}-{min(hi, 1):.1f}: {m.sum().item():>4} tokens, average confidence {top_p[m].mean():.2f}, "
          f"right {right[m].mean():.2f}")

wrong = (top_p >= 0.8) & (right == 0)
nl = tok("\n").input_ids[0]
print(f"   confident (≥ 0.8) but wrong: {wrong.sum().item()} tokens; "
      f"{(top_i[wrong] == nl).float().mean():.0%} of them predicted a line break where the text continues "
      f"(GPT-2 expects a blank line there; this file has none)")

# ---------------------------------------------------------------- 5. the gradient
torch.manual_seed(0)
z = torch.randn(5, requires_grad=True)
target = torch.tensor(2)
F.cross_entropy(z[None], target[None]).backward()
expected = z.softmax(-1).detach() - F.one_hot(target, 5)
print(f"\n5. gradient of the loss w.r.t. 5 logits (target index 2):")
print(f"   autograd:           {z.grad.numpy().round(3)}")
print(f"   softmax − one-hot:  {expected.numpy().round(3)}")
