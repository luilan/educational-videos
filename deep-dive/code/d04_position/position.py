"""How LLMs Work: Deep Dive, episode 4 — Why attention needs position.

1. Attention without position is blind to word order: shuffle the input, and every output vector is the same,
   just shuffled (permutation equivariance). Demonstrated with real GPT-2 weights, position embeddings switched off.
2. With GPT-2's learned position embeddings switched back on, order matters.
3. What GPT-2 learned: its 1,024 position vectors, and how similar nearby positions are.

    pip install -r requirements.txt
    python position.py
Downloads GPT-2 small (about 500 MB); runs on a CPU.
"""
import torch
from transformers import AutoTokenizer, GPT2Model

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = GPT2Model.from_pretrained("openai-community/gpt2").eval()
torch.manual_seed(0)


def last_layer(ids, use_position, causal=False):
    """Run GPT-2's 12 blocks on token embeddings, with or without position embeddings.
    By default the causal mask is turned off, so every token can see every other (as in an encoder)."""
    T = ids.shape[1]
    x = model.wte(ids)
    if use_position:
        x = x + model.wpe(torch.arange(T))
    mask = torch.zeros(1, 1, T, T)
    if causal:                                    # each token sees only itself and earlier tokens
        mask = mask.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
    for block in model.h:
        x = block(x, attention_mask=mask)[0]
    return model.ln_f(x)[0]


a = tok(" dog bites man", return_tensors="pt").input_ids
b = tok(" man bites dog", return_tensors="pt").input_ids
print("tokens:", [tok.decode([int(i)]) for i in a[0]], "and", [tok.decode([int(i)]) for i in b[0]])

with torch.no_grad():
    for use_position in (False, True):
        out_a, out_b = last_layer(a, use_position), last_layer(b, use_position)
        # " dog" is token 0 in a and token 2 in b; compare the final vector for the same word
        same_dog = torch.nn.functional.cosine_similarity(out_a[0], out_b[2], dim=0).item()
        same_man = torch.nn.functional.cosine_similarity(out_a[2], out_b[0], dim=0).item()
        sentence = torch.nn.functional.cosine_similarity(out_a.mean(0), out_b.mean(0), dim=0).item()
        diff = (out_a[[2, 1, 0]] - out_b).abs().max().item()
        print(f"\nposition embeddings {'ON ' if use_position else 'OFF'}:")
        print(f"  ' dog' in both sentences, cosine similarity {same_dog:.6f}")
        print(f"  ' man' in both sentences, cosine similarity {same_man:.6f}")
        print(f"  whole sentences (mean vector), cosine similarity {sentence:.6f}")
        print(f"  largest difference after reordering: {diff:.2e}")

# A twist: with a causal mask, order leaks in even without position embeddings, because the first token sees only
# itself and the last token sees everything.
with torch.no_grad():
    out_a, out_b = last_layer(a, False, causal=True), last_layer(b, False, causal=True)
    same_dog = torch.nn.functional.cosine_similarity(out_a[0], out_b[2], dim=0).item()
    print(f"\nno position embeddings, causal mask ON: ' dog' in both sentences, cosine similarity {same_dog:.6f}")

# 3. What GPT-2 learned
wpe = model.wpe.weight.detach()                                   # (1024, 768)
print(f"\nGPT-2 position embeddings: {tuple(wpe.shape)} -> one learned vector per position, up to 1,024")
sim = torch.nn.functional.cosine_similarity(wpe[100], wpe, dim=1)
for d in (0, 1, 2, 5, 10, 50, 200):
    print(f"  position 100 vs {100 + d:>4}: cosine similarity {sim[100 + d]:.3f}")
