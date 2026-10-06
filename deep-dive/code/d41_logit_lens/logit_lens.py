"""Deep Dive, episode 41: The Logit Lens.

Decode the residual stream after every layer as if it were the last one: apply the model's final norm and unembedding to
the intermediate hidden state. GPT-2 small (12 layers) and Qwen2.5-0.5B (24 layers):
1. Fact prompts: the top guess after each layer, and the probability of the right answer.
2. Over 1,024 tokens of Tiny Shakespeare: after each layer, how often the lens's top token matches the model's final
   prediction, and how often it is the actual next token.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)
FACTS = [("The Eiffel Tower is in the city of", " Paris"), ("The capital of Japan is", " Tokyo"),
         ("Romeo and Juliet was written by William", " Shakespeare"), ("One, two, three, four,", " five")]
text = open(__file__.replace("logit_lens.py", "eval_text.txt")).read()


def lens(model, name):
    """Return a function hidden_states -> logits for every layer, using the final norm and unembedding."""
    norm = model.transformer.ln_f if name == "gpt2" else model.model.norm
    return lambda h: model.lm_head(norm(h))


for name, path in (("gpt2", "openai-community/gpt2"), ("qwen", "Qwen/Qwen2.5-0.5B")):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    decode = lens(model, name)
    print(f"\n=== {path}")
    print("1. fact prompts: top token after each layer (probability of the right answer)")
    for prompt, target in FACTS:
        ids = tok(prompt, return_tensors="pt").input_ids
        tgt = tok(target).input_ids[0]
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states   # embeddings, then after each layer
            probs = [decode(h[0, -1]).softmax(-1) if i < len(hs) - 1 else
                     model(ids).logits[0, -1].softmax(-1) for i, h in enumerate(hs)]
        row = [f"{i}:{tok.decode(p.argmax())!r}({p[tgt]:.2f})" for i, p in enumerate(probs)]
        first = next((i for i, p in enumerate(probs) if p.argmax() == tgt), None)
        print(f"   {prompt!r} → {target!r}: first layer with it on top: {first}")
        print("     " + "  ".join(row))
    ids = tok(text, return_tensors="pt").input_ids[:, :1025]
    with torch.no_grad():
        out = model(ids[:, :-1], output_hidden_states=True)
        final = out.logits[0].argmax(-1)
        nxt = ids[0, 1:]
        print("2. over 1,024 tokens of Tiny Shakespeare: layer, lens top-1 = final prediction, = actual next token")
        for i, h in enumerate(out.hidden_states[:-1]):
            top = decode(h[0]).argmax(-1)
            print(f"   layer {i:>2}: agrees with final {(top == final).float().mean() * 100:5.1f}%   "
                  f"next-token accuracy {(top == nxt).float().mean() * 100:5.1f}%")
        print(f"   final   : agrees 100.0%   next-token accuracy {(final == nxt).float().mean() * 100:5.1f}%")
