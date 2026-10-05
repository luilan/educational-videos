"""LLMs in Practice, episode 10: Quantization — shrinking a model to fit a laptop.

Round every weight matrix of Qwen2.5-0.5B-Instruct to 8, 4, 3 and 2 bits (round-to-nearest, one scale per output
row), then measure the size, how far the weights moved, and the loss on real text the model never saw in this form:
the narration of How LLMs Work. Quantization is simulated (weights are rounded, then stored back as floats) so the
same code runs anywhere; the sizes are computed from the bit widths.

    pip install -r requirements.txt
    python quantize.py
Downloads Qwen2.5-0.5B-Instruct (about 1 GB); runs on a CPU in a few minutes.
"""
import copy
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
text = (Path(__file__).resolve().parents[3] / "how-llms-work" / "NARRATION.md").read_text()
ids = tok(text, return_tensors="pt").input_ids[:, :4096]


def quantize_(w, bits):
    """Round each row of w to 2**bits levels between -max and +max (symmetric, one float scale per row)."""
    levels = 2 ** (bits - 1) - 1                       # e.g. 8 bits -> integers -127 … 127
    scale = w.abs().amax(dim=1, keepdim=True) / levels
    q = torch.clamp(torch.round(w / scale), -levels, levels)
    return q * scale                                    # what the model will compute with


@torch.no_grad()
def loss(model):
    total, n = 0.0, 0
    for i in range(0, ids.shape[1] - 1, 1024):
        chunk = ids[:, i:i + 1024]
        out = model(chunk, labels=chunk)
        total += out.loss.item() * (chunk.shape[1] - 1)
        n += chunk.shape[1] - 1
    return total / n


matrices = [m for name, m in base.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in name]
n_matrix = sum(m.weight.numel() for m in matrices)
n_total = sum(p.numel() for p in base.parameters())
print(f"{n_total:,} parameters, {n_matrix:,} in the transformer's weight matrices ({n_matrix / n_total:.0%})")
print(f"text: {ids.shape[1]:,} tokens of How LLMs Work narration\n")
print(f"{'format':<8}{'size of matrices':>18}{'avg weight change':>20}{'loss':>8}")
print(f"{'fp32':<8}{n_matrix * 4 / 2**20:>15.0f} MB{'0':>20}{loss(base):>8.3f}")
print(f"{'bf16':<8}{n_matrix * 2 / 2**20:>15.0f} MB")
example = matrices[0].weight[0, :6].clone()


def answer(model):
    msg = [{"role": "user", "content": "In one sentence, what is a context window?"}]
    prompt = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
    out = model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=40, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, prompt.shape[1]:], skip_special_tokens=True).strip()


print(f"         answer: {answer(base)!r}")
for bits in (8, 4, 3, 2):
    model = copy.deepcopy(base)
    change = 0.0
    for m in [m for name, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in name]:
        w = m.weight.data
        q = quantize_(w, bits)
        change += ((q - w).abs().sum() / w.abs().sum()).item() * w.numel() / n_matrix
        m.weight.data = q
    size = n_matrix * bits / 8 / 2**20
    print(f"{'int' + str(bits):<8}{size:>15.0f} MB{change:>19.1%}{loss(model):>8.3f}")
    if bits == 4:
        print(f"         first weights before {[round(x, 4) for x in example.tolist()]}")
        print(f"                       after {[round(x, 4) for x in model.model.layers[0].self_attn.q_proj.weight[0, :6].tolist()]}")
    print(f"         answer: {answer(model)!r}")
