# Deep Dive, episode 41 · The Logit Lens

Decode the residual stream after every layer with the model's own final norm and unembedding. GPT-2 small and
Qwen2.5-0.5B.

```bash
pip install -r requirements.txt
python logit_lens.py
```

Results (deterministic):

- GPT-2, “The Eiffel Tower is in the city of”: “the” early on, Rome (layer 9), London (10), Paris (11 and output).
- GPT-2, “Romeo and Juliet was written by William”: “William” to layer 7; Shakespeare 0.36 / 0.91 / 1.00 / 0.96 at layers
  8–11, 0.21 at the output.
- GPT-2 over 1,024 tokens of `eval_text.txt` (Tiny Shakespeare): lens top-1 = final prediction 6.9% at layer 5, 30.0% at
  8, 56.3% at 11.
- Qwen2.5-0.5B: junk tokens in the lens until late; agreement under 3% up to layer 16, 45.7% at 23; Paris first on top
  at layer 22 of 24.
