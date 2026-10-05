# Deep Dive, episode 17 · Cross-Entropy, Deeper

What the loss number measures: GPT-2's per-token losses, perplexity and bits, confident mistakes, calibration, and the
gradient.

```bash
pip install -r requirements.txt
python cross_entropy.py
```

Downloads GPT-2 small (about 500 MB). It prints:

1. **token by token** on “The capital of France is Paris, and the capital of Italy is Rome.”: “Paris” 3% (loss 3.43),
   “Rome” 58% (0.55); average loss 2.344;
2. **perplexity** e^2.344 = 10.4, **3.38 bits per token**; on 1,024 tokens of Shakespeare: loss 4.000, perplexity 54.6,
   1.87 bits per character;
3. **confident mistakes**: −ln p for p = 0.9 … 0.001; the worst 10% of Shakespeare tokens are 32% of the loss;
4. **calibration**: confidence 0.11 / 0.29 / 0.49 / 0.70 / 0.96 vs fraction right 0.14 / 0.33 / 0.46 / 0.67 / 0.67;
   95% of the 81 confident mistakes are a predicted line break (GPT-2 expects a blank line; this file has none);
5. **the gradient** w.r.t. 5 logits: softmax − one-hot, identical to autograd.
