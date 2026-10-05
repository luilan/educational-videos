# Deep Dive, episode 12 · Attention Sinks

Why GPT-2's heads stare at the first token, and why a streaming cache must keep it.

```bash
pip install -r requirements.txt
python attention_sinks.py
```

Downloads GPT-2 small (about 500 MB). The forward pass is written by hand (it matches `transformers` to 2 × 10⁻⁴ in the
logits) so that any attention mask can be used. It prints:

1. **attention on the first token**, 256 tokens of Shakespeare, all 144 heads: 0.393 on average, 59 heads above half;
   per layer from 0.01 (layer 0) to 0.61 (layer 7); the same (0.392–0.395) whatever the first token is, or if the text
   starts mid-sentence;
2. **value vector size**, layers 2–11: first token 1.42, other tokens 6.42;
3. **streaming**, 1,024 tokens, loss on positions 512–1,022:

| attention | loss | perplexity |
|---|---|---|
| full | 3.87 | 48 |
| window 256 | 7.67 | 2,133 |
| window 256 + first 4 tokens | 3.76 | 43 |
