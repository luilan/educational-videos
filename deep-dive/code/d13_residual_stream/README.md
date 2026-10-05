# Deep Dive, episode 13 · The Residual Stream

GPT-2 written as a residual stream (`x = x + attention`, `x = x + MLP`), measured layer by layer, plus a tiny GPT trained
with and without residual connections.

```bash
pip install -r requirements.txt
python residual_stream.py
```

Downloads GPT-2 small (about 500 MB) and trains two 8-layer tiny GPTs (about 10 minutes on a CPU). On 512 tokens of
Shakespeare it prints:

1. **the stream's size** entering each layer: 4.6 (embedding), 49.5, … 216.1, and 554.9 at the end; the first token's
   grows past 3,000 (the attention sink);
2. **each update ÷ the stream**: 15–34% in layers 1–9; 7.1× and 5.3× in layer 0; up to 1.41× in layer 11;
3. **delete one layer** (baseline loss 4.13): 3.74–4.33 for middle layers, 7.95 without layer 0, 6.12 without layer 11
   (`run(ids, skip={3, 6})` deletes several);
4. **8-layer tiny GPT**, validation loss at 100 / 250 / 500 / 1,000 / 1,500 steps:
   with residuals 2.46 / 2.18 / 1.96 / 1.77 / 1.69; without, 3.37 / 3.36 / 3.36 / 3.36 / 3.36 (letter frequencies
   alone: 3.35).
