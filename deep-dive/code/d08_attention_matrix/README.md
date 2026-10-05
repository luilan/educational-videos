# Deep Dive, episode 8 · The Attention Matrix, Entry by Entry

One GPT-2 attention head recomputed by hand with the real weights, every step checked against `transformers`.

```bash
pip install -r requirements.txt
python attention_matrix.py
```

GPT-2 small (about 500 MB) reads “The cat sat on the mat because it was tired”. For layer 4, head 3 (counting from 0)
it prints:

1. **one entry**: query of “it” · key of “cat” = 4.552, divided by √64 = 8 → 0.569;
2. **the matrix**: 10 × 10 weights after the mask and softmax, identical to `transformers` (difference 0.0); “it” gives
   0.837 of its attention to “cat”;
3. **why divide by 8**: random 64-number dot products have a standard deviation of 8.00; without the division the
   largest weight per row averages 0.979 instead of 0.684;
4. **values and heads**: weights × values, 12 heads joined to 768 numbers and mixed by `c_proj`, identical to
   `transformers`;
5. **all 144 heads**: layer 4, head 11 puts 1.00 of its attention on the previous token; 92 of 144 heads put more than
   half on the first token.
