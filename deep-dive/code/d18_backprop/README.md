# Deep Dive, episode 18 · Backprop Through a Transformer

The chain rule by hand, autograd checked against finite differences on a real GPT-2 weight, gradient sizes per layer,
and the time and memory cost of the backward pass.

```bash
pip install -r requirements.txt
python backprop.py
```

Downloads GPT-2 small (about 500 MB) and needs about 4 GB of RAM (a float64 copy for the gradient check). It prints:

1. **chain rule**: loss = (tanh(w·x) − 3)², w = 0.5, x = 2: by hand −3.760292, autograd −3.760291;
2. **finite differences** (float64): layer 5 MLP weight [100, 200]: autograd −0.0000200195, (loss(w+ε) − loss(w−ε)) / 2ε
   = −0.0000200195. Note: the loss is computed with our own `F.cross_entropy`, because the library's built-in loss
   converts the logits to float32, which hides changes this small;
3. **gradient size per layer**: about 2 to 4 in layers 0–7, under 1 in layers 10–11; all 50,257 embedding rows get a
   gradient (the embedding is shared with the output layer);
4. **cost**, 1,024 tokens, float32: 1,443 MiB of saved activations (weights: 475 MiB); forward 0.56 s, backward 1.21 s
   on an 8-thread CPU (times vary by machine).
