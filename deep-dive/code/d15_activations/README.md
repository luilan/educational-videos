# Deep Dive, episode 15 · Activations: ReLU, GELU, SwiGLU

The bend inside every MLP: GELU in GPT-2, SwiGLU in Qwen2.5, and a fair comparison in a tiny GPT.

```bash
pip install -r requirements.txt
python activations.py
```

Downloads GPT-2 small and Qwen2.5-0.5B-Instruct (about 1.5 GB) and trains nine tiny GPTs (about 40 minutes on a CPU).
It prints:

1. **the functions** at a few points; GELU's lowest value is −0.170;
2. **GPT-2, layer 6**: 86% of the 3,072 hidden values are negative before GELU, only 4% end up within 0.01 of zero;
   swapping every GELU for ReLU without retraining raises the loss from 4.128 to 7.227;
3. **Qwen2.5's SwiGLU by hand**, `down(silu(gate(x)) * up(x))`: difference from the model 0.0; 13,074,432 parameters per
   MLP;
4. **tiny GPT**, 4 layers, 3,000 steps, about the same MLP parameters, three seeds:

| activation | MLP parameters | val loss (seeds 1337 / 1 / 2) | mean |
|---|---|---|---|
| ReLU | 131,072 | 1.653 / 1.649 / 1.652 | 1.651 |
| GELU | 131,072 | 1.620 / 1.617 / 1.625 | 1.620 |
| SwiGLU | 132,096 | 1.590 / 1.592 / 1.593 | 1.591 |

In the ReLU model (seed 1337), all 2,048 hidden units fired at least once on 40,960 validation tokens.
