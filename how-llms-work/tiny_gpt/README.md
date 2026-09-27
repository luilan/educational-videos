# Tiny GPT

The character-level GPT built and trained in episodes 11–12 of *How LLMs Work*: about 100 lines of PyTorch,
assembled from the same pieces the series explains (token + position embeddings, causal multi-head attention,
GELU MLP, pre-norm residual blocks, a linear head).

| | |
|---|---|
| Data | `input.txt`: Tiny Shakespeare, 1,115,394 characters, 65-character vocabulary; 90 % train / 10 % validation |
| Model | 4 layers, 4 heads, 128-dim embeddings, context 64 → **818,241 parameters** |
| Training | 5,000 steps, batch 32, AdamW lr 1e-3; about 45 minutes on an 8-core CPU |
| Result | validation loss 4.41 → 2.20 (step 250) → 1.79 (step 1,000) → **1.59** (step 5,000) |

```bash
cd how-llms-work/tiny_gpt
python tiny_gpt.py          # prints samples at steps 0, 250, 1000, 5000; writes training_log.json
```

`training_log.json` is the log from the run shown in the videos (per-step training loss, validation loss every
250 steps, and the generated samples). Episodes 11, 12 and Foundations F14 read it to draw their charts, so re-running
the training will change those videos.
