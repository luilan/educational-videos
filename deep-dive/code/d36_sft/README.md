# Deep Dive, episode 36 · Supervised Fine-Tuning

Turn the base Qwen2.5-0.5B into a tiny assistant: full fine-tuning on 280 chat-formatted conversations (additions,
capitals, words in capital letters), with the loss only on the assistant's tokens (labels −100 elsewhere).

```bash
pip install -r requirements.txt
python sft.py
```

Needs about 8 GB of RAM; 75 steps of 8 examples take a few minutes on an 8-thread CPU. Results on 60 held-out questions
(new numbers, countries, words), identical over two runs:

| | exact answer | right answer anywhere | stops by itself | mean length |
|---|---|---|---|---|
| base, chat format | 0% | 15% | 0% | 40 tokens |
| base, plain “Question: … Answer:” prompt | 40% | 75% | – | – |
| fine-tuned, chat format | 95% | 95% | 100% | 7 tokens |

Answer loss: 4.24 → 0.14. The 3 misses: “48 + 45 = 113.”, “Kiev” for Kyiv, FLOWER → “FLORAL”.
