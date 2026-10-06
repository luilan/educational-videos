# Deep Dive, episode 30 · Multimodal: Images as Tokens

A tiny vision-language model built from scratch: image patches become tokens, and one causal transformer writes a
caption after them.

```bash
pip install -r requirements.txt
python multimodal.py
```

PyTorch only, no downloads (about 10 minutes on a CPU; seeded). It prints:

1. a 32 × 32 image = 16 patches of 8 × 8 × 3 = 192 numbers, each projected to a 128-number token (24,704 parameters in
   the patch projection; 830,103 in total); caption loss 3.164 → 0.040 (step 250) → 0.009 (step 1,500);
2. on 200 new synthetic images: shape 91%, color 100%, position 100%; mistakes: square → circle ×10, circle → square ×4,
   circle → triangle ×2, cross → circle ×1, circle → cross ×1.
