# Deep Dive, episode 22 · Mixed Precision

16-bit number formats, what they break, and how mixed precision training avoids it.

```bash
pip install -r requirements.txt
python mixed_precision.py
```

Downloads GPT-2 small (about 500 MB) and trains three tiny GPTs (about 15 minutes on a CPU). It prints:

1. **formats** (`torch.finfo`): float32 / float16 / bfloat16 largest value, step after 1.0, smallest normal number;
2. **rounding**: 1.0 + 0.001 → 1.0010000 / 1.0009766 / 1.0000000; 1.0 + 0.0001 → 1.0001000 / 1.0 / 1.0;
3. **underflow**: a gradient of 10⁻⁹ is 0 in float16, survives with loss scaling (× 1024, ÷ 1024), and in bfloat16;
4. **training** (4 layers, 2,000 steps): float32 1.644; bfloat16 autocast + float32 weights 1.645; pure bfloat16
   weights 1.677;
5. **GPT-2** on 1,024 tokens of Shakespeare: float32 475 MiB, loss 3.9996; bfloat16 237 MiB, loss 3.9897.
