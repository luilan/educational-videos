# Episode 10 · Quantization: Shrinking a Model to Fit a Laptop

Round every weight matrix of `Qwen/Qwen2.5-0.5B-Instruct` to fewer bits (round-to-nearest, one scale per output
row), and measure size, weight change, loss on real text (the How LLMs Work narration) and an answer.

```bash
pip install -r requirements.txt
python quantize.py
```

Downloads the model (about 1 GB); a few minutes on a CPU. Quantization is simulated (weights are rounded, then kept
as floats) so it runs anywhere; sizes are computed from the bit widths.

| format | size of weight matrices | avg weight change | loss |
|---|---|---|---|
| fp32 | 1,365 MB | 0 | 2.836 |
| int8 | 341 MB | 1.1% | 2.837 |
| int6 | 256 MB | 4.4% | 2.875 |
| int5 | 213 MB | 9.0% | 2.944 |
| int4 | 171 MB | 19.2% | 3.582 |
| int3 | 128 MB | 43.3% | 12.491 |
| int2 | 85 MB | 93.1% | 16.610 (gibberish) |

(The script runs 8, 4, 3 and 2 bits; change the loop to `(6, 5)` for the other rows.)
