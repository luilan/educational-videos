# How Generative Models Work, episode 16 · Audio Tokens: Neural Codecs

A tiny neural audio codec in the style of SoundStream and EnCodec, trained on a CPU on speech from this series'
narrator voices, then compared with two real codecs, Meta's EnCodec and Kyutai's Mimi.

```bash
pip install -r requirements.txt
python3 make_data.py      # ~20 minutes → data/ (30 min of Kokoro speech for training, 40 held-out test sentences)
python3 codec.py          # ~2 hours on 16 threads → out/codec.pt, out/train_log.json
python3 evaluate.py       # ~25 minutes → results.json, tokens.json, out/clip_*.wav, out/mel_*.png
```

`out/codec.pt` is included, so `evaluate.py` runs without training (it still needs `make_data.py`'s test sentences).
`make_data.py` needs Kokoro's model files in `KOKORO_DIR` (default `../../../tts`). EnCodec
(`facebook/encodec_24khz`), Mimi (`kyutai/mimi`) and Whisper small.en come from Hugging Face.

## The codec (`codec.py`)

- **Encoder:** a 7-wide convolution, then four blocks of two residual units and a strided convolution, strides
  2, 4, 5 and 8 (2 · 4 · 5 · 8 = 320), channels 16 → 256, then a convolution down to 64 numbers. 24,000 samples
  per second become **75 vectors per second** of 64 numbers, one every 13.3 ms.
- **Residual vector quantization:** 8 codebooks of 1,024 vectors. The first codebook replaces each vector by its
  nearest codebook vector; the second quantizes what is left (the residual), and so on. Each codebook contributes
  one 10-bit token per frame: 75 × 10 = 750 bit/s per codebook, **6 kbit/s with all 8**. Codebook vectors are
  updated as moving averages of the encoder outputs assigned to them (decay 0.99); unused vectors restart at
  random encoder outputs. Gradients skip the rounding (straight-through estimator), and a commitment term
  (weight 0.25) keeps the encoder close to its codebook vectors.
- **Decoder:** the mirror image with transposed convolutions.
- **Loss:** the distance between log-mel spectrograms of input and output at three window sizes (512, 1,024,
  2,048), plus 0.1 × the waveform's absolute error. No discriminator (GAN).
- **Quantizer dropout** (as in SoundStream): each step uses a random number of codebooks (1–8), so one model works
  at every bitrate.
- Encoder 849,504 parameters, decoder 914,977. 12,000 steps of 16 crops of 0.53 s, AdamW (learning rate 6e-4, cosine), 153 minutes on the CPU, sharing it with other jobs (about 0.5 s per step alone). Codebook use collapsed to a handful of vectors in the first 250 steps, until the dead-vector restarts kicked in; from step 300 on, 300–400 different vectors of the first codebook appear in each batch.

## Data (`make_data.py`)

Sentences from the narration scripts of this channel's other series (6–28 words each), spoken by 8 Kokoro voices
(af_heart, af_bella, af_nicole, af_sarah, am_michael, am_adam, bf_emma, bm_george), until 30 minutes of speech: 379
sentences. Test: 40 sentences from this series' own scripts, never trained on, spoken by af_heart, the narrator.

## Results (2026-11-25)

**Residual quantization.** Share of the encoder output (squared length) still unexplained after each codebook,
averaged over the 40 test sentences: 5.3%, 1.9%, 0.9%, 0.4%, 0.3%, 0.2%, 0.1%, 0.1%. The first codebook captures
95%. Quantizer dropout trains the encoder to put most of the information where one codebook can
reach it. Codebook use (entropy of the chosen tokens): 8.3 bits of 10 for the first codebook,
9.6–9.8 for the others.

**Quality on 40 held-out sentences** (narrator voice, this series' scripts). *Words changed*: Whisper small.en's
transcript of the codec output compared with its transcript of the original audio (word error rate, numbers spelled
out), so the original scores 0 and only the codec's damage counts. *Word errors*: the same transcript against the
script text (the original voice already scores 5.2%: numbers, U-Net, DDPM …).
*Mel distance*: mean absolute difference of log-mel spectrograms (80 bands, FFT 1,024), 0 = identical.

| system | kbit/s | words changed | sentences unchanged | word errors | mel distance |
|---|---|---|---|---|---|
| original (16-bit, 24 kHz) | 384 | 0.0% | 40/40 | 5.2% | 0.000 |
| ours, 1 codebook | 0.75 | 4.4% | 30/40 | 7.5% | 0.940 |
| ours, 2 codebooks | 1.5 | 1.5% | 34/40 | 5.4% | 0.845 |
| ours, 4 codebooks | 3 | 1.2% | 36/40 | 5.6% | 0.815 |
| ours, 8 codebooks | 6 | 1.4% | 35/40 | 5.6% | 0.810 |
| ours, no quantization (64 floats × 75/s) | 153.6 | 1.4% | 35/40 | 5.6% | 0.809 |
| EnCodec 24 kHz | 1.5 | 4.5% | 31/40 | 6.7% | 1.168 |
| EnCodec 24 kHz | 3 | 1.8% | 36/40 | 5.8% | 0.976 |
| EnCodec 24 kHz | 6 | 0.8% | 38/40 | 5.1% | 0.831 |
| EnCodec 24 kHz | 12 | 0.3% | 39/40 | 4.9% | 0.724 |
| Mimi, 8 codebooks | 1.1 | 0.7% | 37/40 | 5.0% | 1.220 |
| Mimi, 32 codebooks | 4.4 | 0.5% | 39/40 | 5.4% | 0.965 |

- One codebook (0.75 kbit/s) changes 4.4% of the words; two (1.5 kbit/s) only
  1.5%. Beyond four codebooks nothing improves: the mel distance with 8
  codebooks (0.810) equals the one with no quantization at all
  (0.809). The small decoder, trained without a discriminator, is the limit.
- Our codec has a lower mel distance than EnCodec at 6 kbit/s and than Mimi, because mel distance is exactly its
  training loss. EnCodec and Mimi are trained with discriminators (and Mimi's first codebook is distilled from a
  speech model), which reward sounding real rather than matching a spectrogram; Mimi changes only
  0.7% of the words at 1.1 kbit/s, with the worst mel distance. Listen to the clips.
- Note: the comparison favors our codec on familiar ground. It was trained on Kokoro voices like the test voice,
  while EnCodec and Mimi are general-purpose (14.9M and 79.3M parameters, trained
  on thousands of hours of varied audio).

**Clips** (`out/clip_*.wav`, test sentence 0: "Each channel lights up where the shape is."): original, ours with 1, 2 and 8 codebooks, EnCodec at
1.5 and 6 kbit/s, Mimi with 8 codebooks. The video plays original, ours 1, ours 8, EnCodec 6, Mimi 8.
`out/mel_*.png`: log-mel spectrograms of four of them.

**Speech as tokens.** The 2.1-second sentence is 9 GPT-2 text tokens, but
157 × 8 = 1,256 tokens for our codec or EnCodec at 6 kbit/s (75 frames/s), and
27 × 8 = 216 for Mimi (12.5 frames/s, codebooks of 2,048). `tokens.json` holds our
codec's 8 × 157 tokens. Speech language models predict such tokens: VALL-E (2023) uses EnCodec tokens,
Moshi (2024) uses Mimi's.
