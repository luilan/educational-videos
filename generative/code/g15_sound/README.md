# How Generative Models Work, episode 15 · Sound as Data

One sentence spoken by Kokoro's `af_heart` voice, the voice that narrates this series, examined as data: the
waveform, the sample rate, the spectrogram, the mel scale, and what a spectrogram throws away.

```bash
pip install -r requirements.txt
python3 sound.py            # ~2 minutes → results.json, waveform.json, mel_edges.json, out/*.png, out/*.wav
```

Needs Kokoro's model files `kokoro-v1.0.onnx` and `voices-v1.0.bin` in `KOKORO_DIR` (default `../../../tts`). Whisper
`small.en` (through faster-whisper) is downloaded on first use.

Sentence: "Sound is a long list of numbers. Our narrator makes thousands of them every second, and a model can learn
to make them too."

## Results (2026-11-22)

**Waveform.** 24,000 samples per second, 172,544 samples, 7.19 seconds, values between −0.38 and 0.58. Stored as
16-bit numbers, that is 384 kbit/s.

**Pitch.** Measured by autocorrelation on 40 ms frames: 390 of the 715 frames are voiced, with a median of 200 Hz
(10th–90th percentile 171–258 Hz). The 30 ms close-up of a vowel (`waveform.json`, from 5.345 s) repeats every
5.0 ms.

**Sample rates.**
- At 24,000 samples/s the highest frequency that can be captured is 12,000 Hz (the Nyquist limit, half the sample
  rate).
- 5.9% of the sentence's energy lies above 4,000 Hz, which an 8,000 samples/s version loses, and 3.2% lies above
  8,000 Hz, which a 16,000 samples/s version loses.
- Aliasing: a 5,000 Hz tone sampled 8,000 times per second without a filter has its spectral peak at exactly
  3,000 Hz.
- `resample_poly` filters before it drops samples. `out/sentence_8k.wav` and `out/sentence_16k.wav` are those
  versions, resampled back to 24 kHz for listening.

**Spectrogram** (`out/spectrogram.png`, 0–12,000 Hz, low frequencies at the bottom, an 80 dB range):
- Settings: 25 ms Hann windows (600 samples) every 10 ms (240), FFT size 1,024. That gives 513 frequency bins
  23.4 Hz apart and 100 frames per second; 717 frames × 513 bins = 367,821 numbers.
- Harmonics (multiples of the pitch) are the stripes at the bottom. Sibilants ("s", "z") are the broad bright
  columns.

**Mel** (`out/mel.png`):
- 80 triangular filters equally spaced on the mel scale, mel(f) = 2595·log10(1 + f/700), from 0 to 12,000 Hz.
  The first filter is 52 Hz wide and the last 877 Hz; 24 filters lie below 1,000 Hz and 17 above 6,000 Hz.
- The log of their energies is the log-mel spectrogram: 80 × 100 = 8,000 numbers per second, against 24,000
  samples.
- Whisper reads the same kind of input: 80 mel bands, 25 ms windows every 10 ms, computed from 16 kHz audio (so up
  to 8,000 Hz).

**Phase.** A spectrogram keeps each frequency's magnitude but not its phase. Each version below was transcribed by
Whisper small.en. "Mismatch" is the distance between the version's re-analyzed magnitudes and the original's, both
scaled to unit length (0 = identical).

| version | Whisper word errors | mismatch |
|---|---|---|
| original | 0% | 0 |
| magnitudes + random phases | 0% | 0.557 |
| Griffin-Lim, 60 rounds (Griffin & Lim 1984) | 0% | 0.099 |
| mel → magnitudes (pseudo-inverse) → Griffin-Lim | 0% | 0.184 |
| 16,000 samples/s | 0% | 0.178 |
| 8,000 samples/s | 0% | 0.243 |

The words survive every version, so they live in the magnitudes. What phase changes is how natural the sound is,
which a word count cannot measure. Listen to the clips of the first sentence (`out/clip_original.wav`,
`clip_8k.wav`, `clip_random_phase.wav`, `clip_griffin_lim_60.wav`); the video plays all four. Turning a spectrogram
back into natural sound is the job of a trained vocoder (episode 17).
