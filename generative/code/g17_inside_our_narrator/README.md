# How Generative Models Work, episode 17 · Text to Speech: Inside Our Narrator

Kokoro v1.0 (`hexgrad/Kokoro-82M`), the text-to-speech model whose `af_heart` voice narrates this series, run one
stage at a time with every intermediate prediction kept, then edited before the decoder.

```bash
pip install -r requirements.txt
python3 inside_kokoro.py      # ~3 minutes → results.json, curves.json, out/*.wav
```

`inside_kokoro.py` loads the PyTorch weights from Hugging Face and repeats what `kokoro.model.KModel.forward_with_tokens`
does, stage by stage (function `synth`). Phonemes come from espeak-ng through `kokoro-onnx`, the same front end that
voices this series (`KOKORO_DIR`, default `../../../tts`, must hold `kokoro-v1.0.onnx` and `voices-v1.0.bin`).
`kokoro.model` is imported without `kokoro/__init__.py`, so the English G2P (misaki, spaCy) is not needed.
The vocoder's noise source is random; `synth` seeds it, so every run gives the same numbers.

Sentence: "Our narrator turns text into speech in three steps."

## Results (2026-11-25)

**Parameters:** 81,810,022 in total. PL-BERT (phoneme BERT, ALBERT-style shared layers) 6,292,480 + its projection
393,728; prosody predictor (durations, pitch, energy) 16,203,828; text encoder 5,606,400; decoder 53,313,586 (65%),
of which the final generator (iSTFTNet) is 19,685,942.

**Phonemes.** The 51 characters become 58 symbols, stress marks included:
`ˌaʊɚ nˈæɹeɪɾɚ tˈɜːnz tˈɛkst ˌɪntʊ spˈiːtʃ ɪn θɹˈiː stˈɛps.` The vocabulary has 114 symbols (178 slots).
Spelling is a poor guide: though `ðˌoʊ`, tough `tˈʌf`, through `θɹˈuː`, thought `θˈɔːt`.

**The voice.** `voices/af_heart.pt` is a table of 510 × 256 numbers: one style vector per input length (Kokoro picks
row 57 for 58 symbols). The first 128 numbers condition the decoder (how the voice sounds), the last 128 the prosody
predictor (how it speaks). Rows for different lengths differ (cosine similarity 0.45 between rows 10 and 200). The
ONNX voices file holds exactly the same numbers.

**Durations.** The predictor gives every symbol a whole number of 25 ms frames (600 samples): 35 of the 58 symbols
get 2 frames, 13 get 1; the final "s" gets 13. With the silences at both ends, 141 frames = 3.525 s = 84,600 samples.
Word start times follow from the durations (`results.json` → `words`); episode 18 uses them as ground truth for
Whisper's timestamps.

**Pitch and energy.** Two curves at 80 values per second (282 values): predicted pitch median 190 Hz (voiced 68% of
the time, up to 311 Hz), and energy.

**Decoder.** Frame features, pitch and energy go through style-conditioned blocks (one upsamples × 2), then the
generator: a source of sine waves at the predicted pitch and its first 8 harmonics, upsampling × 10 and × 6, and a
final layer that predicts the magnitude and phase of 11 frequencies (FFT size 20) every 5 samples; an inverse STFT
gives the waveform: 40 frames/s × 2 × 10 × 6 × 5 = 24,000 samples per second.

**Same model as the narrator.** The PyTorch output matches kokoro-onnx's (which trims the silence at both ends) with
correlation 0.990 once aligned.

**Edits before the decoder.** Pitch measured from the audio by autocorrelation (as in episode 15), words by
Whisper small.en:

| version | length | measured pitch (median) | pitch spread (10–90%) | Whisper |
|---|---|---|---|---|
| original | 3.53 s | 204 Hz | 82 Hz | no errors |
| flat pitch (median where voiced) | 3.53 s | 190 Hz | 8 Hz | no errors |
| pitch × 1.5 | 3.53 s | 282 Hz | 100 Hz | no errors |
| every symbol the mean duration (2 frames) | 3.25 s | 203 Hz | 91 Hz | no errors |
| speed 1.4 (durations ÷ 1.4) | 2.35 s | 203 Hz | 110 Hz | "A narrator turns texting a speech in three steps." (33%) |
| am_michael's sound half, our prosody half | 3.53 s | 200 Hz | 87 Hz | no errors |
| am_michael's whole style vector | 3.95 s | 120 Hz | 61 Hz | no errors |

Swapping only the first half changes the timbre but keeps our pitch and rhythm; the second half carries them (the
whole male style gives 120 Hz and a different length). The video plays `flat_pitch.wav`, `higher_pitch.wav` and
`other_voice_sound.wav`.

A question ("Can you hear my voice go up at the end of a question?", `out/question.wav`) barely rises: median
190 Hz, at most 205 Hz in its last quarter.

**Speed** (8-core Ryzen 7 3700X, 8 threads, no GPU): the 3.525 s sentence takes 0.87 s, 4.1× faster than real time. PL-BERT 40 ms, prosody predictor 29 ms, text encoder 6 ms, decoder 794 ms (91%).
