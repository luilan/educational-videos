# How Generative Models Work, episode 18 · Speech to Text: Inside Whisper

Whisper, the speech recognizer that times every animation in this series, opened up: the log-mel input, the
encoder, greedy decoding token by token, and word timestamps from cross-attention and dynamic time warping. Then
how accurate those timestamps are (Kokoro tells us the truth), how many words it gets wrong on this series'
narration, and what noise and silence do.

```bash
pip install -r requirements.txt
python3 whisper_inside.py                 # ~25 minutes on a CPU → results.json, attention.json, out/*
EXAMPLE_ONLY=1 python3 whisper_inside.py  # ~1 minute: only the step-by-step example (results_example.json)
```

Needs Kokoro's model files in `KOKORO_DIR` (default `../../../tts`) for the phonemizer, Kokoro's PyTorch weights
and Whisper from Hugging Face, and the series' narration in `../../voice/` (made by `./narrate.sh`).

## 1. One sentence, step by step (`openai/whisper-small.en`, Hugging Face transformers)

Sentence (spoken by Kokoro's af_heart): "Whisper listens to thirty seconds of sound and writes down every word it
hears." (5.08 s).

- **Model:** 241,734,144 parameters: encoder 88,154,112, decoder 153,580,032
  (its token embedding alone is 51,864 × 768). 12 encoder and 12 decoder
  layers, width 768, 12 heads.
- **Input:** audio resampled to 16,000 samples/s, padded to 30 s, turned into an 80 × 3,000
  log-mel spectrogram (25 ms windows every 10 ms). The sentence fills the first 508 columns
  (`out/input_30s.png`, `out/input_sound.png`).
- **Encoder:** two convolutions (kernel 3; the second with stride 2) give 1,500 vectors of
  768, one per 20 ms, then 12 transformer layers.
- **Decoder:** starts from `<|startoftranscript|><|notimestamps|>` and picks the most likely token each step
  (greedy): “ Whis” 81%, “per” 99%, “ listens” 98%, “ to” 100%, “ 30” 85%, “ seconds” 100% … For “30” the runner-up is “ thirty” at 14%. It stops with
  `<|endoftext|>`. Text: “Whisper listens to 30 seconds of sound and writes down every word it hears.”.
- **Time tokens:** without `<|notimestamps|>`, the model writes them itself (20 ms steps):
  `<|0.00|> Whisper listens to 30 seconds of sound and writes down every word it hears.<|4.64|>`

## 2. Word timestamps

Whisper's word times are not trained outputs. `inside()` recomputes them the way openai/whisper's `timing.py` does:

1. Decode the text, then run the decoder once more over all tokens, keeping the **cross-attention** weights.
2. Keep the 19 **alignment heads** listed in the model's generation config (heads whose attention follows the
   audio), over the frames that hold sound (50 per second).
3. Normalize each head over the tokens, smooth along time (median filter, width 7), average the heads.
4. Row *t* of the attention is the step that **predicts** token *t* + 1, so the rows that produced the text are
   rows 1 … n of the decoder input (after `<|startoftranscript|>`). Using the rows of the tokens themselves puts
   every word one token late.
5. **Dynamic time warping** (`dtw`): the cheapest path through −attention from the first token and frame to the
   last, moving only down, right or diagonally. A word starts at the frame where the path first reaches its first
   token.

`attention.json` holds the averaged map (scaled to 0–1) and the path, as drawn in the video.

Example sentence, our DTW against Kokoro's true word starts (ms): -350, -105, -230, -65, -115, -140, -60, -445, -90, -175, -205, -125, -130, -70.
The first word always gets time 0 (the path starts at the first frame).

## 3. How exact are the timestamps?

Kokoro's predicted durations say where every word starts (25 ms frames; they match the audio onsets within about
30 ms). 40 sentences from this series' scripts (no digits), spoken by Kokoro, transcribed by faster-whisper small.en
with `word_timestamps=True`, exactly as `framework/voice.py` does for the animation cues; 275 words could be
paired with the script.

- Median error 115 ms; 90% of words within 325 ms; 43% within
  100 ms, 82% within 200 ms.
- Almost always **early**: 98% of the words, mean -141 ms. Histogram (50 ms bins from
  −400 to +400 ms): [12, 23, 3, 12, 42, 65, 89, 23, 3, 3, 0, 0, 0, 0, 0, 0].
- So every cue in this series fires about a tenth of a second before its word is heard.

## 4. Word errors, noise, silence (faster-whisper, beam 5)

Word error rate with numbers spelled out on both sides ("10" = "ten").

- **This series' narration:** 196 narration sections of 18 episodes, 6,955 words:
  small.en 2.3%, tiny.en 3.5%. Most frequent
  differences for small.en (said → heard): and → nothing (×15), u net → unit (×11), a → one (×7), autoencoders → auto encoders (×4), nothing → one (×4), u net → unet (×4), in code → encode (×4), point → nothing (×4). Many are spelling choices (“autoencoders” → “auto encoders”,
  “a thousand” → “1,000”, “one and a half” → “1.5”), not mishearing. The narration of episode 18 is the draft version.
- **Noise:** 20 of the sentences with white noise at falling signal-to-noise ratios: clean 1.1%,
  20 dB 1.1%, 10 dB 1.6%, 5 dB 1.1%, 0 dB 6.0% (noise as loud as
  the voice; `out/noise_0.wav` is played in the video), −5 dB 19.1%.
- **Silence:** 10 s of digital silence → “You” (the segment's own no-speech probability
  0.92). With faster-whisper's voice activity filter (`vad_filter=True`) nothing is
  written. 10 s of white noise → nothing, with or without the filter.
