"""How Generative Models Work: Images and Sound, episode 18 — Speech to Text: Inside Whisper (series finale).

Every number comes from code/g18_inside_whisper (whisper_inside.py, 2026-11-25): openai/whisper-small.en, step by
step on one sentence spoken by Kokoro, then timestamps, word errors, noise and silence with faster-whisper small.en,
the model that times this series' animations. The video plays one clip from out/ (the sentence at 0 dB).
"""
TITLE = "Speech to Text: Inside Whisper"
TAGLINE = "Log-mel in, tokens out, and where each word's time comes from."
NEXT = None  # series finale
SECTIONS = [
    # 1. Hook
    "Every animation in this series is timed by a second model. Whisper listens to our narrator and writes down "
    "each word, with the moment it was spoken. In this last episode, we open it up.",
    # 2. Input
    "Whisper always hears 30 seconds at a time: sound at 16,000 samples a second, turned into a log-mel "
    "spectrogram, 80 bands and 3,000 columns. Our five-second sentence fills one sixth. The rest is padding.",
    # 3. Encoder
    "Two convolutions, the second with a stride of two, turn the columns into 1,500 vectors, one every 20 "
    "milliseconds. Then 12 transformer layers let every moment attend to every other.",
    # 4. Decoder
    "The decoder is a language model that also attends to the audio, through cross-attention. It starts with two "
    "special tokens: start of transcript, and no timestamps. Then it predicts one token at a time.",
    # 5. Greedy steps
    "Whis, per, listens: each is the most likely next token. At thirty it hesitates: 85 percent for the digits, "
    "14 for the word. An end-of-text token stops it.",
    # 6. Training and time tokens
    "Whisper was trained on 680,000 hours of transcribed audio from the web, simply to predict the next token. It "
    "can even write time tokens itself, in steps of 20 milliseconds.",
    # 7. Cross-attention
    "But where do the times of single words come from? While predicting a token, 19 of the decoder's attention "
    "heads look at the stretch of audio that token comes from. Averaged, they trace a diagonal.",
    # 8. DTW
    "Dynamic time warping finds the best path through this map that only moves forward. Where the path reaches "
    "a word gives its start time.",
    # 9. Accuracy
    "Our narrator knows the truth: Kokoro decided how long every sound lasts. Over 275 words, Whisper's start "
    "times were a median of 115 milliseconds off, almost always early. So every cue in this series fires about a "
    "tenth of a second before its word.",
    # 10. Word errors
    "And the words themselves? Across this whole series' narration, the small model gets about 2 words in 100 "
    "wrong, the tiny one about 3.5. The favorite mistake: U-Net, heard as unit.",
    # 11. Noise
    "Noise barely matters. Errors stay near 1 percent down to 5 decibels, and reach 6 percent at zero decibels, "
    "where the noise is as loud as the voice. Listen.",
    # 12. Silence
    "But give it ten seconds of silence, and it writes: you. Trained to always transcribe something, it "
    "hallucinates. A voice detector that skips silent stretches fixes it.",
    # 13. Code
    "In code: log-mel, encoder, decode token by token, then align with the attention.",
    # 14. Finale
    "That completes the series: from pixels and noise, to voices, and back to text. Thanks for watching.",
]
