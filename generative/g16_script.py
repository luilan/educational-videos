"""How Generative Models Work: Images and Sound, episode 16 — Audio Tokens: Neural Codecs.

Every number comes from code/g16_neural_codecs (make_data.py, codec.py, evaluate.py, 2026-11-25): a tiny codec
trained on 30 minutes of Kokoro speech, scored on 40 held-out sentences against Meta's EnCodec and Kyutai's Mimi.
The video plays five clips from out/.
"""
TITLE = "Audio Tokens: Neural Codecs"
TAGLINE = "Turning sound into tokens a model can predict."
NEXT = "Text to Speech: Inside Our Narrator"
SECTIONS = [
    # 1. Hook
    "A language model writes tokens. To make speech the same way, sound must become tokens too. But one second "
    "of our narrator is 24,000 numbers. A neural codec turns them into a few hundred tokens.",
    # 2. Encoder
    "Our tiny codec starts with an encoder: convolutions with strides of 2, 4, 5 and 8. Together they turn every "
    "320 samples into one vector of 64 numbers: 75 vectors a second.",
    # 3. Vector quantization
    "Each vector is then replaced by the nearest of 1,024 learned vectors, the codebook. Only its number is kept: "
    "a token worth 10 bits.",
    # 4. Residual
    "One codebook captures 95 percent of the vector. A second codebook quantizes what the first missed, the "
    "residual. A third, what is still left. After eight, a tenth of a percent remains. Eight tokens per frame: "
    "600 tokens a second, 6 kilobits.",
    # 5. Decoder and training
    "A mirror-image decoder turns the chosen vectors back into sound. Training compares spectrograms of input and "
    "output. Each codebook vector drifts toward the vectors assigned to it, and randomly using fewer codebooks "
    "teaches one model every bitrate. We trained it for two and a half hours on this CPU.",
    # 6. Results
    "On 40 sentences it never heard, how many words does Whisper hear differently? With one codebook, 4 in 100. "
    "With two, just 1.5.",
    # 7. Listen
    "Listen: the original, then one codebook, then eight.",
    # 8. Bottleneck
    "Beyond four codebooks, nothing improves. Even with no rounding at all, the spectrogram distance stays at "
    "0.81. Our small decoder is the limit, not the bits.",
    # 9. Real codecs
    "Real codecs are bigger, and trained with judges, the GAN game of episode 4. Meta's EnCodec, at 6 kilobits, "
    "changes fewer than 1 word in 100. Kyutai's Mimi needs just 1.1 kilobits. Yet ours scores a lower spectrogram "
    "distance than both, because that is exactly what it was trained on. Listen and decide.",
    # 10. Speech as language
    "Now speech is a sequence of tokens. Our two-second sentence is 9 text tokens, but 1,256 EnCodec tokens, or "
    "216 for Mimi. A transformer that predicts them one by one is a speech language model. Vall-E and Moshi work "
    "this way.",
    # 11. Code
    "In code: encode, quantize residual by residual, then decode the sum.",
    # 12. Next
    "Next: text to speech, inside our narrator.",
]
