"""Study guide content for How Generative Models Work: Images and Sound, episode 16: Audio Tokens: Neural Codecs.

Build:  python framework/study_guide.py generative g16 --video generative/media/videos/g16_scene/1080p60/CodecVideo.mp4
Every number comes from code/g16_neural_codecs (make_data.py, codec.py, evaluate.py, 2026-11-25).
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 16",
    "title": "Audio Tokens: Neural Codecs",
    "tagline": "Turning sound into tokens a model can predict.",
    "duration": "2:53",
    "intro": """<p>A language model predicts tokens. To generate speech the same way, sound has to become tokens first.
This lesson builds a tiny <b>neural audio codec</b> and trains it on a CPU: an encoder that turns 24,000 samples a
second into 75 vectors, <b>residual vector quantization</b> that turns each vector into a few integers (tokens), and
a decoder that turns them back into sound. Then it compares the codec with two real ones, Meta's <b>EnCodec</b> and
Kyutai's <b>Mimi</b>, and counts how many tokens a sentence of speech becomes.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 15 (samples, spectrograms), episode 2
(autoencoders) and episode 4 (GANs). Code: <code>code/g16_neural_codecs</code> (PyTorch; training takes about two
hours on a CPU, and the trained codec is included). The video plays five clips; they are in
<code>code/g16_neural_codecs/out</code>.</div>""",
}

CONCEPTS = [
    {
        "title": "From samples to tokens",
        "segment": (7.7, 45.5),
        "figures": [{"t": 21.1, "caption": "A language model writes tokens. One second of our narrator is 24,000 "
                                           "numbers; a codec turns it into a few hundred tokens."},
                    {"t": 35.6, "caption": "The encoder: strided convolutions (÷2, ÷4, ÷5, ÷8) turn every 320 "
                                           "samples into one vector of 64 numbers, 75 per second."},
                    {"t": 45.3, "caption": "Vector quantization: replace the vector by the nearest of 1,024 codebook "
                                           "vectors and keep only its number, a 10-bit token."}],
        "body": [
            """<p>An LLM generates text one token at a time from a vocabulary of a few tens of thousands. Speech as
raw samples is a poor fit: 24,000 numbers per second, each a continuous value. A <b>neural audio codec</b> is an
autoencoder whose bottleneck is made of <b>discrete tokens</b>, so that sound can be compressed, transmitted, and
generated like text.</p>""",
            "{fig0}",
            """<p>The <b>encoder</b> is a stack of convolutions. Four of them have strides 2, 4, 5 and 8, so their
product, 320, is the number of samples behind each output vector. At 24,000 samples per second that gives
<b>75 vectors per second</b> (one every 13.3 ms), each 64 numbers long. Our encoder has 849,504 parameters.</p>""",
            "{fig1}",
            """<p>A vector of 64 real numbers is not yet a token. <b>Vector quantization</b> replaces it by the nearest
of 1,024 learned vectors, the <b>codebook</b>, and keeps only that vector's index. An index from 0 to 1,023 is
log2(1,024) = 10 bits: one token.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>A codec is an autoencoder with a discrete bottleneck:
encode to a slow sequence of vectors, then replace each vector by the number of its nearest codebook entry.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "An encoder's strides are 2, 4, 6 and 8 on 24,000 samples per second. How many "
                                    "vectors per second does it produce?",
             "answer": "62.5.", "why": "2 · 4 · 6 · 8 = 384 samples per vector; 24,000 / 384 = 62.5.",
             "key": {"parts": [{"label": None, "value": 62.5, "tol": 0.1, "unit": "vectors/s"}]}},
            {"kind": "number", "q": "How many bits is one token from a codebook of 2,048 vectors?",
             "answer": "11 bits.", "why": "log2(2,048) = 11 (Mimi's codebooks).",
             "key": {"parts": [{"label": None, "value": 11, "tol": 0, "unit": "bits"}]}},
        ],
    },
    {
        "title": "Residual vector quantization",
        "segment": (45.68, 81.9),
        "figures": [{"t": 62.6, "caption": "Share of the vector left unexplained after 1, 2, … 8 codebooks: 5.3%, "
                                           "1.9%, 0.9% … 0.1%. Eight tokens per frame: 600 tokens/s, 6 kbit/s."},
                    {"t": 81.7, "caption": "A mirror-image decoder; training compares spectrograms of input and "
                                           "output. 1.76 million parameters, 12,000 steps, 30 min of speech."}],
        "body": [
            """<p>One codebook of 1,024 vectors is too coarse to describe speech well. Making it bigger does not
scale (a 20-bit codebook would need a million vectors). Instead, <b>residual vector quantization</b> (RVQ) chains
small codebooks: the first quantizes the vector, the second quantizes the <b>residual</b>, what the first missed,
the third what is still left, and so on. The decoder receives the sum of the chosen vectors.</p>""",
            "{fig0}",
            """<p>In our trained codec, the first codebook captures 95% of each vector (by squared length); after two,
1.9% is left; after eight, 0.1%. Each frame becomes <b>8 tokens</b>: 8 × 75 = 600 tokens per second, and
600 × 10 bits = <b>6 kbit/s</b>, 64 times less than the 384 kbit/s of 16-bit audio. Using fewer codebooks gives
lower bitrates: 0.75 kbit/s per codebook.</p>""",
            """<p>Training: the <b>decoder</b> mirrors the encoder with transposed convolutions. The loss compares
log-mel spectrograms of the input and the output at three window sizes (plus a small waveform term). Rounding to
the nearest vector has no gradient, so gradients skip it (the straight-through estimator). The codebook vectors
are not trained by gradients either: each one drifts toward the average of the encoder vectors assigned to it, and
unused ones restart at random encoder outputs. Finally, <b>quantizer dropout</b> (from SoundStream): each training
step uses a random number of codebooks, 1 to 8, so one model works at every bitrate.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Residual quantization stacks small codebooks, each one
encoding the error of the ones before it. More codebooks, more bits, more detail.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A codec uses 75 frames per second and 4 codebooks of 1,024 vectors. What is "
                                    "its bitrate, in kbit/s?",
             "answer": "3 kbit/s.", "why": "75 × 4 × 10 bits = 3,000 bit/s.",
             "key": {"parts": [{"label": None, "value": 3, "tol": 0, "unit": "kbit/s"}]}},
            {"kind": "mc", "q": "What does the second codebook in residual vector quantization encode?",
             "options": ["The same vector again, more precisely", "The difference between the vector and the first "
                         "codebook's choice", "The next frame", "The phase"],
             "answer": "B.", "why": "Each stage quantizes the residual left by the stages before it.",
             "key": {"choice": 1}},
            {"kind": "short", "q": "Why train with a random number of codebooks at each step?",
             "answer": "So the same model works at every bitrate, from 1 to 8 codebooks (quantizer dropout).",
             "why": "Without it, decoding with fewer codebooks than in training would fail."},
        ],
    },
    {
        "title": "How good is it?",
        "segment": (82.13, 142.8),
        "figures": [{"t": 91.3, "caption": "Words Whisper hears differently than in the original: 4.4% with one "
                                           "codebook, 1.5% with two."},
                    {"t": 114.8, "caption": "Beyond four codebooks nothing improves: 8 codebooks and no rounding at "
                                            "all both reach a mel distance of 0.81."},
                    {"t": 136.0, "caption": "Against real codecs: EnCodec (14.9M parameters) and Mimi (79.3M). Ours "
                                            "has the lowest mel distance, its own training loss."}],
        "body": [
            """<p>The test: 40 sentences from this series' scripts that the codec never heard, in the narrator's
voice. Whisper transcribes the original and each compressed version; <b>words changed</b> counts how many words of
its transcript differ (0 for the original). One codebook (0.75 kbit/s) changes 4.4% of the words; two codebooks
(1.5 kbit/s), 1.5%; four, 1.2%. The video plays the original, one codebook and eight.</p>""",
            "{fig0}",
            """<p>Past four codebooks nothing improves. The decisive check is to skip quantization entirely and decode
the encoder's vectors directly: the mel distance is 0.809, the same as with 8 codebooks (0.810). The rounding no
longer costs anything; the small decoder, trained only to match spectrograms, is the limit.</p>""",
            "{fig1}",
            """<p>Real codecs are larger and trained on thousands of hours with <b>discriminators</b>, the judges of
episode 4. EnCodec at 6 kbit/s changes 0.8% of the words; Mimi (12.5 frames per second, 8 codebooks of 2,048)
changes 0.7% at only 1.1 kbit/s. Yet our codec has the lowest mel distance of the three (0.810 against 0.831 and
1.220), because that is exactly the number it was trained to minimize. A metric is not a listener: GAN-trained
codecs produce sound that is natural rather than spectrogram-exact. (Also note the comparison favors ours: it was
trained on voices like the test voice; the others are general-purpose.)</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>A few kilobits per second keep speech intelligible. Beyond
that, quality depends on the decoder and on how it is trained, not on the number of bits.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Because our codec has a lower mel distance than Mimi, it sounds better.”",
             "answer": "False.", "why": "Mel distance is our codec's training loss. Mimi is trained with "
                                        "discriminators to sound real; listen to the clips.",
             "key": {"value": False}},
            {"kind": "short", "q": "How does the video show that the decoder, not the quantizer, limits our codec?",
             "answer": "Decoding the encoder's vectors with no quantization at all gives the same mel distance (0.809) "
                       "as 8 codebooks (0.810).",
             "why": "If rounding were the problem, removing it would help."},
        ],
    },
    {
        "title": "Speech as a language",
        "segment": (142.97, 172.6),
        "figures": [{"t": 161.0, "caption": "A 2.1-second sentence: 9 text tokens, 1,256 EnCodec tokens (6 kbit/s), "
                                            "216 Mimi tokens."},
                    {"t": 166.2, "caption": "Residual vector quantization in code."}],
        "body": [
            """<p>Once sound is tokens, generating speech is a language-modeling problem. The test sentence “Each
channel lights up where the shape is.” lasts 2.1 seconds. It is 9 GPT-2 text tokens, but 157 frames × 8 codebooks
= 1,256 tokens for EnCodec at 6 kbit/s (or for our codec), and 27 × 8 = 216 for Mimi, whose lower frame rate is
designed for exactly this. A transformer that predicts these tokens one by one, then a codec decoder that turns
them into sound, is a <b>speech language model</b>. VALL-E (2023) generates EnCodec tokens; Moshi (2024)
generates Mimi tokens.</p>""",
            "{fig0}",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Codec tokens turn speech generation into next-token
prediction. Fewer tokens per second make it cheaper, which is why newer codecs lower the frame rate.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many tokens are 10 seconds of speech for Mimi with 8 codebooks at 12.5 "
                                    "frames per second?",
             "answer": "1,000.", "why": "10 × 12.5 × 8 = 1,000 (EnCodec at 6 kbit/s: 10 × 75 × 8 = 6,000).",
             "key": {"parts": [{"label": None, "value": 1000, "tol": 0, "unit": "tokens"}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Encode a sentence with the trained codec and decode it "
                                  "with fewer codebooks.",
             "code": """model = Codec(); model.load_state_dict(torch.load("out/codec.pt")); model.eval()
tokens = model.encode(x, 8)          # (8, frames) integers
y2 = model.decode(tokens[:2])        # only the first two codebooks: 1.5 kbit/s""",
             "answer": "codec.py's Codec.encode and Codec.decode; evaluate.py does this for every bitrate.",
             "why": "Listen to y2 against the original: the words survive at 1.5 kbit/s."},
        ],
    },
]
