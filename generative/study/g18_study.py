"""Study guide content for How Generative Models Work: Images and Sound, episode 18: Speech to Text: Inside Whisper.

Build:  python framework/study_guide.py generative g18 --video generative/media/videos/g18_scene/1080p60/WhisperVideo.mp4
Every number comes from code/g18_inside_whisper (whisper_inside.py, 2026-11-25).
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 18 (series finale)",
    "title": "Speech to Text: Inside Whisper",
    "tagline": "Log-mel in, tokens out, and where each word's time comes from.",
    "duration": "3:02",
    "intro": """<p>Every animation in this series is timed by Whisper, a speech recognizer that writes down each word
of the narration together with the moment it is spoken. This lesson opens it up on one sentence: the 30-second
<b>log-mel</b> input, the <b>encoder</b>, the <b>decoder</b> that writes tokens one at a time, and how <b>word
timestamps</b> come out of cross-attention and dynamic time warping. Then it measures Whisper on this series'
own narration: how early its timestamps are, how many words it gets wrong, and what noise and silence do.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 15 (log-mel spectrograms) and episode 17
(Kokoro, the narrator, whose durations give the true word times here); How LLMs Work for transformers and
attention. Code: <code>code/g18_inside_whisper</code> (transformers, faster-whisper, Kokoro; about 25 minutes on a
CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "From sound to tokens",
        "segment": (7.7, 73.1),
        "figures": [{"t": 34.4, "caption": "Whisper always hears 30 seconds: 80 mel bands × 3,000 columns. Our "
                                           "5.1-second sentence fills one sixth; the rest is padding."},
                    {"t": 48.0, "caption": "Two convolutions (the second with stride 2) give 1,500 vectors, one per "
                                           "20 ms; then 12 transformer layers."},
                    {"t": 61.2, "caption": "The decoder is a language model with cross-attention to the audio. It "
                                           "starts from two special tokens."},
                    {"t": 72.9, "caption": "Greedy decoding: “ 30” at 85%, “ thirty” at 14%. An end-of-text token "
                                           "stops it."}],
        "body": [
            """<p>Whisper (OpenAI, 2022) is an <b>encoder-decoder transformer</b>. The English-only small model used
here has 242 million parameters: 88 million in the encoder and 154 million in the decoder, of which 40 million are
its token embedding (51,864 tokens × 768).</p>
<p>Its input is always <b>30 seconds</b> of audio at 16,000 samples per second, turned into a <b>log-mel
spectrogram</b> (episode 15): 80 bands, 25 ms windows every 10 ms, 3,000 columns. Shorter audio is padded; our
5.1-second sentence fills the first 508 columns.</p>""",
            "{fig0}",
            """<p>The <b>encoder</b> starts with two convolutions over time (kernel 3); the second has stride 2, so
3,000 columns become <b>1,500 vectors</b> of 768 numbers, one per 20 ms. Twelve transformer layers then let every
moment attend to every other.</p>""",
            "{fig1}",
            """<p>The <b>decoder</b> is a language model like GPT, with one addition: <b>cross-attention</b>. In every
layer, its tokens also attend to the encoder's 1,500 vectors. Decoding starts from special tokens,
<code>&lt;|startoftranscript|&gt;</code> and here <code>&lt;|notimestamps|&gt;</code>, then predicts one text token
at a time.</p>""",
            "{fig2}",
            """<p>With <b>greedy</b> decoding the most likely token wins each step: “ Whis” (81%), “per” (99%),
“ listens” (98%) … At “thirty” the model hesitates between the digits “ 30” (85%) and the word “ thirty” (14%):
both are correct transcripts, and its training data mostly wrote digits. <code>&lt;|endoftext|&gt;</code> ends the
transcript.</p>""",
            "{fig3}",
            """<div class="box key"><b class="t">Key idea</b>Whisper is a language model that listens: an encoder turns
30 seconds of log-mel into 1,500 vectors, and a decoder writes the transcript token by token while attending to
them.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many encoder vectors does Whisper produce for its 30-second window, and how "
                                    "many milliseconds does each cover?",
             "answer": "1,500 vectors, 20 ms each.", "why": "3,000 columns at 10 ms, halved by the stride-2 "
                                                           "convolution.",
             "key": {"parts": [{"label": "vectors", "value": 1500, "tol": 0, "unit": ""},
                               {"label": "ms", "value": 20, "tol": 0, "unit": "ms"}]}},
            {"kind": "mc", "q": "What lets Whisper's decoder use the audio?",
             "options": ["Its token embedding", "Cross-attention to the encoder's vectors", "The time tokens",
                         "Greedy decoding"],
             "answer": "B.", "why": "Every decoder layer attends to the encoder output; otherwise it would be a "
                                    "plain language model.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "Where word times come from",
        "segment": (73.35, 106.9),
        "figures": [{"t": 85.6, "caption": "Trained on 680,000 hours to predict the next token, Whisper can also "
                                           "write time tokens itself, in 20 ms steps."},
                    {"t": 98.1, "caption": "19 alignment heads, averaged: each token's attention lands on the stretch "
                                           "of audio it comes from, tracing a diagonal."},
                    {"t": 106.7, "caption": "Dynamic time warping: the best path that only moves forward. Where it "
                                            "reaches a word is the word's start."}],
        "body": [
            """<p>Whisper was trained on 680,000 hours of transcribed audio from the web, with the same objective as
an LLM: predict the next token. Its vocabulary includes <b>time tokens</b> (<code>&lt;|0.00|&gt;</code>,
<code>&lt;|0.02|&gt;</code> … every 20 ms), so it can mark where segments start and end: here
<code>&lt;|0.00|&gt;</code> … <code>&lt;|4.64|&gt;</code>. But time tokens only mark segments, not single words.</p>""",
            "{fig0}",
            """<p>Word times come from inside the model. While the decoder predicts a token, some of its
cross-attention heads look at the audio that token comes from. The model's configuration lists 19 such
<b>alignment heads</b> for small.en. Normalized, smoothed and averaged, their attention forms a map of tokens ×
20 ms frames with a bright diagonal. One subtlety: the attention row at position <i>t</i> belongs to the step
that <i>predicts</i> token <i>t</i> + 1, so the rows must be shifted by one (forgetting this makes every word one
token late).</p>""",
            "{fig1}",
            """<p><b>Dynamic time warping</b> (DTW) finds the cheapest path through this map from the first token and
frame to the last, moving only forward: down, right or diagonally. The frame where the path first reaches a word's
first token is that word's start time. faster-whisper does exactly this when asked for
<code>word_timestamps=True</code>, and that is how this series' narration gets its animation cues.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Word timestamps are not predicted directly; they are read
off the cross-attention of a few heads by finding a monotonic path with dynamic time warping.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Whisper was trained to output the start time of every word.”",
             "answer": "False.", "why": "It was trained on segment-level time tokens; word times come from its "
                                        "cross-attention plus DTW after decoding.",
             "key": {"value": False}},
            {"kind": "short", "q": "Why may the DTW path only move down, right or diagonally?",
             "answer": "Speech and text go forward together: a later word cannot start before an earlier one, so the "
                       "path must be monotonic in both tokens and time.",
             "why": "That constraint is what makes the alignment robust to noisy attention."},
        ],
    },
    {
        "title": "How good is it?",
        "segment": (107.13, 166.2),
        "figures": [{"t": 125.0, "caption": "Whisper's word starts minus Kokoro's true starts: median error 115 ms, "
                                            "98% early."},
                    {"t": 138.4, "caption": "On this series' narration: small.en 2.3% of words wrong, tiny.en 3.5%. "
                                            "Favorite mistake: U-Net heard as “unit”."},
                    {"t": 155.6, "caption": "White noise: about 1% errors down to 5 dB, 6% at 0 dB (noise as loud as "
                                            "the voice), 19% at −5 dB."},
                    {"t": 166.0, "caption": "Ten seconds of silence: Whisper writes “You”. A voice detector that skips "
                                            "silence fixes it."}],
        "body": [
            """<p>How exact are the word times? Our narrator knows the truth: Kokoro's predicted durations (episode
17) say exactly where each word starts. On 40 sentences, 275 words could be paired. faster-whisper's start times
had a <b>median error of 115 ms</b>, 82% were within 200 ms, and 98% were <b>early</b> (mean −141 ms). So every cue
in this series fires about a tenth of a second before its word, which suits animations well.</p>""",
            "{fig0}",
            """<p>How many words does it get wrong? Across 196 narration sections from 18 episodes (6,955 words),
with numbers spelled out on both sides, small.en gets <b>2.3%</b> of the words wrong and tiny.en (39 million
parameters) 3.5%. Many differences are spelling choices (“autoencoders” → “auto encoders”, “a thousand” →
“1,000”). The favorite real mistake: “U-Net” heard as “unit”. And “In code:”, which ends many episodes, is often
heard as “encode”.</p>""",
            "{fig1}",
            """<p>Noise barely matters: with white noise added to 20 sentences, errors stay near 1% down to a
signal-to-noise ratio of 5 dB and reach 6.0% at 0 dB, where the noise is as loud as the voice (the video plays it),
then 19% at −5 dB.</p>""",
            "{fig2}",
            """<p>Silence is harder than noise. Given ten seconds of digital silence, Whisper writes “You”, even
though it estimates a 92% probability that there is no speech. Trained to always transcribe something, it
<b>hallucinates</b>. A voice activity detector that removes silent stretches first (faster-whisper's
<code>vad_filter</code>) fixes it.</p>""",
            "{fig3}",
            """<div class="box key"><b class="t">Key idea</b>Whisper is accurate on clear speech and robust to noise,
but its timestamps run about 0.1 s early, and with nothing to hear it may invent words.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A word's true start is 2.40 s. Whisper is 115 ms early. What start time does it "
                                    "report, in seconds?",
             "answer": "2.285 s (about 2.28–2.29).", "why": "2.40 − 0.115.",
             "key": {"parts": [{"label": None, "value": 2.285, "tol": 0.01, "unit": "s"}]}},
            {"kind": "number", "q": "A signal-to-noise ratio of 0 dB means the noise has what fraction of the "
                                    "voice's power?",
             "answer": "1 (the same power).", "why": "dB = 10·log10(P_signal / P_noise); 0 dB means the ratio is 1.",
             "key": {"parts": [{"label": None, "value": 1, "tol": 0, "unit": ""}]}},
            {"kind": "short", "q": "Why does Whisper write “You” for pure silence?",
             "answer": "It is trained to write a transcript for audio, and the decoder always picks the most "
                       "likely next token; with no speech, a short common word can still be the most likely start. "
                       "Nothing in decoding forces it to write nothing.",
             "why": "Its own no-speech probability (0.92) knows better; a voice detector uses that kind of signal."},
        ],
    },
    {
        "title": "In code, and the whole series",
        "segment": (166.43, 181.9),
        "figures": [{"t": 171.8, "caption": "Whisper in six lines: log-mel, encoder, greedy decoding, alignment."},
                    {"t": 179.5, "caption": "The series: learning to generate, diffusion, text to image, sound."}],
        "body": [
            "{fig0}",
            """<p>This episode closes the series. Part 1 learned what generating means (autoencoders, VAEs, GANs).
Part 2 built diffusion step by step (noise, denoisers, sampling, flow matching). Part 3 turned it into text to
image (conditioning, guidance, CLIP, latent diffusion, transformers). Part 4 did sound: spectrograms, codecs that
make sound into tokens, the narrator that makes speech, and Whisper, which turns it back into text.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Generation and recognition use the same building blocks:
spectrograms, tokens, transformers and attention.</div>""",
        ],
        "exercises": [
            {"kind": "code", "q": "<b>Try it yourself.</b> Transcribe a file with word timestamps, the way this "
                                  "series times its animations.",
             "code": """from faster_whisper import WhisperModel
w = WhisperModel("small.en", device="cpu", compute_type="int8")
segments, _ = w.transcribe("example.wav", word_timestamps=True)
for s in segments:
    print([(x.word, round(x.start, 2)) for x in s.words])""",
             "answer": "whisper_inside.py's inside() does the same from scratch: greedy decoding, alignment heads, "
                       "DTW.",
             "why": "Compare its word starts with out/example.wav's true starts in results.json."},
        ],
    },
]
