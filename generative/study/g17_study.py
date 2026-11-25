"""Study guide content for How Generative Models Work: Images and Sound, episode 17: Text to Speech: Inside Our Narrator.

Build:  python framework/study_guide.py generative g17 --video generative/media/videos/g17_scene/1080p60/NarratorVideo.mp4
Every number comes from code/g17_inside_our_narrator (inside_kokoro.py, 2026-11-25): Kokoro v1.0, run stage by stage.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 17",
    "title": "Text to Speech: Inside Our Narrator",
    "tagline": "Phonemes, a style vector, durations, pitch, and a vocoder.",
    "duration": "2:55",
    "intro": """<p>The voice narrating this series is Kokoro, an open text-to-speech model with 82 million parameters.
This lesson runs it one stage at a time on one sentence and keeps every intermediate prediction: the
<b>phonemes</b>, the voice's <b>style vector</b>, the <b>duration</b> of each phoneme, the <b>pitch</b> and
<b>energy</b> curves, and the <b>vocoder</b> that turns them into sound. Because the predictions are just numbers,
we can also edit them before the vocoder and listen to the result.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 15 (sound as data, spectrograms and phase) and
episode 4 (GANs). Code: <code>code/g17_inside_our_narrator</code> (PyTorch, the Kokoro weights from Hugging Face,
Whisper; about 3 minutes on a CPU). The video plays three edited versions of the sentence; they are in
<code>code/g17_inside_our_narrator/out</code>.</div>""",
}

CONCEPTS = [
    {
        "title": "Phonemes and a voice",
        "segment": (7.7, 46.0),
        "figures": [{"t": 18.6, "caption": "Kokoro v1.0: 81.8 million parameters. The decoder alone holds 53.3 million."},
                    {"t": 34.3, "caption": "Spelling is a poor guide to sound. The sentence's 51 characters become "
                                           "58 phoneme symbols, including stress marks."},
                    {"t": 45.8, "caption": "A voice is a style vector of 256 numbers: half for how it sounds, half "
                                           "for how it speaks."}],
        "body": [
            """<p>Kokoro is built in stages. A small <b>phoneme BERT</b> (6.7 million parameters with its projection)
reads the input in context, a <b>prosody predictor</b> (16.2 million) decides timing, pitch and loudness, a
<b>text encoder</b> (5.6 million) describes what is said, and a <b>decoder</b> (53.3 million, 65% of the model)
makes the sound.</p>""",
            "{fig0}",
            """<p>The input is not letters but <b>phonemes</b>, symbols for sounds, produced by a rule-based front end
(espeak-ng here). English spelling is a poor guide: <i>though</i>, <i>tough</i>, <i>through</i> and <i>thought</i>
share the letters “ough” but sound nothing alike (ðˌoʊ, tˈʌf, θɹˈuː, θˈɔːt). Our sentence, “Our narrator turns
text into speech in three steps.”, has 51 characters and becomes 58 symbols, including the stress marks ˈ and ˌ.
Kokoro knows 114 symbols.</p>""",
            "{fig1}",
            """<p>A <b>voice</b> is not a recording but a <b>style vector</b> of 256 numbers. The first 128 go to the
decoder and shape <i>how the voice sounds</i> (its timbre). The last 128 go to the prosody predictor and shape
<i>how it speaks</i>: rhythm and melody. (Kokoro actually stores 510 such vectors per voice, one for each input
length; for 58 symbols it uses row 57.)</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Text to speech starts from phonemes, not letters, and a
voice is a vector: one half for how it sounds, one half for how it speaks.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why does Kokoro read phonemes instead of letters?",
             "options": ["Phonemes are shorter", "Spelling does not tell you how a word sounds", "Letters cannot be "
                         "embedded", "Phonemes need no context"],
             "answer": "B.", "why": "“ough” sounds different in though, tough, through and thought; phonemes "
                                    "spell sounds directly.",
             "key": {"choice": 1}},
            {"kind": "short", "q": "Which half of the style vector would you change to make the voice speak with a "
                                   "different rhythm, and which to change its timbre?",
             "answer": "Rhythm and melody: the second half (128 numbers, to the prosody predictor). Timbre: the "
                       "first half (to the decoder).",
             "why": "The video swaps only the first half and the rhythm and pitch stay the same."},
        ],
    },
    {
        "title": "Durations",
        "segment": (46.3, 72.3),
        "figures": [{"t": 63.9, "caption": "Frames per phoneme symbol (1 frame = 25 ms). 35 of 58 get 2 frames; "
                                           "the final s gets 13. In all: 141 frames, 3.53 s."},
                    {"t": 72.0, "caption": "Each phoneme's vector is repeated once per frame: “narrator” becomes 17 "
                                           "frames."}],
        "body": [
            """<p>After the phoneme BERT reads the sentence, the first prediction is <b>how long each phoneme
lasts</b>, as a whole number of <b>frames</b> of 25 ms (600 samples at 24,000 samples per second). Most symbols
get 2 frames (50 ms). Stress marks and short sounds get 1. The final “s” gets 13 frames, and the silences at the
start and end are also predicted. The sentence totals 141 frames: 3.525 seconds, 84,600 samples.</p>""",
            "{fig0}",
            """<p>Then comes a simple but crucial step, the <b>alignment</b>: each phoneme's vector is copied once
for every frame it lasts. The six sounds of “narrator” (plus a stress mark) become 17 frame vectors. From here on
the model works in <b>time</b>, 40 frames per second, not in symbols. This is what lets a text model produce
speech of the right length, and it is also where speed is controlled: Kokoro's speed setting divides every
duration. At speed 1.4 the sentence lasts 2.35 s, and Whisper already starts to mishear it (“texting a speech”).</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>A duration predictor turns symbols into time: each phoneme
gets a number of 25 ms frames, and its vector is repeated that many times.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A phoneme gets 3 frames. How long does it last, in milliseconds?",
             "answer": "75 ms.", "why": "3 × 25 ms.",
             "key": {"parts": [{"label": None, "value": 75, "tol": 0, "unit": "ms"}]}},
            {"kind": "number", "q": "A sentence is predicted to last 200 frames. How many samples does Kokoro "
                                    "produce?",
             "answer": "120,000.", "why": "600 samples per frame: 200 × 600 = 120,000, i.e. 5 seconds.",
             "key": {"parts": [{"label": None, "value": 120000, "tol": 0, "unit": "samples"}]}},
        ],
    },
    {
        "title": "Pitch and energy",
        "segment": (72.5, 122.4),
        "figures": [{"t": 83.8, "caption": "Two predicted curves, 80 values a second: pitch (median 190 Hz) and "
                                           "energy."},
                    {"t": 95.0, "caption": "Flattened pitch: Whisper still hears every word, but the voice sounds "
                                           "like a robot."},
                    {"t": 108.5, "caption": "Pitch × 1.5: measured 204 → 282 Hz, same words, same 3.53 s."},
                    {"t": 121.5, "caption": "Another voice's first half with our second half: our pitch (200 Hz) and "
                                            "rhythm, another timbre."}],
        "body": [
            """<p>From the frame vectors and the speaking half of the style, the prosody predictor makes two more
curves at 80 values per second (twice the frame rate): the <b>pitch</b> (fundamental frequency, F0) and the
<b>energy</b> (how loud each moment is). For our sentence the predicted pitch has a median of 190 Hz and is near
zero in unvoiced sounds and pauses. Together the curves are the melody of the sentence.</p>""",
            "{fig0}",
            """<p>Since they are just numbers, we can <b>edit them before the decoder</b>. With the pitch flattened
to its median wherever the voice is voiced, Whisper still transcribes the sentence without errors, so the words
do not depend on the melody. The measured pitch spread drops from 82 Hz to 8 Hz, and the result sounds flat and
robotic (the video plays it).</p>""",
            "{fig1}",
            """<p>Multiplying the pitch by 1.5 raises the measured pitch from 204 to 282 Hz with the same words and
the same length. Giving every symbol the same 2 frames shortens the sentence to 3.25 s, and Whisper still reads it
perfectly. (Measured pitch comes from the audio by autocorrelation, as in episode 15, so it differs a little from
the predicted curve.)</p>""",
            "{fig2}",
            """<p>Finally, swapping only the <b>first half</b> of the style vector for a male voice's (am_michael)
changes the timbre, while the pitch (200 Hz) and the length (3.53 s) stay ours: the second half carries rhythm and
melody. Swapping the whole vector gives the male voice's own prosody: 120 Hz and 3.95 s.</p>""",
            "{fig3}",
            """<div class="box key"><b class="t">Key idea</b>Prosody is predicted as explicit curves. That makes it
controllable: edit pitch, energy or durations and the decoder follows.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“With the pitch curve flattened, Whisper can no longer understand the sentence.”",
             "answer": "False.", "why": "It still makes no errors; the words live elsewhere. What changes is how "
                                        "natural the voice sounds.",
             "key": {"value": False}},
            {"kind": "number", "q": "A predicted pitch curve sits at 190 Hz. You multiply it by 1.5 before the "
                                    "decoder. What pitch do you expect, in Hz?",
             "answer": "285 Hz.", "why": "190 × 1.5 = 285 (the video measures 204 → 282 Hz from the audio).",
             "key": {"parts": [{"label": None, "value": 285, "tol": 1, "unit": "Hz"}]}},
            {"kind": "mc", "q": "You swap only the first half of the style vector for another voice's. What "
                                "changes?",
             "options": ["The words", "The rhythm and pitch", "The timbre (how the voice sounds)", "Nothing"],
             "answer": "C.", "why": "The first half conditions the decoder; pitch and rhythm come from the second "
                                    "half through the prosody predictor.",
             "key": {"choice": 2}},
        ],
    },
    {
        "title": "The vocoder",
        "segment": (122.55, 175.0),
        "figures": [{"t": 146.4, "caption": "The decoder: sine waves at the predicted pitch, upsampling, then the "
                                            "strength and phase of 11 frequencies every 5 samples, and an inverse "
                                            "Fourier transform."},
                    {"t": 160.2, "caption": "Trained against judges (episode 4). On this CPU the 3.5 s sentence takes "
                                            "0.87 s; the decoder takes 91%."},
                    {"t": 166.6, "caption": "The whole pipeline in five lines."}],
        "body": [
            """<p>The <b>decoder</b> turns frame vectors, pitch and energy into a waveform. It is a
<b>vocoder</b> (an iSTFTNet). It does not start from nothing: a <b>source</b> module turns the predicted pitch into
sine waves at that pitch and its first 8 harmonics, which the network then shapes. Its layers upsample the 80
steps per second ×10 and ×6 to 4,800, and the last layer predicts, every 5 samples, the <b>magnitude and phase</b>
of 11 frequencies (an FFT of size 20). An inverse short-time Fourier transform gives 24,000 samples per second:
40 frames × 2 × 10 × 6 × 5.</p>""",
            "{fig0}",
            """<p>This closes the loop with episode 15. A spectrogram keeps magnitudes and throws away phase, and
Griffin-Lim could only guess the phase back, which sounded rough. A vocoder <b>predicts the phase</b> along with
the magnitudes. It sounds natural because it was trained with <b>judges</b>: discriminator networks that try to
tell real recordings from generated ones, the forger-and-judge game of episode 4.</p>""",
            "{fig1}",
            """<p>On this 8-core CPU (8 threads, no GPU) the 3.525-second sentence takes 0.87 s, 4 times faster than
real time. The decoder takes 0.79 s of that (91%); the phoneme BERT, predictor and text encoder together take
75 ms.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>The vocoder starts from sine waves at the predicted pitch
and predicts magnitudes and phases directly. GAN-style training makes the result sound natural.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The decoder's last layer predicts a set of 11 frequencies every 5 samples, at "
                                    "24,000 samples per second. How many sets per second?",
             "answer": "4,800.", "why": "24,000 / 5 = 4,800.",
             "key": {"parts": [{"label": None, "value": 4800, "tol": 0, "unit": "per second"}]}},
            {"kind": "short", "q": "Why does Kokoro sound more natural than episode 15's Griffin-Lim "
                                   "reconstruction from a spectrogram?",
             "answer": "It predicts the phase instead of guessing it, and it was trained with discriminators that "
                       "punish anything that does not sound like a real recording.",
             "why": "Magnitudes alone keep the words; natural sound needs the phase."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Run Kokoro stage by stage and edit the pitch.",
             "code": """out = synth(m, ids, ref, edit_f0=lambda f: f * 1.5)
sf.write("higher.wav", out["audio"], 24000)
print(out["dur"], out["f0"][:10])""",
             "answer": "inside_kokoro.py's synth() returns the audio and every intermediate (durations, pitch, "
                       "energy).",
             "why": "Try edit_dur to change the rhythm, or sound_ref to borrow another voice's timbre."},
        ],
    },
]
