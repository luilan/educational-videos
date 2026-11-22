"""Study guide content for How Generative Models Work: Images and Sound, episode 15: Sound as Data.

Build:  python framework/study_guide.py generative g15 --video generative/media/videos/g15_scene/1080p60/SoundVideo.mp4
Every number comes from code/g15_sound (sound.py, 2026-11-22): one sentence spoken by Kokoro's af_heart voice.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 15",
    "title": "Sound as Data",
    "tagline": "From air pressure to pictures of sound.",
    "duration": "2:54",
    "intro": """<p>Part 4 turns from pictures to sound. Before any model can make speech, sound has to become numbers.
This lesson measures one sentence spoken by the voice that narrates this series: its samples, its pitch, what the
sample rate allows, the <b>spectrogram</b> and the <b>mel scale</b> that speech models read and write, and what a
spectrogram throws away.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> the Foundations series' waves and Fourier transform.
Code: <code>code/g15_sound</code> (numpy, scipy, the Kokoro voice and Whisper; about 2 minutes on a CPU). The video
plays four short audio clips; they are in <code>code/g15_sound/out</code>.</div>""",
}

CONCEPTS = [
    {
        "title": "Samples",
        "segment": (7.7, 75.0),
        "figures": [{"t": 29.0, "caption": "One sentence: 172,544 numbers in 7.2 seconds. Loud parts swing wide; "
                                           "pauses stay near zero."},
                    {"t": 37.3, "caption": "30 ms of a vowel: the wave repeats every 5.0 ms, 200 times a second, "
                                           "the voice's pitch."},
                    {"t": 57.3, "caption": "Aliasing: a 5,000 Hz tone sampled 8,000 times a second gives the same "
                                           "samples as a 3,000 Hz tone."},
                    {"t": 74.8, "caption": "5.9% of the energy lies above 4,000 Hz. The video plays the sentence at "
                                           "24,000, then at 8,000 samples a second."}],
        "body": [
            """<p>A microphone measures air pressure. Digital audio stores that measurement many times a second as a
list of numbers, the <b>waveform</b>. Our narrator's voice uses <b>24,000 samples per second</b>. One 7.19-second
sentence is 172,544 numbers, between −0.38 and 0.58; as 16-bit integers that is 384 kilobits per second.</p>""",
            "{fig0}",
            """<p>Up close, a vowel is periodic. The 30 ms close-up repeats every 5.0 ms, 200 times a second. That
rate is the <b>pitch</b> (the fundamental frequency). Measured by autocorrelation across the whole sentence, the
median is 200 Hz and most voiced frames lie between 171 and 258 Hz.</p>""",
            "{fig1}",
            """<p>To capture a frequency you need more than two samples per cycle. A sample rate of
<i>f</i><sub>s</sub> therefore captures frequencies only up to <i>f</i><sub>s</sub>/2, the <b>Nyquist limit</b>:
12,000 Hz for our voice. A higher frequency does not just vanish; it <b>aliases</b>. A 5,000 Hz tone sampled 8,000
times per second produces exactly the samples of a 3,000 Hz tone, and that is what plays back. That is why audio is
low-pass filtered before sampling.</p>""",
            "{fig2}",
            """<p>Telephones used 8,000 samples per second, so nothing above 4,000 Hz. That is only 5.9% of this
sentence's energy, but it includes much of the hiss of “s”, and it is easy to hear.</p>""",
            "{fig3}",
            """<div class="box key"><b class="t">Key idea</b>Sound is a long list of numbers. The sample rate sets
the highest frequency it can hold: half the rate.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Whisper resamples audio to 16,000 samples per second. What is the highest "
                                    "frequency it can see, in Hz?",
             "answer": "8,000 Hz.", "why": "Half the sample rate (Nyquist).",
             "key": {"parts": [{"label": None, "value": 8000, "tol": 1, "unit": "Hz"}]}},
            {"kind": "number", "q": "A 7,000 Hz tone is sampled 8,000 times a second without a filter. At what "
                                    "frequency does it come out, in Hz?",
             "answer": "1,000 Hz.", "why": "It folds around the 4,000 Hz limit: 8,000 − 7,000 = 1,000.",
             "key": {"parts": [{"label": None, "value": 1000, "tol": 1, "unit": "Hz"}]}},
        ],
    },
    {
        "title": "Pictures of sound",
        "segment": (75.2, 121.4),
        "figures": [{"t": 91.3, "caption": "Spectrogram: 25 ms windows every 10 ms, a Fourier transform of each. "
                                           "Time across, frequency (0–12,000 Hz) up, brightness = loudness."},
                    {"t": 101.3, "caption": "Harmonics at the bottom, bending with the pitch; the broad columns are "
                                            "the hiss of “s”."},
                    {"t": 121.3, "caption": "80 mel filters: 52 Hz wide at the bottom, 877 Hz at the top; below, "
                                            "the log-mel spectrogram."}],
        "body": [
            """<p>What we hear is frequencies, and raw samples hide them. The <b>short-time Fourier transform</b>
cuts the sound into short overlapping windows (25 ms, i.e. 600 samples, tapered by a Hann window, every 10 ms) and
takes each window's Fourier transform. The magnitudes, one column per window, form a <b>spectrogram</b>. Here that
is 513 frequency bins 23.4 Hz apart and 100 columns per second.</p>""",
            "{fig0}",
            """<p>Reading it: the evenly spaced stripes at the bottom are the <b>harmonics</b> of the voice (200,
400, 600 Hz … for a 200 Hz pitch). They bend when the pitch moves. The broad bright regions that rise and fall are
the vowels' resonances. Noisy consonants such as “s” are broad columns reaching the top.</p>""",
            "{fig1}",
            """<p>Our ears separate low pitches much better than high ones. The <b>mel scale</b>, mel(f) =
2595·log10(1 + f/700), spaces frequencies the way we hear them: 1,000 Hz is 1,000 mel, but 12,000 Hz is only 3,266.
Eighty triangular filters, equally spaced in mel, are 52 Hz wide at the bottom and 877 Hz wide at the top. 24 of
them lie below 1,000 Hz and only 17 above 6,000 Hz. The log of each filter's energy gives the <b>log-mel
spectrogram</b>: 80 × 100 = 8,000 numbers per second instead of 24,000 samples, laid out as a picture. Whisper
reads exactly this kind of input (80 bands, 25 ms / 10 ms, from 16 kHz audio), and many speech generators predict
it.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>A spectrogram turns sound into a picture of frequency
over time. The mel scale squeezes it to match our hearing: fine at the bottom, coarse at the top.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With 25 ms windows every 10 ms, how many spectrogram columns does a 3-second "
                                    "clip have (approximately)?",
             "answer": "About 300.", "why": "One column every 10 ms: 3 / 0.01 ≈ 300 (298 exactly, counting only "
                                            "full windows).",
             "key": {"parts": [{"label": None, "value": 300, "tol": 3, "unit": "columns"}]}},
            {"kind": "mc", "q": "Why are mel filters narrow at low frequencies and wide at high ones?",
             "options": ["High frequencies carry no information", "To follow hearing: we tell low pitches apart far "
                         "more finely than high ones", "Because the Fourier transform is less accurate at high "
                         "frequencies", "To make the spectrogram square"],
             "answer": "B.", "why": "Equal steps in mel are roughly equal steps in perceived pitch.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "What the picture loses",
        "segment": (121.7, 166.6),
        "figures": [{"t": 133.9, "caption": "Kept: how strong each frequency is. Lost: its phase. With random "
                                            "phases, Whisper still reads every word."},
                    {"t": 149.6, "caption": "Griffin-Lim: alternate between sound and spectrogram 60 times, keeping "
                                            "the true magnitudes each round."},
                    {"t": 160.0, "caption": "Natural sound from a spectrogram takes a trained vocoder (episode 17)."},
                    {"t": 166.5, "caption": "The log-mel spectrogram in code."}],
        "body": [
            """<p>A spectrogram stores each frequency's magnitude, not its <b>phase</b> (where in its cycle each
wave is). To get sound back, phases must be supplied. With random phases, Whisper still transcribes the sentence
with no word errors, so the words live in the magnitudes. But the sound is rough; the video plays it. Re-analyzed,
its magnitudes no longer match the original's (mismatch 0.557, where 0 means identical): random phases
make the overlapping windows interfere.</p>""",
            "{fig0}",
            """<p><b>Griffin-Lim</b> (Griffin &amp; Lim, 1984) guesses the phase iteratively. Turn magnitudes plus a
phase guess into sound, analyze that sound again, keep its phase, put the true magnitudes back, and repeat. After 60
rounds the mismatch falls to 0.099, and Whisper still makes no errors. It is closer, but not the original: the
video plays both. Starting from the 80-band mel spectrogram instead, which also loses detail, the mismatch is
0.184.</p>""",
            "{fig1}",
            """<p>So turning a spectrogram back into natural speech is itself a learning problem. The model that
does it is a <b>vocoder</b>, and the voice narrating this series has one (episode 17). Another route skips
spectrograms altogether and turns sound into tokens (episode 16).</p>""",
            "{fig2}",
            "{fig3}",
            """<div class="box key"><b class="t">Key idea</b>A spectrogram keeps what is said but not quite how it
sounds. Bringing back natural sound needs the phase, and a trained vocoder supplies it.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Since Whisper reads the random-phase version perfectly, the phase does not "
                                "matter.”",
             "answer": "False.", "why": "It does not matter for recognizing words, but it matters a great deal for "
                                        "how natural the sound is, which is what a speech generator must get right.",
             "key": {"value": False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> A log-mel spectrogram in numpy.",
             "code": """frames = [x[i:i + 600] * hann for i in range(0, len(x) - 600, 240)]
spec = abs(rfft(frames, 1024))
mel = mel_filters @ spec ** 2
log_mel = log10(mel)""",
             "answer": "sound.py does exactly this (stft, mel_filters), then Griffin-Lim to go back.",
             "why": "Window, Fourier transform, mel filters, log: four lines from sound to a picture."},
        ],
    },
]
