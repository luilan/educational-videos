"""How Generative Models Work: Images and Sound, episode 15 — Sound as Data. One entry per narration section.

Every number comes from code/g15_sound (sound.py, 2026-11-22): one sentence spoken by Kokoro's af_heart voice, the
voice that narrates this series. The video plays four short clips from code/g15_sound/out between sections.
"""
TITLE = "Sound as Data"
TAGLINE = "From air pressure to pictures of sound."
NEXT = "Audio Tokens: Neural Codecs"
SECTIONS = [
    # 1. Hook
    "Part four: sound. The voice you are hearing is a list of numbers, 24,000 of them every second. Each one says "
    "how far the air is pushed at one instant.",
    # 2. The waveform
    "Here is one sentence from our narrator: seven seconds, 172,544 numbers. Loud parts swing wide, pauses stay "
    "near zero.",
    # 3. Close-up
    "Zoom into 30 milliseconds of a vowel, and the wave repeats, about 200 times a second. That is the pitch of "
    "the voice.",
    # 4. Sampling
    "How many numbers per second do you need? More than two per cycle. At 24,000 samples a second, the highest "
    "frequency is 12,000 hertz. Sample a 5,000 hertz tone only 8,000 times a second, and it comes out as 3,000 "
    "hertz. That is aliasing, so audio is filtered before sampling.",
    # 5. Listen: sample rates
    "At 8,000 samples a second, everything above 4,000 hertz is gone. That is only 6 percent of the energy. "
    "Listen: first at 24,000, then at 8,000.",
    # 6. Spectrogram
    "Raw samples hide what we hear: frequencies. So cut the sound into 25 millisecond windows, every 10 "
    "milliseconds, and take each window's Fourier transform. The result is a spectrogram: time across, frequency "
    "up, loudness as color.",
    # 7. Reading it
    "The stripes at the bottom are the voice's harmonics, multiples of its pitch. They bend as the pitch moves. "
    "The noise reaching the top is the hiss of each s.",
    # 8. Mel
    "Our ears tell low pitches apart far better than high ones. The mel scale follows that. 80 filters: 52 hertz "
    "wide at the bottom, 877 at the top. The log of their energies is the log-mel spectrogram, what Whisper "
    "reads. It turns 24,000 numbers a second into 8,000.",
    # 9. Phase
    "But a spectrogram keeps only how strong each frequency is, not its phase: where each wave is in its cycle. "
    "Choose the phases at random, and Whisper still reads every word. But listen.",
    # 10. Griffin-Lim
    "Griffin and Lim, in 1984, found a way to guess the phase back from the magnitudes, going back and forth "
    "sixty times. Closer, but listen again.",
    # 11. Vocoders
    "So turning a spectrogram back into natural sound is a job for a trained model, a vocoder. Our narrator has "
    "one. We will open it in episode 17.",
    # 12. Code
    "In code: cut the samples into windows, take the Fourier transform of each, and apply the mel filters.",
    # 13. Next
    "Next: turning sound into tokens, with neural codecs.",
]
