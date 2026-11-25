"""How Generative Models Work: Images and Sound, episode 17 — Text to Speech: Inside Our Narrator.

Every number comes from code/g17_inside_our_narrator (inside_kokoro.py, 2026-11-25): Kokoro v1.0 run one stage at a
time on "Our narrator turns text into speech in three steps." The video plays three clips from out/ between sections.
"""
TITLE = "Text to Speech: Inside Our Narrator"
TAGLINE = "Phonemes, a style vector, durations, pitch, and a vocoder."
NEXT = "Speech to Text: Inside Whisper"
SECTIONS = [
    # 1. Hook
    "Every word in this series is spoken by Kokoro, an open model with 82 million parameters. It runs on an "
    "ordinary CPU. Let's open it up and watch each step.",
    # 2. Phonemes
    "First, text becomes phonemes, symbols for sounds, because spelling is a poor guide. Though, tough, through "
    "and thought share four letters and sound nothing alike. Our sentence's 51 characters become 58 symbols, "
    "including marks for stress.",
    # 3. Style vector
    "A voice is not a recording. It is a vector of 256 numbers. The first half shapes how the voice sounds. The "
    "second half shapes how it speaks: its rhythm and melody.",
    # 4. Durations
    "A small Bert model reads the phonemes in context. Then comes the first prediction: how long each phoneme "
    "lasts, in frames of 25 milliseconds. Most get two frames. The final s gets 13. The whole sentence: 141 "
    "frames, three and a half seconds.",
    # 5. Alignment
    "Each phoneme's vector is now copied once for every frame it lasts. From here on, the model works in time, "
    "not in symbols.",
    # 6. Pitch and energy
    "Two more curves, 80 values a second: the pitch, around 190 hertz for this voice, and the energy, how loud "
    "each moment is. Together they are the melody of the sentence.",
    # 7. Flat pitch
    "These are just numbers, so we can edit them before the decoder. Flatten the pitch, and Whisper still hears "
    "every word. But listen.",
    # 8. Higher pitch
    "Multiply the pitch by one and a half. The measured pitch goes from 204 to 282 hertz, with the same words and "
    "rhythm.",
    # 9. Another voice
    "Now swap only the first half of the style vector for a male voice's. The pitch and rhythm stay ours. Only "
    "the sound of the voice changes.",
    # 10. Decoder
    "The decoder turns frames into sound. It holds 53 million parameters, two thirds of the model. It starts from "
    "sine waves at the predicted pitch, upsamples 60 times, and predicts the strength and the phase of 11 "
    "frequencies every 5 samples. An inverse Fourier transform gives the waveform. The phase that Griffin-Lim had "
    "to guess in episode 15 is simply predicted.",
    # 11. Training and speed
    "It learned with judges: networks trained to tell real recordings from generated ones, the forger and judge "
    "of episode 4. That is why it sounds natural. On this CPU, the decoder takes most of the time.",
    # 12. Code
    "In code: phonemes in, durations, repeat each phoneme, predict pitch and energy, then decode.",
    # 13. Next
    "Next, the last episode: the other direction. Speech to text, inside Whisper.",
]
