"""Generate narration for one video. From a series folder: python ../framework/voice.py v01

Writes voice/<video>/s<N>.wav and voice/<video>/sections.json (durations + word timestamps).
"""
import importlib, json, os, re, sys

import soundfile as sf
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

VOICE = "af_heart"
TTS_DIR = os.path.join(os.path.dirname(__file__), "..", "tts")

sys.path.insert(0, os.getcwd())  # run from a series folder, e.g. how-llms-work/
video = sys.argv[1]
sections = importlib.import_module(f"{video}_script").SECTIONS
out = os.path.join("voice", video)
os.makedirs(out, exist_ok=True)

k = Kokoro(f"{TTS_DIR}/kokoro-v1.0.onnx", f"{TTS_DIR}/voices-v1.0.bin")
w = WhisperModel("small.en", device="cpu", compute_type="int8")
meta = []
for i, text in enumerate(sections, 1):
    audio, sr = k.create(text, voice=VOICE, speed=1.0, lang="en-us")
    path = os.path.join(out, f"s{i}.wav")
    sf.write(path, audio, sr)
    segs, _ = w.transcribe(path, word_timestamps=True)
    words = [{"w": re.sub(r"[^a-z0-9]", "", x.word.lower()), "t": round(x.start, 2)}
             for s in segs for x in s.words]
    meta.append({"file": path, "dur": round(len(audio) / sr, 2), "words": words})
    print(f"s{i}: {meta[-1]['dur']:5.1f}s  " + " ".join(f"{x['w']}@{x['t']}" for x in words))
json.dump(meta, open(os.path.join(out, "sections.json"), "w"), indent=1)
print(f"total {sum(m['dur'] for m in meta):.1f}s")
