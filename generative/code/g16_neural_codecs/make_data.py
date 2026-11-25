"""Speech for the tiny codec: sentences from this channel's own narration scripts, spoken by Kokoro.

    python3 make_data.py        # ~20 minutes on 16 cores → data/train.npy, data/test.npy, data/test_*.wav, data/manifest.json

Training: sentences from every series except this one, each spoken by one of 8 Kokoro voices.
Test: 40 sentences from this series' scripts (never trained on), spoken by af_heart, the narrator's voice.
Needs the Kokoro model files (KOKORO_DIR, default ../../../tts: kokoro-v1.0.onnx, voices-v1.0.bin).
"""
import glob
import importlib.util
import json
import os
import random
import re

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
KOKORO_DIR = os.environ.get("KOKORO_DIR", os.path.join(HERE, "../../../tts"))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
SR = 24000
VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "am_michael", "am_adam", "bf_emma", "bm_george"]
TRAIN_MINUTES = 30
WORKERS = 4
N_TEST = 40


def sentences(path):
    spec = importlib.util.spec_from_file_location("s", path)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception:
        return []
    out = []
    for sec in getattr(m, "SECTIONS", []):
        for s in re.split(r"(?<=[.!?])\s+", sec):
            n = len(s.split())
            if 6 <= n <= 28 and not re.search(r"[<>{}\[\]_=]", s):
                out.append(s.strip())
    return out


_k = None


def speak(job):
    """Worker: one Kokoro instance per process, 4 threads each."""
    global _k
    if _k is None:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 4
        sess = ort.InferenceSession(f"{KOKORO_DIR}/kokoro-v1.0.onnx", opt, providers=["CPUExecutionProvider"])
        _k = Kokoro.from_session(sess, f"{KOKORO_DIR}/voices-v1.0.bin")
    text, voice = job
    a, sr = _k.create(text, voice=voice, speed=1.0, lang="en-us")
    assert sr == SR
    return a.astype(np.float32)


if __name__ == "__main__":
    from multiprocessing import Pool

    rng = random.Random(0)
    scripts = sorted(glob.glob(os.path.join(REPO, "*", "*_script.py")))
    test_pool = [s for p in scripts if "/generative/" in p for s in sentences(p)]
    train_pool = [s for p in scripts if "/generative/" not in p for s in sentences(p)]
    train_pool = sorted(set(train_pool) - set(test_pool))
    rng.shuffle(train_pool)
    test = rng.sample(sorted(set(test_pool)), N_TEST)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)

    jobs = [(s, VOICES[i % len(VOICES)]) for i, s in enumerate(train_pool)]
    pause = np.zeros(SR // 10, np.float32)
    chunks, meta, total = [], [], 0
    with Pool(WORKERS) as pool:
        for (s, v), a in zip(jobs, pool.imap(speak, jobs, chunksize=4)):
            chunks += [a, pause]
            total += len(a) + len(pause)
            meta.append({"voice": v, "text": s, "samples": len(a)})
            if len(meta) % 50 == 0:
                print(f"train {len(meta)} sentences, {total / SR / 60:.1f} min", flush=True)
            if total >= TRAIN_MINUTES * 60 * SR:
                pool.terminate()
                break
        np.save(os.path.join(HERE, "data/train.npy"), np.concatenate(chunks))
    with Pool(WORKERS) as pool:
        audio = pool.map(speak, [(s, "af_heart") for s in test])
    tests = []
    for i, (s, a) in enumerate(zip(test, audio)):
        sf.write(os.path.join(HERE, f"data/test_{i:02d}.wav"), a, SR)
        tests.append({"file": f"data/test_{i:02d}.wav", "text": s, "samples": len(a)})
    json.dump({"voices": VOICES, "train_sentences": len(meta), "train_seconds": total / SR,
               "train_pool": len(train_pool), "test": tests, "train": meta},
              open(os.path.join(HERE, "data/manifest.json"), "w"), indent=1)
    print(f"done: {len(meta)} train sentences, {total / SR / 60:.1f} min; {len(tests)} test sentences")
