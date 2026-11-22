"""Episode 15: sound as data. Everything is measured on one sentence spoken by the series' narrator voice.

    python3 sound.py            # ~2 minutes → out/*.wav, out/*.png, results.json

1. The waveform: Kokoro (af_heart, the voice that narrates this series) speaks a sentence at 24,000 samples per
   second. How many numbers, their range, and a 30 ms close-up of a vowel; the pitch from autocorrelation.
2. Sample rates: the same sentence at 16,000 and 8,000 samples per second, how much of its energy lies above each
   Nyquist limit, and aliasing: a 5,000 Hz tone sampled 8,000 times per second without a filter comes out at 3,000 Hz.
3. The spectrogram: short-time Fourier transform, 25 ms Hann windows every 10 ms.
4. The mel scale: 80 triangular filters equally spaced in mel, log energies (the log-mel spectrogram).
5. What a spectrogram throws away: the phase. Griffin-Lim (1984) guesses it back from the magnitudes alone. Whisper
   (small.en, the model that times this series' animations) transcribes each version.

Needs the Kokoro model files (KOKORO_DIR, default ../../../tts: kokoro-v1.0.onnx, voices-v1.0.bin).
"""
import json
import os
import re

import numpy as np
import soundfile as sf
from PIL import Image
from scipy.signal import resample_poly

KOKORO_DIR = os.environ.get("KOKORO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../tts"))
TEXT = ("Sound is a long list of numbers. Our narrator makes thousands of them every second, "
        "and a model can learn to make them too.")
SR = 24_000
WIN, HOP, NFFT = 600, 240, 1024          # 25 ms window, 10 ms hop
N_MELS = 80
RES = {}


def mel(f):
    return 2595 * np.log10(1 + np.asarray(f, float) / 700)


def mel_inv(m):
    return 700 * (10 ** (np.asarray(m, float) / 2595) - 1)


def stft(x):
    w = np.hanning(WIN)
    n = 1 + (len(x) - WIN) // HOP
    frames = np.stack([x[i * HOP:i * HOP + WIN] * w for i in range(n)])
    return np.fft.rfft(frames, NFFT, axis=1).T                                  # (NFFT/2+1, frames)


def istft(S, length):
    w = np.hanning(WIN)
    frames = np.fft.irfft(S.T, NFFT, axis=1)[:, :WIN] * w
    out, norm = np.zeros(length), np.zeros(length)
    for i, f in enumerate(frames):
        out[i * HOP:i * HOP + WIN] += f
        norm[i * HOP:i * HOP + WIN] += w ** 2
    return out / np.maximum(norm, 1e-8)


def griffin_lim(mag, length, iters, seed=0):
    rng = np.random.default_rng(seed)
    phase = np.exp(2j * np.pi * rng.random(mag.shape))
    for _ in range(iters):
        x = istft(mag * phase, length)
        phase = np.exp(1j * np.angle(stft(x)))
    return istft(mag * phase, length)


def mel_filters(sr=SR, n=N_MELS):
    """Triangles equally spaced on the mel scale from 0 Hz to sr/2, each normalized to unit area (Slaney style)."""
    freqs = np.fft.rfftfreq(NFFT, 1 / sr)
    edges = mel_inv(np.linspace(0, mel(sr / 2), n + 2))
    fb = np.zeros((n, len(freqs)))
    for i in range(n):
        lo, c, hi = edges[i:i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
        fb[i] *= 2 / (hi - lo)
    return fb, edges


def colormap(v):
    """v in [0, 1] → RGB, a dark-purple-to-yellow ramp (close to matplotlib's magma)."""
    stops = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21],
                      [250, 193, 39], [252, 255, 164]], float)
    p = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(p.astype(int), len(stops) - 2)
    f = (p - i)[..., None]
    return ((1 - f) * stops[i] + f * stops[i + 1]).astype(np.uint8)


def save_image(db, path, lo=-80, scale=(2, 2)):
    """A (rows = low to high frequency, cols = time) dB array → PNG, low frequencies at the bottom."""
    v = (db - db.max() - lo) / -lo
    img = colormap(np.flipud(v))
    img = np.kron(img, np.ones((scale[0], scale[1], 1), np.uint8))
    Image.fromarray(img).save(path)


def words(s):
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)


if __name__ == "__main__":
    from faster_whisper import WhisperModel
    from kokoro_onnx import Kokoro

    os.makedirs("out", exist_ok=True)
    k = Kokoro(f"{KOKORO_DIR}/kokoro-v1.0.onnx", f"{KOKORO_DIR}/voices-v1.0.bin")
    audio, sr = k.create(TEXT, voice="af_heart", speed=1.0, lang="en-us")
    assert sr == SR
    x = np.asarray(audio, np.float64)
    sf.write("out/sentence.wav", x, SR)

    # 1. the waveform
    RES["waveform"] = {"text": TEXT, "sample_rate": SR, "samples": len(x), "seconds": len(x) / SR,
                       "min": float(x.min()), "max": float(x.max()),
                       "bytes_16bit": 2 * len(x), "kbit_per_s_16bit": SR * 16 / 1000}
    env = np.abs(x[:len(x) // 400 * 400]).reshape(-1, 400).max(1)                # loudest value in each 1/60 s (431 values)
    # pitch by autocorrelation on 40 ms frames with enough energy; lags for 70–400 Hz
    F, f0 = 960, []
    lags = np.arange(SR // 400, SR // 70)
    for i in range(0, len(x) - F, HOP):
        fr = x[i:i + F] - x[i:i + F].mean()
        e = (fr ** 2).mean()
        if e < 0.1 * (x ** 2).mean():                                           # too quiet: silence
            f0.append(0.0)
            continue
        ac = np.correlate(fr, fr, "full")[F - 1:]
        ac = ac / ac[0]
        best = lags[np.argmax(ac[lags])]
        f0.append(SR / best if ac[best] > 0.5 else 0.0)
    f0 = np.array(f0)
    voiced = f0[f0 > 0]
    RES["pitch"] = {"frames": len(f0), "voiced_frames": int(len(voiced)), "median_hz": float(np.median(voiced)),
                    "p10_hz": float(np.percentile(voiced, 10)), "p90_hz": float(np.percentile(voiced, 90))}
    # a 30 ms close-up of a vowel: the loud, strongly periodic frame whose pitch is nearest the median
    best_i, best_score = 0, -1
    for j, i in enumerate(range(0, len(x) - F, HOP)):
        if f0[j] == 0 or abs(f0[j] - np.median(voiced)) > 10:
            continue
        fr = x[i:i + F] - x[i:i + F].mean()
        ac = np.correlate(fr, fr, "full")[F - 1:]
        score = (fr ** 2).mean() * ac[int(round(SR / f0[j]))] / ac[0]
        if score > best_score:
            best_i, best_score = i + F // 2, score
    i0 = best_i
    close = x[i0 - 360:i0 + 360]
    RES["closeup"] = {"start_s": (i0 - 360) / SR, "samples": len(close)}
    json.dump({"envelope": [round(float(v), 4) for v in env], "envelope_hop_s": 400 / SR,
               "closeup": [round(float(v), 4) for v in close], "f0_hz": [round(float(v), 1) for v in f0],
               "f0_hop_s": HOP / SR}, open("waveform.json", "w"))

    # 2. sample rates
    spec = np.abs(np.fft.rfft(x)) ** 2
    fq = np.fft.rfftfreq(len(x), 1 / SR)
    RES["sample_rates"] = {"energy_above_4000_hz": float(spec[fq > 4000].sum() / spec.sum()),
                           "energy_above_8000_hz": float(spec[fq > 8000].sum() / spec.sum())}
    for low in (16_000, 8_000):
        y = resample_poly(x, low // 1000, SR // 1000)                         # filters, then keeps fewer samples
        back = resample_poly(y, SR // 1000, low // 1000)
        sf.write(f"out/sentence_{low // 1000}k.wav", back, SR)
        RES["sample_rates"][f"{low}"] = {"samples": len(y)}
    t = np.arange(8000) / 8000
    tone = np.sin(2 * np.pi * 5000 * t)                                         # 5 kHz, sampled 8,000 times a second
    peak = np.fft.rfftfreq(len(tone), 1 / 8000)[np.argmax(np.abs(np.fft.rfft(tone)))]
    RES["aliasing"] = {"tone_hz": 5000, "sample_rate": 8000, "heard_hz": float(peak)}

    # 3. spectrogram
    S = stft(x)
    mag = np.abs(S)
    db = 20 * np.log10(mag + 1e-6)
    save_image(db, "out/spectrogram.png")
    RES["spectrogram"] = {"window_ms": 1000 * WIN / SR, "hop_ms": 1000 * HOP / SR, "frames": S.shape[1],
                          "bins": S.shape[0], "bin_hz": SR / NFFT, "numbers": int(S.size),
                          "frames_per_second": SR / HOP}

    # 4. mel
    fb, edges = mel_filters()
    M = fb @ mag ** 2
    mdb = 10 * np.log10(M + 1e-10)
    save_image(mdb, "out/mel.png", scale=(6, 2))
    widths = edges[2:] - edges[:-2]
    RES["mel"] = {"filters": N_MELS, "frames": M.shape[1], "numbers": int(M.size),
                  "numbers_per_second": N_MELS * SR / HOP,
                  "first_filter_width_hz": float(widths[0]), "last_filter_width_hz": float(widths[-1]),
                  "filters_below_1000_hz": int((edges[1:-1] < 1000).sum()),
                  "filters_above_6000_hz": int((edges[1:-1] > 6000).sum()),
                  "mel_of_1000_hz": float(mel(1000)), "mel_of_12000_hz": float(mel(12000))}
    json.dump({"edges_hz": [round(float(e), 1) for e in edges]}, open("mel_edges.json", "w"))

    # 5. phase
    versions = {"original": x}
    versions["random_phase"] = istft(mag * np.exp(2j * np.pi * np.random.default_rng(0).random(mag.shape)), len(x))
    versions["griffin_lim_60"] = griffin_lim(mag, len(x), 60)
    mag_from_mel = np.sqrt(np.clip(np.linalg.pinv(fb) @ M, 0, None))           # rough undo of the mel step
    versions["from_mel_griffin_lim_60"] = griffin_lim(mag_from_mel, len(x), 60)
    versions["8k"] = sf.read("out/sentence_8k.wav")[0]
    versions["16k"] = sf.read("out/sentence_16k.wav")[0]
    w = WhisperModel("small.en", device="cpu", compute_type="int8")
    RES["phase"] = {}
    for name, y in versions.items():
        y = y / max(1e-9, np.abs(y).max()) * 0.9
        if name not in ("8k", "16k", "original"):
            sf.write(f"out/sentence_{name}.wav", y, SR)
        text = " ".join(s.text.strip() for s in w.transcribe(resample_poly(y, 2, 3).astype(np.float32))[0])   # Whisper takes 16 kHz
        a = np.abs(stft(y))
        sc = float(np.linalg.norm(a / np.linalg.norm(a) - mag / np.linalg.norm(mag)))             # 0 = same magnitudes
        RES["phase"][name] = {"whisper": text, "word_error_rate": wer(TEXT, text), "spectral_distance": sc}
        print(name, RES["phase"][name], flush=True)
    # short clips of the first sentence, for listening (the video plays them)
    env20 = np.abs(x[:len(x) // 480 * 480]).reshape(-1, 480).max(1)                # 20 ms peaks
    quiet = np.where(env20 < 0.02)[0]
    end = int(quiet[(quiet > 75) & np.r_[np.diff(quiet) == 1, False]][0]) * 480 + 2400   # first pause after 1.5 s
    RES["clip"] = {"seconds": end / SR, "versions": ["original", "8k", "random_phase", "griffin_lim_60"]}
    for name in RES["clip"]["versions"]:
        y = versions[name][:end]
        sf.write(f"out/clip_{name}.wav", y / np.abs(y).max() * 0.6, SR)
    json.dump(RES, open("results.json", "w"), indent=1)
    print(json.dumps({k_: v for k_, v in RES.items() if k_ != "phase"}, indent=1))
