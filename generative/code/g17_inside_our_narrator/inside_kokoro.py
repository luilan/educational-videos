"""Inside Kokoro, the 82-million-parameter text-to-speech model that narrates this series.

    python3 inside_kokoro.py      # ~5 minutes on a CPU → results.json, curves.json, out/*.wav

Runs Kokoro v1.0 (hexgrad/Kokoro-82M, PyTorch weights from Hugging Face) one stage at a time, the way
kokoro.model.KModel.forward_with_tokens does, and keeps every intermediate prediction:

  text → phonemes → PL-BERT + duration predictor → frames per phoneme → pitch (F0) and energy curves
       → decoder (iSTFTNet vocoder) → 24,000 samples per second

Then it edits the predictions before the decoder (flat pitch, higher pitch, even durations, another voice's sound)
and lets Whisper small.en transcribe every version.

Phonemes come from espeak-ng through kokoro-onnx (the same front end that voices this series).
"""
import importlib.util
import json
import os
import re
import sys
import time
import types

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KOKORO_DIR = os.environ.get("KOKORO_DIR", os.path.join(HERE, "../../../tts"))
SR = 24000
FRAME = 600                       # samples per duration frame: 25 ms
TEXT = "Our narrator turns text into speech in three steps."
QUESTION = "Can you hear my voice go up at the end of a question?"
VOICE, OTHER = "af_heart", "am_michael"


def load_kmodel():
    """kokoro.model without kokoro/__init__.py (which pulls in the English G2P and spaCy)."""
    spec = importlib.util.find_spec("kokoro")
    if "kokoro" not in sys.modules:
        pkg = types.ModuleType("kokoro")
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules["kokoro"] = pkg
    from kokoro.model import KModel
    from loguru import logger
    logger.disable("kokoro")                                             # no per-sentence debug lines
    return KModel(repo_id="hexgrad/Kokoro-82M").eval()


def voice_pack(name):
    from huggingface_hub import hf_hub_download
    return torch.load(hf_hub_download("hexgrad/Kokoro-82M", f"voices/{name}.pt"))      # (510, 1, 256)


@torch.no_grad()
def synth(m, ids, ref, speed=1.0, edit_dur=None, edit_f0=None, sound_ref=None, timing=None):
    """KModel.forward_with_tokens, one stage at a time. ref: (1, 256) style; [:128] = sound, [128:] = prosody."""
    torch.manual_seed(0)                                                 # the vocoder's noise source is random
    ids = torch.LongTensor([[0, *ids, 0]])
    n = ids.shape[1]
    lengths = torch.LongTensor([n])
    mask = torch.zeros(1, n, dtype=torch.bool)
    t = [time.perf_counter()]
    s = ref[:, 128:]
    bert = m.bert(ids, attention_mask=(~mask).int())                     # PL-BERT reads the phonemes in context
    d_en = m.bert_encoder(bert).transpose(-1, -2)
    t.append(time.perf_counter())
    d = m.predictor.text_encoder(d_en, s, lengths, mask)
    x, _ = m.predictor.lstm(d)
    dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1) / speed     # frames per phoneme
    dur = torch.round(dur).clamp(min=1).long().squeeze()
    if edit_dur is not None:
        dur = edit_dur(dur)
    idx = torch.repeat_interleave(torch.arange(n), dur)
    aln = torch.zeros(n, len(idx))
    aln[idx, torch.arange(len(idx))] = 1                                 # phoneme i fills dur[i] frames
    en = d.transpose(-1, -2) @ aln[None]
    f0, energy = m.predictor.F0Ntrain(en, s)                             # two curves, 80 values per second
    if edit_f0 is not None:
        f0 = edit_f0(f0)
    t.append(time.perf_counter())
    t_en = m.text_encoder(ids, lengths, mask)                            # what to say, one vector per phoneme
    asr = t_en @ aln[None]
    t.append(time.perf_counter())
    audio = m.decoder(asr, f0, energy, (sound_ref if sound_ref is not None else ref)[:, :128]).squeeze()
    t.append(time.perf_counter())
    if timing is not None:
        timing.update(bert=t[1] - t[0], predictor=t[2] - t[1], text_encoder=t[3] - t[2], decoder=t[4] - t[3])
    return {"audio": audio.numpy(), "dur": dur.numpy(), "f0": f0[0].numpy(), "energy": energy[0].numpy()}


def pitch_track(x, hop=240, F=960):
    """Autocorrelation pitch on 40 ms frames (as in episode 15); 0 = silent or unvoiced."""
    lags = np.arange(SR // 400, SR // 70)
    out = []
    for i in range(0, len(x) - F, hop):
        fr = x[i:i + F] - x[i:i + F].mean()
        if (fr ** 2).mean() < 0.1 * (x ** 2).mean():
            out.append(0.0)
            continue
        ac = np.correlate(fr, fr, "full")[F - 1:]
        ac = ac / ac[0]
        best = lags[np.argmax(ac[lags])]
        out.append(SR / best if ac[best] > 0.5 else 0.0)
    return np.array(out)


def words(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ")).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)


def word_times(phonemes, dur):
    """Start and end (seconds) of each word, from the predicted durations. Spaces separate words."""
    frames = np.concatenate([[0], np.cumsum(dur)])          # dur[0] is the start token
    out, start, cur = [], None, ""
    for i, p in enumerate(phonemes):
        k = i + 1                                           # position in dur (after the start token)
        if p == " " or p in ".,?!;:":
            if cur:
                out.append({"phonemes": cur, "start": frames[start] * FRAME / SR, "end": frames[k] * FRAME / SR})
            cur, start = "", None
        else:
            if start is None:
                start = k
            cur += p
    return out


if __name__ == "__main__":
    from faster_whisper import WhisperModel
    from kokoro_onnx import Kokoro

    torch.set_num_threads(int(os.environ.get("THREADS", 8)))
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    m = load_kmodel()
    k = Kokoro(f"{KOKORO_DIR}/kokoro-v1.0.onnx", f"{KOKORO_DIR}/voices-v1.0.bin")
    whisper = WhisperModel("small.en", device="cpu", compute_type="int8")

    def transcribe(y):
        sf.write("/tmp/g17_eval.wav", y, SR)
        return " ".join(s.text.strip() for s in whisper.transcribe("/tmp/g17_eval.wav", beam_size=5)[0])

    R = {}
    params = {name: sum(p.numel() for p in c.parameters()) for name, c in m.named_children()}
    R["parameters"] = dict(params, total=sum(params.values()))
    R["decoder_parts"] = {"generator": sum(p.numel() for p in m.decoder.generator.parameters()),
                          "harmonics": m.decoder.generator.m_source.l_sin_gen.harmonic_num + 1,
                          "istft_n_fft": m.decoder.generator.post_n_fft,
                          "istft_hop": 5, "upsample": [10, 6]}

    # 1. phonemes
    ph = k.tokenizer.phonemize(TEXT, "en-us")
    ids = [m.vocab[p] for p in ph if p in m.vocab]
    R["phonemes"] = {"text": TEXT, "phonemes": ph, "symbols": len(ph), "ids": ids, "vocab_used": len(m.vocab),
                     "vocab_slots": 178, "letters": len(TEXT)}
    for w in ("though", "tough", "through", "thought"):
        R["phonemes"].setdefault("spelling", {})[w] = k.tokenizer.phonemize(w, "en-us")

    # 2. the voice: a table of style vectors, one per input length
    pack, other = voice_pack(VOICE), voice_pack(OTHER)
    ref = pack[len(ids) - 1]
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0))
    R["voice"] = {"shape": list(pack.shape), "row_used": len(ids) - 1,
                  "same_voice_len10_vs_len200": cos(pack[10], pack[200]),
                  "sound_half_vs_other_voice": cos(pack[len(ids) - 1][:, :128], other[len(ids) - 1][:, :128]),
                  "prosody_half_vs_other_voice": cos(pack[len(ids) - 1][:, 128:], other[len(ids) - 1][:, 128:]),
                  "onnx_voice_equal": bool(np.allclose(k.voices[VOICE], pack.numpy(), atol=1e-6))}

    # 3. one full pass, every intermediate kept
    timing = {}
    synth(m, ids, ref)                                       # warm-up
    base = synth(m, ids, ref, timing=timing)
    a = base["audio"]
    R["timing_s"] = timing
    R["audio"] = {"samples": len(a), "seconds": len(a) / SR, "frames": int(base["dur"].sum()),
                  "samples_per_frame": len(a) / int(base["dur"].sum()),
                  "f0_values": len(base["f0"]), "f0_per_second": len(base["f0"]) / (len(a) / SR),
                  "real_time_factor": len(a) / SR / sum(timing.values())}
    onnx_a, _ = k.create(TEXT, voice=VOICE, speed=1.0, lang="en-us")
    # kokoro-onnx trims the silence at both ends; line the two up before comparing
    off = max(range(0, len(a) - len(onnx_a) + 1, 1), key=lambda o: float(np.dot(a[o:o + len(onnx_a)], onnx_a)))
    R["audio"]["onnx_samples"] = len(onnx_a)
    R["audio"]["onnx_offset"] = off
    R["audio"]["onnx_correlation"] = float(np.corrcoef(onnx_a, a[off:off + len(onnx_a)])[0, 1])
    dur = base["dur"]
    labels = ["(start)"] + list(ph) + ["(end)"]
    R["durations"] = {"frames": dur.tolist(), "labels": labels, "frame_ms": FRAME / SR * 1000,
                      "longest_inside": labels[int(np.argmax(dur[1:-1])) + 1], "longest_inside_frames": int(dur[1:-1].max()),
                      "mean_frames_inside": float(dur[1:-1].mean())}
    R["words"] = word_times(ph, dur)
    f0 = base["f0"]
    voiced = f0[f0 > 10]
    R["pitch"] = {"median_hz": float(np.median(voiced)), "min_hz": float(voiced.min()), "max_hz": float(voiced.max()),
                  "voiced_fraction": float((f0 > 10).mean())}
    R["energy"] = {"min": float(base["energy"].min()), "max": float(base["energy"].max())}

    # the harmonic source the vocoder starts from: sine waves at the predicted pitch
    with torch.no_grad():
        g = m.decoder.generator
        f0_up = g.f0_upsamp(torch.from_numpy(f0)[None, None]).transpose(1, 2)
        har, _, _ = g.m_source(f0_up)
    har = har[0, :, 0].numpy()

    # 4. edits before the decoder
    med = float(np.median(voiced))
    edits = {
        "original": {},
        "flat_pitch": {"edit_f0": lambda f: torch.where(f > 10, torch.full_like(f, med), f)},
        "higher_pitch": {"edit_f0": lambda f: f * 1.5},
        "even_durations": {"edit_dur": lambda d: torch.cat([d[:1], torch.full_like(d[1:-1], int(round(float(d[1:-1].float().mean())))), d[-1:]])},
        "faster": {"speed": 1.4},
        "other_voice_sound": {"sound_ref": other[len(ids) - 1]},
        "other_voice_all": {"ref_override": other[len(ids) - 1]},
    }
    R["edits"], audio = {}, {}
    for name, kw in edits.items():
        kw = dict(kw)
        r = kw.pop("ref_override", ref)
        out = synth(m, ids, r, **kw)
        y = out["audio"]
        p = pitch_track(y)
        hyp = transcribe(y)
        R["edits"][name] = {"seconds": len(y) / SR, "frames": int(out["dur"].sum()),
                            "measured_median_pitch_hz": float(np.median(p[p > 0])),
                            "measured_pitch_spread_hz": float(np.percentile(p[p > 0], 90) - np.percentile(p[p > 0], 10)),
                            "whisper": hyp, "word_error_rate": wer(TEXT, hyp)}
        audio[name] = y
        sf.write(os.path.join(HERE, f"out/{name}.wav"), y, SR)
        print(name, R["edits"][name], flush=True)

    # 5. a question: the pitch curve rises at the end
    qph = k.tokenizer.phonemize(QUESTION, "en-us")
    qids = [m.vocab[p] for p in qph if p in m.vocab]
    q = synth(m, qids, pack[len(qids) - 1])
    qv = q["f0"][q["f0"] > 10]
    tail = q["f0"][-int(0.25 * len(q["f0"])):]
    R["question"] = {"text": QUESTION, "phonemes": qph, "median_hz": float(np.median(qv)),
                     "max_last_quarter_hz": float(tail.max()), "whisper": transcribe(q["audio"])}
    sf.write(os.path.join(HERE, "out/question.wav"), q["audio"], SR)

    json.dump({"style": ref[0].tolist(), "other_style": other[len(ids) - 1][0].tolist(), "f0": f0.tolist(), "energy": base["energy"].tolist(), "question_f0": q["f0"].tolist(),
               "flat_f0": [med if v > 10 else float(v) for v in f0], "higher_f0": (f0 * 1.5).tolist(),
               "harmonic_source": har[int(1.0 * SR):int(1.0 * SR) + 720].tolist(),
               "audio": a[int(1.0 * SR):int(1.0 * SR) + 720].tolist(),
               "envelope": np.abs(a[:len(a) // 400 * 400]).reshape(-1, 400).max(1).tolist()},
              open(os.path.join(HERE, "curves.json"), "w"))
    json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k_: R[k_] for k_ in ("parameters", "audio", "pitch", "voice", "timing_s")}, indent=1))
