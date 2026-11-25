"""Bitrate against quality: the tiny codec at 1–8 codebooks, Meta's EnCodec and Kyutai's Mimi, on 40 held-out sentences.

    python3 evaluate.py           # ~25 minutes → results.json, tokens.json, out/clip_*.wav, out/mels.png

Quality: words changed (Whisper small.en's transcript of the codec's output against its transcript of the original;
word errors against the script text are kept too), and mel distance (mean absolute difference of log-mel spectrograms, 80 bands,
the same measure for every codec; 0 = identical).
Needs out/codec.pt (codec.py) and data/ (make_data.py). EnCodec and Mimi come from Hugging Face (transformers).
"""
import json
import os
import re

import numpy as np
import soundfile as sf
import torch
from PIL import Image

import codec as C

HERE = os.path.dirname(os.path.abspath(__file__))
SR = C.SR
CLIP = 0                                          # test sentence used for the listening clips and pictures


def words(s):
    """Lower case, no punctuation, numbers spelled out (so "10" and "ten" count as the same word)."""
    from num2words import num2words
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s.lower()).replace("%", " percent")
    s = re.sub(r"\d+(\.\d+)?", lambda m: " " + num2words(float(m.group()) if "." in m.group() else int(m.group())).replace(" and ", " ") + " ", s)
    return re.sub(r"[^a-z0-9' ]", " ", s.replace("-", " ")).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)


def logmel(x):
    return C.log_mel(torch.as_tensor(x, dtype=torch.float32)[None], 1024)[0].numpy()


def mel_distance(x, y):
    n = min(len(x), len(y))
    return float(np.abs(logmel(x[:n]) - logmel(y[:n])).mean())


def colormap(v):                                  # v in [0, 1] → dark blue · purple · orange · yellow
    stops = np.array([[0.02, 0.02, 0.1], [0.35, 0.1, 0.5], [0.85, 0.35, 0.25], [1.0, 0.9, 0.35]])
    p = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(p.astype(int), len(stops) - 2)
    f = (p - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f) * 255


def mel_picture(x, lo):
    m = logmel(x) / np.log(10) * 10           # dB
    v = (m - lo) / 80
    return colormap(v[::-1]).astype(np.uint8)


if __name__ == "__main__":
    from faster_whisper import WhisperModel
    from transformers import AutoTokenizer, EncodecModel, MimiModel

    torch.set_num_threads(14)
    man = json.load(open(os.path.join(HERE, "data/manifest.json")))
    tests = man["test"]
    model = C.Codec()
    model.load_state_dict(torch.load(os.path.join(HERE, "out/codec.pt")))
    model.eval()
    enc = EncodecModel.from_pretrained("facebook/encodec_24khz").eval()
    mimi = MimiModel.from_pretrained("kyutai/mimi").eval()
    whisper = WhisperModel("small.en", device="cpu", compute_type="int8")

    def transcribe(y):
        sf.write("/tmp/g16_eval.wav", y, SR)
        return " ".join(s.text.strip() for s in whisper.transcribe("/tmp/g16_eval.wav", beam_size=5)[0])

    def ours(x, n):
        return model.decode(model.encode(x, n), len(x)).numpy()

    @torch.no_grad()
    def encodec(x, bw):
        t = torch.from_numpy(x)[None, None]
        e = enc.encode(t, bandwidth=bw)
        return enc.decode(e.audio_codes, e.audio_scales)[0][0, 0, :len(x)].numpy(), e.audio_codes.shape[2]

    @torch.no_grad()
    def mimi_codec(x, n):
        t = torch.from_numpy(x)[None, None]
        codes = mimi.encode(t, num_quantizers=n).audio_codes
        return mimi.decode(codes).audio_values[0, 0, :len(x)].numpy(), codes.shape[1]

    @torch.no_grad()
    def unquantized(x):                         # encoder → decoder with no rounding at all: the decoder's limit
        z = model.encoder(torch.nn.functional.pad(torch.from_numpy(x), (0, (-len(x)) % C.HOP))[None])
        return model.decoder(z)[0, :len(x)].numpy()

    systems = [("ours", n, f"{n * 0.75:g}", lambda x, n=n: ours(x, n)) for n in (1, 2, 4, 8)]
    systems += [("ours_unquantized", 0, f"{75 * 64 * 32 / 1000:g}", unquantized)]
    systems += [("encodec", bw, f"{bw:g}", lambda x, bw=bw: encodec(x, bw)[0]) for bw in (1.5, 3.0, 6.0, 12.0)]
    systems += [("mimi", n, f"{n * 11 * 12.5 / 1000:g}", lambda x, n=n: mimi_codec(x, n)[0]) for n in (8, 32)]
    rows = {f"{name}_{k:g}": {"system": name, "setting": k, "kbps": float(kb), "wer": [], "changed": [], "mel": []}
            for name, k, kb, _ in systems}
    rows["original"] = {"system": "original", "setting": 0, "kbps": 384.0, "wer": [], "changed": [], "mel": []}
    clips, transcripts = {}, []
    for i, t in enumerate(tests):
        x, _ = sf.read(os.path.join(HERE, t["file"]), dtype="float32")
        ref = transcribe(x)                     # what Whisper hears in the uncompressed voice
        rows["original"]["wer"].append(wer(t["text"], ref))
        rows["original"]["changed"].append(0.0)
        rows["original"]["mel"].append(0.0)
        transcripts.append({"text": t["text"], "original": ref})
        for name, k, _, f in systems:
            y = f(x)
            key = f"{name}_{k:g}"
            hyp = transcribe(y)
            transcripts[-1][key] = hyp
            rows[key]["wer"].append(wer(t["text"], hyp))
            rows[key]["changed"].append(wer(ref, hyp))
            rows[key]["mel"].append(mel_distance(x, y))
            if i == CLIP:
                clips[key] = y
        if i == CLIP:
            clips["original"] = x
        print(i, {k: round(np.mean(v["changed"]), 3) for k, v in rows.items()}, flush=True)

    summary = {k: {"system": v["system"], "setting": v["setting"], "kbps": v["kbps"],
                   "word_error_rate": float(np.mean(v["wer"])), "mel_distance": float(np.mean(v["mel"])),
                   "words_changed": float(np.mean(v["changed"])),
                   "sentences_unchanged": int(sum(w == 0 for w in v["changed"]))} for k, v in rows.items()}

    # residual vector quantization: how much of the encoder output each extra codebook explains
    remaining = np.zeros(C.N_Q)
    usage = np.zeros((C.N_Q, C.CODES))
    with torch.no_grad():
        for t in tests:
            x, _ = sf.read(os.path.join(HERE, t["file"]), dtype="float32")
            z = model.encoder(torch.nn.functional.pad(torch.from_numpy(x), (0, (-len(x)) % C.HOP))[None])
            r = z[0].T
            for b, book in enumerate(model.rvq.books):
                idx = book.nearest(r)
                usage[b] += np.bincount(idx.numpy(), minlength=C.CODES)
                r = r - book.embed[idx]
                remaining[b] += float((r ** 2).sum() / (z[0] ** 2).sum())
    remaining /= len(tests)
    p = usage / usage.sum(1, keepdims=True)
    entropy = [float(-(q[q > 0] * np.log2(q[q > 0])).sum()) for q in p]

    # the example sentence as tokens
    x = clips["original"]
    codes = model.encode(x, C.N_Q).numpy()
    e_codes = encodec(x, 6.0)[1]
    m_codes = mimi_codec(x, 8)[1]
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    text_tokens = len(tok(tests[CLIP]["text"])["input_ids"])
    json.dump({"text": tests[CLIP]["text"], "seconds": len(x) / SR, "codes": codes.tolist()},
              open(os.path.join(HERE, "tokens.json"), "w"))

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    for k in ("original", "ours_1", "ours_2", "ours_8", "encodec_1.5", "encodec_6", "mimi_8"):
        sf.write(os.path.join(HERE, f"out/clip_{k}.wav"), clips[k], SR)
    lo = float(logmel(x).max() / np.log(10) * 10 - 80)
    pics = [mel_picture(clips[k], lo) for k in ("original", "ours_1", "ours_8", "encodec_6")]
    for k, pic in zip(("original", "ours_1", "ours_8", "encodec_6"), pics):
        Image.fromarray(pic).resize((pic.shape[1] * 2, pic.shape[0] * 3), Image.NEAREST).save(
            os.path.join(HERE, f"out/mel_{k}.png"))

    tl = json.load(open(os.path.join(HERE, "out/train_log.json")))
    results = {
        "codec": {"sample_rate": SR, "frames_per_second": SR / C.HOP, "hop": C.HOP, "latent_dim": C.DIM,
                  "codebooks": C.N_Q, "codebook_size": C.CODES, "bits_per_token": int(np.log2(C.CODES)),
                  "kbps_per_codebook": SR / C.HOP * np.log2(C.CODES) / 1000,
                  "encoder_parameters": tl["encoder_parameters"], "decoder_parameters": tl["decoder_parameters"],
                  "train_steps": tl["steps"], "train_minutes_of_audio": tl["train_minutes"],
                  "train_wall_minutes": tl["log"][-1]["minutes"]},
        "data": {"train_sentences": man["train_sentences"], "voices": man["voices"],
                 "test_sentences": len(tests), "test_voice": "af_heart"},
        "systems": summary,
        "rvq_remaining_energy": remaining.tolist(),
        "codebook_entropy_bits": entropy,
        "example": {"text": tests[CLIP]["text"], "seconds": len(x) / SR, "samples": len(x),
                    "text_tokens_gpt2": text_tokens, "ours_frames": codes.shape[1],
                    "ours_tokens": int(codes.size), "encodec_6k_codebooks": int(e_codes),
                    "encodec_frames": int(np.ceil(len(x) / 320)), "mimi_codebooks": int(m_codes),
                    "mimi_frames": int(np.ceil(len(x) / 1920))},
        "parameters": {"encodec": sum(p.numel() for p in enc.parameters()),
                       "mimi": sum(p.numel() for p in mimi.parameters())},
    }
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    json.dump(transcripts, open(os.path.join(HERE, "transcripts.json"), "w"), indent=1)
    for k, v in summary.items():
        print(f"{k:12s} {v['kbps']:7.2f} kbit/s  word errors {v['word_error_rate'] * 100:5.1f}%  "
              f"changed {v['words_changed'] * 100:5.1f}%  mel {v['mel_distance']:.3f}  unchanged {v['sentences_unchanged']}/{len(tests)}")
    print("remaining energy after each codebook:", np.round(remaining, 3))
