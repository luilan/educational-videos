"""Inside Whisper: from log-mel input to words with timestamps, on this series' own narration.

    python3 whisper_inside.py     # ~25 minutes on a CPU → results.json, attention.json, out/*.png

1. One sentence through openai/whisper-small.en step by step (Hugging Face transformers): the 30-second log-mel input,
   the encoder's 1,500 vectors, greedy decoding token by token, and the special tokens around the text.
2. Word timestamps the way Whisper computes them: cross-attention of 19 "alignment heads", normalized, smoothed, then
   dynamic time warping (DTW) to find the monotonic path from tokens to audio frames.
3. How exact are those timestamps? Kokoro tells us where every word really starts (its predicted durations), so we can
   score faster-whisper's word timestamps, the ones that time every animation in this series.
4. Word errors on this series' narration (episodes 1–17), for whisper tiny.en and small.en; noise; silence.

Needs the Kokoro model files (KOKORO_DIR, default ../../../tts) for the phonemizer, and the series' narration in
../../voice (made by ./narrate.sh). Models come from Hugging Face.
"""
import difflib
import glob
import importlib.util
import json
import os
import re
import sys
import types
from collections import Counter

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from scipy.ndimage import median_filter
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES = os.path.abspath(os.path.join(HERE, "../.."))
KOKORO_DIR = os.environ.get("KOKORO_DIR", os.path.join(HERE, "../../../tts"))
SR, FRAME = 24000, 600
EXAMPLE = "Whisper listens to thirty seconds of sound and writes down every word it hears."


# ----------------------------------------------------------------------------- text helpers
def words(s):
    """Lower case, no punctuation, numbers spelled out (so "10" and "ten" count as the same word)."""
    from num2words import num2words
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s.lower()).replace("%", " percent")
    s = re.sub(r"\d+(\.\d+)?", lambda m: " " + num2words(float(m.group()) if "." in m.group() else int(m.group())).replace(" and ", " ") + " ", s)
    return re.sub(r"[^a-z0-9' ]", " ", s.replace("-", " ")).split()


def edits(ref, hyp):
    """Word error rate and the substitutions, deletions and insertions behind it."""
    r, h = words(ref), words(hyp)
    sm = difflib.SequenceMatcher(a=r, b=h, autojunk=False)
    errs, subs = 0, []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            errs += max(i2 - i1, j2 - j1)
            subs.append((" ".join(r[i1:i2]), " ".join(h[j1:j2])))
    return errs / max(1, len(r)), subs


# ----------------------------------------------------------------------------- Kokoro: audio with known word times
def load_kokoro():
    spec = importlib.util.find_spec("kokoro")
    if "kokoro" not in sys.modules:
        pkg = types.ModuleType("kokoro")
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules["kokoro"] = pkg
    from huggingface_hub import hf_hub_download
    from kokoro.model import KModel
    from loguru import logger
    logger.disable("kokoro")                                             # no per-sentence debug lines
    from kokoro_onnx import Kokoro
    m = KModel(repo_id="hexgrad/Kokoro-82M").eval()
    pack = torch.load(hf_hub_download("hexgrad/Kokoro-82M", "voices/af_heart.pt"))
    k = Kokoro(f"{KOKORO_DIR}/kokoro-v1.0.onnx", f"{KOKORO_DIR}/voices-v1.0.bin")
    return m, pack, k


@torch.no_grad()
def speak_with_times(kok, text):
    """Kokoro audio plus the true start of every word (from its predicted phoneme durations)."""
    m, pack, k = kok
    torch.manual_seed(0)                                                 # the vocoder's noise source is random
    ph = k.tokenizer.phonemize(text, "en-us")
    keep = [p for p in ph if p in m.vocab]
    ids = [m.vocab[p] for p in keep]
    out = m(keep and "".join(keep), pack[len(ids) - 1], return_output=True)
    dur = out.pred_dur.numpy()
    frames = np.concatenate([[0], np.cumsum(dur)])
    starts, cur = [], None
    for i, p in enumerate(keep):
        if p == " " or p in ".,?!;:—":
            cur = None
        elif cur is None:
            cur = i + 1
            starts.append(frames[i + 1] * FRAME / SR)
    return out.audio.numpy(), starts, ph


# ----------------------------------------------------------------------------- Whisper internals
def dtw(cost):
    """Cheapest monotonic path through cost (tokens × frames): steps down, right, or diagonal."""
    N, M = cost.shape
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0
    trace = np.zeros((N + 1, M + 1), int)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            c = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            t = int(np.argmin(c))
            D[i, j] = cost[i - 1, j - 1] + c[t]
            trace[i, j] = t
    i, j, path = N, M, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        t = trace[i, j]
        i, j = (i - 1, j - 1) if t == 0 else (i - 1, j) if t == 1 else (i, j - 1)
    return path[::-1]


@torch.no_grad()
def inside(model, proc, audio16):
    """Greedy decoding with cross-attention kept; word start times by DTW, as in openai/whisper timing.py."""
    tok = proc.tokenizer
    feats = proc(audio16, sampling_rate=16000, return_tensors="pt").input_features          # (1, 80, 3000)
    enc = model.model.encoder(feats).last_hidden_state                                     # (1, 1500, 768)
    sot = [tok.convert_tokens_to_ids("<|startoftranscript|>"), tok.convert_tokens_to_ids("<|notimestamps|>")]
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    ids, steps = list(sot), []
    while len(ids) < 100:
        out = model.model.decoder(input_ids=torch.tensor([ids]), encoder_hidden_states=enc)
        logits = model.proj_out(out.last_hidden_state[0, -1])
        p = torch.softmax(logits, -1)
        top = torch.topk(p, 3)
        nxt = int(top.indices[0])
        steps.append({"token": tok.decode([nxt]), "p": float(top.values[0]),
                      "alternatives": [[tok.decode([int(i)]), float(v)] for v, i in zip(top.values[1:], top.indices[1:])]})
        ids.append(nxt)
        if nxt == eot:
            break
    out = model.model.decoder(input_ids=torch.tensor([ids]), encoder_hidden_states=enc, output_attentions=True)
    heads = model.generation_config.alignment_heads
    n_frames = int(np.ceil(len(audio16) / 16000 * 50))                                    # 50 encoder frames per second
    w = torch.stack([out.cross_attentions[l][0, h] for l, h in heads])                    # (19, tokens, 1500)
    w = w[:, :, :n_frames]
    std, mean = torch.std_mean(w, dim=-2, keepdim=True)
    w = ((w - mean) / std).numpy()
    w = median_filter(w, (1, 1, 7))
    # row t of the attention is the step that PREDICTS token t + 1: take the rows that produced the text tokens
    matrix = w.mean(0)[len(sot) - 1:-2]
    path = dtw(-matrix.astype(np.float64))
    first = {}
    for i, j in path:
        first.setdefault(i, j)
    text_ids = ids[len(sot):-1]
    pieces = [tok.decode([t]) for t in text_ids]
    word_starts, wlist = [], []
    for i, pc in enumerate(pieces):
        if pc.startswith(" ") or i == 0:
            wlist.append(pc.strip())
            word_starts.append(first[i] / 50)
        else:
            wlist[-1] += pc
    return {"features": feats[0].numpy(), "encoder_shape": list(enc.shape), "ids": ids, "steps": steps,
            "pieces": pieces, "matrix": matrix, "path": path, "words": wlist, "starts": word_starts,
            "text": tok.decode(text_ids).strip()}


def to16k(x):
    return resample_poly(x, 2, 3).astype(np.float32)


def colormap(v):
    stops = np.array([[0.02, 0.02, 0.1], [0.35, 0.1, 0.5], [0.85, 0.35, 0.25], [1.0, 0.9, 0.35]])
    p = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(p.astype(int), len(stops) - 2)
    f = (p - i)[..., None]
    return ((stops[i] * (1 - f) + stops[i + 1] * f) * 255).astype(np.uint8)


def match_starts(ref_text, ref_starts, hyp_words):
    """Pair reference words with Whisper words that read the same; return (true start, whisper start) pairs."""
    r = words(ref_text)
    if len(r) != len(ref_starts):
        return []
    h = [(words(w.word)[0] if words(w.word) else "", w.start) for w in hyp_words]
    sm = difflib.SequenceMatcher(a=r, b=[x[0] for x in h], autojunk=False)
    pairs = []
    for a, b, n in sm.get_matching_blocks():
        for t in range(n):
            pairs.append((ref_starts[a + t], h[b + t][1]))
    return pairs


if __name__ == "__main__":
    from faster_whisper import WhisperModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(int(os.environ.get("THREADS", 14)))
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    R = {}
    kok = load_kokoro()
    proc = WhisperProcessor.from_pretrained("openai/whisper-small.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en", attn_implementation="eager").eval()
    cfg = model.config
    R["model"] = {"parameters": sum(p.numel() for p in model.parameters()),
                  "encoder_parameters": sum(p.numel() for p in model.model.encoder.parameters()),
                  "decoder_parameters": sum(p.numel() for p in model.model.decoder.parameters()),
                  "layers": [cfg.encoder_layers, cfg.decoder_layers], "width": cfg.d_model,
                  "heads": cfg.encoder_attention_heads, "vocab": cfg.vocab_size, "mel_bands": cfg.num_mel_bins,
                  "alignment_heads": len(model.generation_config.alignment_heads),
                  "conv": [[c.in_channels, c.out_channels, c.kernel_size[0], c.stride[0]]
                           for c in (model.model.encoder.conv1, model.model.encoder.conv2)]}

    # 1–2. the example sentence, step by step
    a24, true_starts, ph = speak_with_times(kok, EXAMPLE)
    sf.write(os.path.join(HERE, "out/example.wav"), a24, SR)
    a16 = to16k(a24)
    ins = inside(model, proc, a16)
    f = ins["features"]
    R["input"] = {"seconds": len(a16) / 16000, "samples_16k": len(a16), "mel_shape": list(f.shape),
                  "frames_with_sound": int(np.ceil(len(a16) / 160)), "padded_to_seconds": 30,
                  "min": float(f.min()), "max": float(f.max())}
    R["encoder"] = {"shape": ins["encoder_shape"], "ms_per_vector": 20}
    R["decoding"] = {"ids": ins["ids"], "tokens": [proc.tokenizer.decode([i]) for i in ins["ids"]],
                     "steps": ins["steps"], "text": ins["text"], "word_error_rate": edits(EXAMPLE, ins["text"])[0]}
    R["timestamps_example"] = {"words": ins["words"], "whisper_dtw_start": ins["starts"], "kokoro_start": true_starts,
                               "errors_ms": [1000 * (a - b) for a, b in zip(ins["starts"], true_starts)]
                               if len(ins["starts"]) == len(true_starts) else None}
    gen = model.generate(proc(a16, sampling_rate=16000, return_tensors="pt").input_features, return_timestamps=True)
    R["timestamp_tokens"] = proc.tokenizer.decode(gen[0], decode_with_timestamps=True)
    # pictures: the 30-second input and the attention matrix with its DTW path
    v = (f - f.min()) / (f.max() - f.min())
    Image.fromarray(colormap(v[::-1])).resize((1500, 240), Image.BILINEAR).save(os.path.join(HERE, "out/input_30s.png"))
    n_sound = R["input"]["frames_with_sound"]
    Image.fromarray(colormap(v[::-1, :n_sound + 20])).resize((4 * (n_sound + 20), 240), Image.NEAREST).save(
        os.path.join(HERE, "out/input_sound.png"))
    mtx = ins["matrix"]
    mv = (mtx - np.percentile(mtx, 5)) / (np.percentile(mtx, 99.5) - np.percentile(mtx, 5))
    json.dump({"matrix": np.round(np.clip(mv, 0, 1), 3).tolist(), "path": ins["path"], "pieces": ins["pieces"],
               "frames": mtx.shape[1]}, open(os.path.join(HERE, "attention.json"), "w"))

    if os.environ.get("EXAMPLE_ONLY"):                                  # quick run: parts 1–2 only
        json.dump(R, open(os.path.join(HERE, "results_example.json"), "w"), indent=1, ensure_ascii=False)
        sys.exit(0)

    # 3. timestamp accuracy on 40 sentences from this series' scripts
    rng = np.random.default_rng(0)
    pool = []
    for p in sorted(glob.glob(os.path.join(SERIES, "g*_script.py"))):
        spec = importlib.util.spec_from_file_location("s", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for sec in mod.SECTIONS:
            pool += [s for s in re.split(r"(?<=[.!?])\s+", sec) if 6 <= len(s.split()) <= 25 and not re.search(r"\d", s)]
    sents = [pool[i] for i in rng.choice(len(pool), 40, replace=False)]
    small = WhisperModel("small.en", device="cpu", compute_type="int8")
    tiny = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    errs, kept = [], 0
    for s in sents:
        a, st, _ = speak_with_times(kok, s)
        sf.write("/tmp/g18_ts.wav", a, SR)
        segs, _ = small.transcribe("/tmp/g18_ts.wav", word_timestamps=True)
        hw = [w for sg in segs for w in sg.words]
        pairs = match_starts(s, st, hw)
        kept += len(pairs)
        errs += [1000 * (h - t) for t, h in pairs]
    e = np.abs(errs)
    R["timestamp_accuracy"] = {"sentences": len(sents), "words": kept, "median_abs_ms": float(np.median(e)),
                               "mean_signed_ms": float(np.mean(errs)), "within_100ms": float((e <= 100).mean()),
                               "within_200ms": float((e <= 200).mean()), "p90_abs_ms": float(np.percentile(e, 90)),
                               "histogram_ms": np.histogram(np.clip(errs, -400, 400), bins=16, range=(-400, 400))[0].tolist()}
    print("timestamps", R["timestamp_accuracy"], flush=True)

    # 4. word errors on this series' narration
    narr = []
    for p in sorted(glob.glob(os.path.join(SERIES, "g*_script.py"))):
        vid = os.path.basename(p)[:3]
        spec = importlib.util.spec_from_file_location("s", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for i, sec in enumerate(mod.SECTIONS, 1):
            wav = os.path.join(SERIES, "voice", vid, f"s{i}.wav")
            if os.path.exists(wav):
                narr.append((vid, sec, wav))
    R["narration"] = {"sections": len(narr), "episodes": len({v for v, _, _ in narr}),
                      "words": sum(len(words(s)) for _, s, _ in narr)}
    for name, wm in (("tiny.en", tiny), ("small.en", small)):
        tot_err, tot_words, subs = 0, 0, Counter()
        for vid, sec, wav in narr:
            hyp = " ".join(sg.text.strip() for sg in wm.transcribe(wav, beam_size=5)[0])
            rate, sb = edits(sec, hyp)
            n = len(words(sec))
            tot_err += rate * n
            tot_words += n
            subs.update(f"{a} → {b}" for a, b in sb)
        R["narration"][name] = {"word_error_rate": tot_err / tot_words, "top_errors": subs.most_common(15)}
        print(name, R["narration"][name]["word_error_rate"], subs.most_common(8), flush=True)

    # noise: the 40 sentences again with white noise at falling signal-to-noise ratios
    R["noise"] = {}
    rng = np.random.default_rng(1)
    clean = [speak_with_times(kok, s)[0] for s in sents[:20]]
    for snr in (None, 20, 10, 5, 0, -5):
        tot = 0.0
        for s, a in zip(sents[:20], clean):
            y = a if snr is None else a + rng.normal(0, np.sqrt((a ** 2).mean() / 10 ** (snr / 10)), len(a))
            sf.write("/tmp/g18_n.wav", y.astype(np.float32), SR)
            if s is sents[0]:
                sf.write(os.path.join(HERE, f"out/noise_{'clean' if snr is None else snr}.wav"), y.astype(np.float32), SR)
            hyp = " ".join(sg.text.strip() for sg in small.transcribe("/tmp/g18_n.wav", beam_size=5)[0])
            tot += edits(s, hyp)[0]
        R["noise"]["clean" if snr is None else str(snr)] = tot / 20
        print("snr", snr, tot / 20, flush=True)

    # nothing to hear: silence and pure noise
    R["no_speech"] = {}
    for name, y in (("silence", np.zeros(10 * SR, np.float32)),
                    ("noise", np.random.default_rng(2).normal(0, 0.05, 10 * SR).astype(np.float32))):
        sf.write("/tmp/g18_x.wav", y, SR)
        for vad in (False, True):
            segs, info = small.transcribe("/tmp/g18_x.wav", beam_size=5, vad_filter=vad)
            segs = list(segs)
            R["no_speech"][f"{name}{'_vad' if vad else ''}"] = {
                "text": " ".join(s.text.strip() for s in segs),
                "no_speech_prob": [round(s.no_speech_prob, 3) for s in segs]}
    print(R["no_speech"])
    json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False)
