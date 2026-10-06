"""Build StepStone lesson packs (format 1) from the study guides.

    python stepstone/build_packs.py                 # all series → stepstone/dist/
    python stepstone/build_packs.py deep-dive       # one series

Each series becomes one ZIP:
    manifest.json        series metadata + lesson list (with YouTube ids and optional offline-video URLs)
    lessons/<id>.json    intro, prerequisites, concepts (blocks + exercises with answer keys)
    img/<file>.jpg       the guide's video frames
    cover.jpg            series artwork (when available)
and stepstone/dist/catalog.json lists every pack. The format is documented in stepstone/FORMAT.md.
Run with the manim venv (needs bs4); answer keys must pass framework/answer_keys.py check first.
"""
import hashlib
import html
import importlib.util
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "stepstone" / "dist"
FORMAT = 1
PACKS_RELEASE, VIDEOS_RELEASE = "stepstone-packs", "stepstone-videos"   # GitHub releases holding the files
REPO = "https://github.com/luilan/educational-videos"
PAGES = "https://luilan.github.io/educational-videos"
JOBS = Path("/home/codex/secrets/youtube-jobs")                 # upload checkpoints: lesson → YouTube id
OLD_RENDERS = Path("/home/codex/manim-video/llm_series/media/videos")
INLINE = {"b", "i", "code", "sub", "sup", "br"}
RENAME = {"strong": "b", "em": "i"}

SERIES = {
    "foundations": {"prefix": "f", "title": "How LLMs Work: Foundations", "order": 1, "requires": [],
                    "subject": "Machine learning", "cover": "youtube/covers/FoundationsCover.jpg",
                    "description": "The math and code behind language models: vectors, dot products, matrices, "
                                   "nonlinearity, exponentials and logs, probability, softmax, statistics, waves, "
                                   "gradients, NumPy, neural networks, autograd and train/validation data."},
    "how-llms-work": {"prefix": "v", "title": "How LLMs Work", "order": 2, "requires": ["foundations"],
                      "subject": "Machine learning", "cover": "youtube/covers/LLMCover.jpg",
                      "description": "A large language model, built up from scratch: tokens, embeddings, position, "
                                     "attention, the MLP, transformer blocks, probabilities, training, a real tiny GPT, "
                                     "the KV cache and chatbots."},
    "llms-in-practice": {"prefix": "p", "title": "LLMs in Practice", "order": 3, "requires": ["how-llms-work"],
                         "subject": "Machine learning", "cover": "youtube/covers/PracticeCover.jpg",
                         "description": "How real products are built on a language model: prompts and context, "
                                        "sampling, embeddings and search, RAG, tool use, agents, fine-tuning and LoRA, "
                                        "quantization, evaluation and safety."},
    "deep-dive": {"prefix": "d", "title": "How LLMs Work: Deep Dive", "order": 4, "requires": ["how-llms-work"],
                  "subject": "Machine learning", "cover": "youtube/covers/DeepDiveCover.jpg",
                  "description": "Inside a modern LLM, piece by piece: tokenization, position, attention, "
                                 "normalization, training and scaling, mixture of experts, inference tricks, "
                                 "post-training and interpretability."},
}


# ---------------------------------------------------------------------------------------------- HTML → blocks
def inline(node):
    """Sanitised inline HTML: only b, i, code, sub, sup, br survive; everything else is unwrapped."""
    out = []
    for ch in node.children if isinstance(node, Tag) else [node]:
        if isinstance(ch, NavigableString):
            out.append(html.escape(str(ch), quote=False))
        elif isinstance(ch, Tag):
            name = RENAME.get(ch.name, ch.name)
            if name == "br":
                out.append("<br>")
            elif name in INLINE:
                out.append(f"<{name}>{inline(ch)}</{name}>")
            else:
                out.append(inline(ch))
    return "".join(out)


def squash(s):
    return re.sub(r"\s+", " ", s).strip()


def figures_in(text, figs):
    return [figs[int(n)] for n in re.findall(r"\{fig(\d)\}", text)]


def table_block(t):
    rows = [[squash(inline(c)) for c in tr.find_all(["th", "td"])] for tr in t.find_all("tr")]
    head = None
    first = t.find("tr")
    if first and all(c.name == "th" for c in first.find_all(["th", "td"])):
        head, rows = rows[0], rows[1:]
    return {"type": "table", "header": head, "rows": rows}


def blocks(fragment, figs=()):
    """Convert one HTML fragment (a body entry, an intro, a question...) to a list of blocks."""
    soup = BeautifulSoup(fragment, "html.parser")
    out, loose = [], []

    def flush():
        text = squash("".join(loose))
        loose.clear()
        if text:
            out.append({"type": "text", "html": text})

    for node in soup.contents:
        if isinstance(node, NavigableString):
            s = str(node)
            if re.search(r"\{fig\d\}", s):
                flush()
                out.append({"type": "figures", "items": figures_in(s, figs)})
            else:
                loose.append(html.escape(s, quote=False))
            continue
        name = RENAME.get(node.name, node.name)
        if name in INLINE:
            loose.append(f"<{name}>{inline(node)}</{name}>" if name != "br" else "<br>")
            continue
        flush()
        cls = node.get("class") or []
        style = node.get("style", "")
        if name == "p":
            b = {"type": "text", "html": squash(inline(node))}
            if "text-align:center" in style.replace(" ", ""):
                b["align"] = "center"
            out.append(b)
        elif name == "div" and "box" in cls:
            title = node.find("b", class_="t")
            if title:
                title_text = squash(title.get_text())
                title.decompose()
            else:
                title_text = None
            out.append({"type": "callout", "style": "key" if "key" in cls else "note", "title": title_text,
                        "html": squash(inline(node))})
        elif name == "div" and "figrow" in cls:
            out.append({"type": "figures", "items": figures_in(node.get_text(), figs)})
        elif name == "pre":
            out.append({"type": "code", "language": "python", "text": node.get_text().strip("\n")})
        elif name == "table":
            out.append(table_block(node))
        else:                                                      # e.g. hand-built matrices: render as HTML
            out.append({"type": "html", "html": str(node)})
    flush()
    return out


# ---------------------------------------------------------------------------------------------- lessons
def load(path):
    spec = importlib.util.spec_from_file_location("m", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def youtube_ids(series):
    ids = {}
    for f in (JOBS / f"{series}-state.json", JOBS / f"{series}-ledger.json"):
        if f.exists():
            s = json.loads(f.read_text())
            for i, e in enumerate(s.get("episodes", [])):
                if e.get("video_id"):
                    ids[i + 1] = e["video_id"]
            for n, v in (s.get("released") or {}).items():          # release ledger: {"1": "<video id>"}
                if isinstance(v, str) and v:
                    ids[int(n)] = v
    return ids


def render_path(series, vid):
    for base in (ROOT / series / "media" / "videos", OLD_RENDERS):
        hits = sorted((base / f"{vid}_scene" / "1080p60").glob("*.mp4"))
        if hits:
            return hits[0]
    return None


def exercise(e, ci, ei):
    x = {"id": f"c{ci + 1}e{ei + 1}", "kind": e["kind"], "prompt": blocks(e["q"])}
    if e["kind"] == "mc":
        x["options"] = [squash(inline(BeautifulSoup(o, "html.parser"))) for o in e["options"]]
    if e["kind"] == "code":
        x["code"] = e.get("code", "")
    key = e.get("key")
    if e["kind"] in ("short", "code") or key == {"self": True}:
        x["grading"] = "self"
    else:
        x["grading"] = "auto"
        x["key"] = key
    x["answer"] = blocks(e["answer"])
    x["why"] = blocks(e["why"])
    return x


def lesson(series, path, youtube, images):
    m = load(path)
    vid = path.name.split("_")[0]
    number = int(vid[1:])
    concepts = []
    for ci, c in enumerate(m.CONCEPTS):
        figs = []
        for f in c["figures"]:
            name = f"{vid}_{f['t']:07.2f}.jpg"
            src = path.parent / "img" / name
            assert src.exists(), f"missing figure {src}"
            images[name] = src
            figs.append({"src": f"img/{name}", "caption": squash(inline(BeautifulSoup(f["caption"], "html.parser"))),
                         "t": f["t"]})
        body = [b for entry in c["body"] for b in blocks(entry, figs)]
        concepts.append({"id": f"c{ci + 1}", "title": c["title"], "segment": list(c["segment"]), "blocks": body,
                         "exercises": [exercise(e, ci, ei) for ei, e in enumerate(c["exercises"])]})
    L = m.LESSON
    video = {"youtube": youtube.get(number), "offline": None}
    return {"format": FORMAT, "id": vid, "number": number, "label": L["label"], "title": L["title"],
            "tagline": L["tagline"], "duration": L["duration"], "intro": blocks(L["intro"]),
            "prereq": blocks(L["prereq"]), "video": video, "concepts": concepts,
            "links": {"study_pdf": f"{REPO}/blob/main/{series}/study/{vid}_study.pdf",
                      "code": f"{REPO}/tree/main/{series}/code"}}


def build(series, offline=None):
    """offline: {lesson id: {"url", "size", "sha256"}} for videos published as release assets."""
    cfg = SERIES[series]
    youtube = youtube_ids(series)
    images, lessons = {}, []
    for path in sorted((ROOT / series / "study").glob(f"{cfg['prefix']}[0-9][0-9]_study.py")):
        lessons.append(lesson(series, path, youtube, images))
    for L in lessons:
        if offline and L["id"] in offline:
            L["video"]["offline"] = offline[L["id"]]
    manifest = {"format": FORMAT, "id": series, "title": cfg["title"], "description": cfg["description"],
                "subject": cfg["subject"], "language": "en", "order": cfg["order"], "requires": cfg["requires"],
                "source": f"{REPO}/tree/main/{series}", "cover": None,
                "lessons": [{"id": L["id"], "number": L["number"], "title": L["title"], "tagline": L["tagline"],
                             "duration": L["duration"], "file": f"lessons/{L['id']}.json",
                             "concepts": len(L["concepts"]),
                             "exercises": sum(len(c["exercises"]) for c in L["concepts"]),
                             "video": L["video"]} for L in lessons]}
    cover = ROOT / cfg["cover"]
    if cover.exists():
        manifest["cover"] = "cover.jpg"
    content = json.dumps([manifest, lessons], sort_keys=True, ensure_ascii=False).encode()
    manifest["content_hash"] = hashlib.sha256(content + b"".join(p.read_bytes() for _, p in sorted(images.items()))).hexdigest()[:16]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        def put(name, data):                                       # fixed timestamps: same content → same bytes
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if name.endswith(".jpg") else zipfile.ZIP_DEFLATED
            z.writestr(info, data)
        put("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
        for L in lessons:
            put(f"lessons/{L['id']}.json", json.dumps(L, ensure_ascii=False, indent=1))
        for name, p in sorted(images.items()):
            put(f"img/{name}", p.read_bytes())
        if manifest["cover"]:
            put("cover.jpg", cover.read_bytes())
    return manifest, buf.getvalue()


def main(names, offline=None):
    DIST.mkdir(parents=True, exist_ok=True)
    cache = DIST / "offline.json"                               # last published offline-video links (publish.py)
    if offline is None:
        offline = json.loads(cache.read_text()) if cache.exists() else {}
    else:
        cache.write_text(json.dumps(offline, indent=1))
    cat_path = DIST / "catalog.json"
    old = {p["id"]: p for p in json.loads(cat_path.read_text())["packs"]} if cat_path.exists() else {}
    packs = []
    for series in names or sorted(SERIES, key=lambda s: SERIES[s]["order"]):
        manifest, data = build(series, (offline or {}).get(series))
        prev = old.get(series)
        version = prev["version"] + (prev["content_hash"] != manifest["content_hash"]) if prev else 1
        name = f"{series}-v{version}.zip"
        (DIST / name).write_bytes(data)
        packs.append({"id": series, "title": manifest["title"], "description": manifest["description"],
                      "subject": manifest["subject"], "order": manifest["order"], "requires": manifest["requires"],
                      "version": version, "content_hash": manifest["content_hash"], "file": name,
                      "url": f"{REPO}/releases/download/{PACKS_RELEASE}/{name}",
                      "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "lessons": len(manifest["lessons"])})
        print(f"{name}: {len(manifest['lessons'])} lessons, {len(data) / 1e6:.1f} MB, hash {manifest['content_hash']}")
    merged = {**old, **{p["id"]: p for p in packs}}
    cat = {"format": FORMAT, "name": "StepStone catalog · Lui's lessons",
           "packs": sorted(merged.values(), key=lambda p: p["order"])}
    app = ROOT / "stepstone" / "app.json"                      # latest app release, for in-app update notices
    if app.exists():
        cat["app"] = json.loads(app.read_text())
    cat_path.write_text(json.dumps(cat, ensure_ascii=False, indent=1))
    return packs


if __name__ == "__main__":
    main(sys.argv[1:])
