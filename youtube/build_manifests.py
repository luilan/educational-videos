"""Build the YouTube upload manifests (uploader/series.py format) for both series.

    python youtube/build_manifests.py

Writes youtube/foundations.json and youtube/how-llms-work.json: one playlist each, one entry per video with
title, description (summary, chapters, series links, credits, AI disclosure), Education category, public (user decision 2026-10-03).
Chapters come from each study guide's concept segments; YouTube ignores all chapters of a video unless the
first starts at 0:00 and every chapter lasts at least 10 seconds, so a too-short chapter absorbs the next one.
"""
import glob
import html
import importlib.util
import json
import os
import re

import av

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDERS = "/home/codex/manim-video/llm_series/media/videos"
MIN_CHAPTER = 10
CATEGORY_EDUCATION = "27"
REPO = "https://github.com/luilan/educational-videos"

CREDITS = ("Credits\n"
           "Animated with Manim Community Edition (https://www.manim.community), the community-maintained "
           "version of Manim, the math animation engine created by Grant Sanderson (3Blue1Brown).")
TINY_SHAKESPEARE = "Training text: Tiny Shakespeare, from Andrej Karpathy's char-rnn project."
AI_DISCLOSURE = ("Made with AI assistance\n"
                 "The script, animation code and examples in this video were created with the help of AI "
                 "(Claude, by Anthropic). The narration is an AI-generated voice (Kokoro text-to-speech).")

SERIES = {
    "foundations": {
        "prefix": "f", "label": "How LLMs Work: Foundations",
        "playlist": ("How LLMs Work: Foundations",
                     "Short videos on the math and code behind large language models: vectors, dot products, "
                     "matrices, nonlinearity, exponentials and logs, probability, softmax, statistics, waves, "
                     "gradients, NumPy, neural networks, autograd and train/validation data. Each video points "
                     "to the How LLMs Work episodes that use it."),
    },
    "how-llms-work": {
        "prefix": "v", "label": "How LLMs Work",
        "playlist": ("How LLMs Work",
                     "A large language model, built up from scratch: tokens, embeddings, position, attention, "
                     "the MLP, transformer blocks, probabilities, training, a real tiny GPT, and two bonus "
                     "episodes on the KV cache and on turning a GPT into a chatbot. New to the math? Start with "
                     "the How LLMs Work: Foundations playlist."),
    },
}


def load(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plain(text):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", text))).strip()


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def chapters(concepts, duration):
    marks = [[c["segment"][0], c["title"]] for c in concepts]
    marks[0][0] = 0  # the ~8 s title intro joins the first concept
    merged = []
    for start, title in marks:
        if merged and start - merged[-1][0] < MIN_CHAPTER:
            merged[-1][1] += " · " + title  # previous chapter too short: it absorbs this one
        else:
            merged.append([start, title])
    while len(merged) > 1 and duration - merged[-1][0] < MIN_CHAPTER:
        last = merged.pop()
        merged[-1][1] += " · " + last[1]
    assert len(merged) >= 3 and merged[0][0] == 0
    return [f"{mmss(start)} {title}" for start, title in merged]


def foundations_index():
    """Main episode -> [(F number, title)] from the Foundations scripts' USED_IN."""
    uses = {}
    for n in range(1, 15):
        script = load(os.path.join(ROOT, "foundations", f"f{n:02d}_script.py"))
        for episode, _ in script.USED_IN:
            uses.setdefault(episode, []).append((n, script.TITLE))
    return uses


def build(series):
    cfg = SERIES[series]
    background = foundations_index() if series == "how-llms-work" else {}
    episodes = []
    for n in range(1, 15):
        vid = f"{cfg['prefix']}{n:02d}"
        script = load(os.path.join(ROOT, series, f"{vid}_script.py"))
        study = load(os.path.join(ROOT, series, "study", f"{vid}_study.py"))
        (video,) = glob.glob(os.path.join(RENDERS, f"{vid}_scene", "1080p60", "*.mp4"))
        with av.open(video) as container:
            duration = float(container.duration / av.time_base)

        if series == "how-llms-work":
            tag = f"Ep. {n}" + (" (bonus)" if n >= 13 else "")
            title = f"{script.TITLE} | How LLMs Work, {tag}"
            where = f"Episode {n} of the How LLMs Work series" + (" (bonus episode)." if n >= 13 else ".")
            refs = background.get(n)
            if refs:
                items = [f"F{f:02d} ({t})" for f, t in refs]
                listed = ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else items[0]
                where += " Background math in Foundations " + listed + "."
        else:
            title = f"{script.TITLE} | How LLMs Work: Foundations F{n:02d}"
            where = f"Foundations F{n:02d}" + (" (extra)" if n >= 12 else "") + ", a pre-series for How LLMs Work."
            if n == 11:
                where += " The code in episodes 3 to 13 uses these NumPy basics."
            else:
                eps = [str(e) for e, _ in script.USED_IN]
                where += " Used in episode" + ("s " if len(eps) > 1 else " ") + \
                    (", ".join(eps[:-1]) + " and " + eps[-1] if len(eps) > 1 else eps[0]) + "."

        summary = plain(study.LESSON["intro"])
        summary = re.sub(r"\b([Tt]his (bonus )?)lesson\b", r"\1video", summary)
        # Drop sentences that only make sense in the study guide (its exercises), not under a video.
        summary = " ".join(sentence for sentence in re.split(r"(?<=[.!?])\s+", summary)
                           if not re.search(r"questions ask you|your prediction", sentence))
        links = (f"Study guide for this lesson (PDF): {REPO}/blob/main/{series}/study/{vid}_study.pdf\n"
                 f"All lessons, study guides and source code: {REPO}")
        parts = [script.TAGLINE + ".", links, summary, "Chapters\n" + "\n".join(chapters(study.CONCEPTS, duration)),
                 where, CREDITS]
        if vid in ("v11", "v12", "f14"):
            parts[-1] += "\n" + TINY_SHAKESPEARE
        parts.append(AI_DISCLOSURE)
        description = "\n\n".join(parts)
        assert "<" not in title + description and ">" not in title + description
        assert len(title) <= 100 and len(description.encode()) <= 5000, vid
        assert not re.search(r"\b(guide|lesson|Check \d|exercises?|questions ask)\b", summary), (vid, summary)
        episodes.append({"file": video, "title": title, "description": description,
                         "category_id": CATEGORY_EDUCATION, "privacy": "public",
                         "made_for_kids": False, "contains_synthetic_media": False})
    title, description = cfg["playlist"]
    description += f"\n\nStudy guides (PDF) and source code: {REPO}"
    return {"playlist": {"title": title, "description": description, "privacy": "public"},
            "episodes": episodes}


# ---------------------------------------------------------------------------- LLMs in Practice (released in batches)
PRACTICE = "llms-in-practice"
PRACTICE_PLAYLIST = ("LLMs in Practice",
                     "How real products are built on top of a language model: prompts and context, context windows, "
                     "sampling, embeddings and search, RAG, tool use, agents, fine-tuning and LoRA, quantization, "
                     "evaluation and safety. Every episode has a study guide (PDF) and runnable code. "
                     "Follows the How LLMs Work series.")


def practice_files(n):
    """Paths of everything episode n needs before release, or None if any is missing."""
    vid = f"p{n:02d}"
    base = os.path.join(ROOT, PRACTICE)
    code = glob.glob(os.path.join(base, "code", f"{vid}_*"))
    video = glob.glob(os.path.join(base, "media", "videos", f"{vid}_scene", "1080p60", "*.mp4"))
    files = {"script": os.path.join(base, f"{vid}_script.py"), "scene": os.path.join(base, f"{vid}_scene.py"),
             "study": os.path.join(base, "study", f"{vid}_study.py"),
             "pdf": os.path.join(base, "study", f"{vid}_study.pdf")}
    if len(code) != 1 or len(video) != 1 or not all(os.path.exists(f) for f in files.values()):
        return None
    return dict(files, code=code[0], video=video[0])


def build_practice(numbers, playlist_id=None):
    """Manifest for the given LLMs in Practice episode numbers; reuse playlist_id once the playlist exists."""
    episodes = []
    for n in numbers:
        f = practice_files(n)
        assert f, f"episode {n} is not complete"
        vid = f"p{n:02d}"
        script, study = load(f["script"]), load(f["study"])
        with av.open(f["video"]) as container:
            duration = float(container.duration / av.time_base)
        code_url = f"{REPO}/tree/main/{PRACTICE}/code/{os.path.basename(f['code'])}"
        links = (f"Study guide for this lesson (PDF): {REPO}/blob/main/{PRACTICE}/study/{vid}_study.pdf\n"
                 f"Code for this episode: {code_url}\n"
                 f"All lessons, study guides and source code: {REPO}")
        summary = plain(study.LESSON["intro"])
        summary = re.sub(r"\b([Tt]his (bonus )?)lesson\b", r"\1video", summary)
        where = (f"Episode {n} of LLMs in Practice, the follow-on to How LLMs Work. "
                 "New to transformers? Start with the How LLMs Work playlist.")
        parts = [script.TAGLINE + ".", links, summary,
                 "Chapters\n" + "\n".join(chapters(study.CONCEPTS, duration)), where, CREDITS, AI_DISCLOSURE]
        description = "\n\n".join(parts)
        title = f"{script.TITLE} | LLMs in Practice, Ep. {n}"
        assert "<" not in title + description and ">" not in title + description, vid
        assert len(title) <= 100 and len(description.encode()) <= 5000, vid
        episodes.append({"file": f["video"], "title": title, "description": description,
                         "category_id": CATEGORY_EDUCATION, "privacy": "public",
                         "made_for_kids": False, "contains_synthetic_media": False})
    if playlist_id:
        playlist = {"id": playlist_id}
    else:
        title, description = PRACTICE_PLAYLIST
        playlist = {"title": title, "description": description + f"\n\nStudy guides (PDF) and source code: {REPO}",
                    "privacy": "public"}
    return {"playlist": playlist, "episodes": episodes}


if __name__ == "__main__":
    for series in SERIES:  # Foundations first: the main episodes point back to it
        path = os.path.join(ROOT, "youtube", f"{series}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(build(series), f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("wrote", os.path.relpath(path, ROOT))
