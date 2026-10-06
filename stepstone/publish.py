"""Publish StepStone content to GitHub: offline videos, lesson packs and the catalog.

    python stepstone/publish.py            # everything (idempotent: only uploads what is new or changed)
    python stepstone/publish.py --no-videos

1. Videos → release "stepstone-videos" as <series>-<lesson>.mp4 (uploaded when missing or a different size).
2. Packs are rebuilt with those offline-video links → release "stepstone-packs" (new versions uploaded,
   superseded versions of the same series deleted).
3. catalog.json → docs/stepstone/catalog.json, served by GitHub Pages at
   https://luilan.github.io/educational-videos/stepstone/catalog.json (commit and push docs/ afterwards).
Needs the gh CLI logged in with write access to luilan/educational-videos.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import build_packs as bp

GH = "luilan/educational-videos"


def gh(*args, check=True):
    return subprocess.run(["gh", *args, "-R", GH], capture_output=True, text=True, check=check)


def ensure_release(tag, title, notes):
    if gh("release", "view", tag, check=False).returncode != 0:
        gh("release", "create", tag, "--title", title, "--notes", notes, "--latest=false")
        print("created release", tag)


def assets(tag):
    out = gh("release", "view", tag, "--json", "assets").stdout
    return {a["name"]: a for a in json.loads(out)["assets"]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_videos():
    ensure_release(bp.VIDEOS_RELEASE, "StepStone offline videos",
                   "Lesson videos for offline study in the StepStone app. Same videos as on YouTube.")
    have = assets(bp.VIDEOS_RELEASE)
    offline = {}
    for series, cfg in bp.SERIES.items():
        for p in sorted((bp.ROOT / series / "study").glob(f"{cfg['prefix']}[0-9][0-9]_study.py")):
            vid = p.name[:3]
            src = bp.render_path(series, vid)
            if src is None:
                continue
            name = f"{series}-{vid}.mp4"
            size = src.stat().st_size
            if name not in have or have[name]["size"] != size:
                tmp = Path("/tmp") / name                          # gh names the asset after the file
                shutil.copyfile(src, tmp)
                gh("release", "upload", bp.VIDEOS_RELEASE, str(tmp), "--clobber")
                tmp.unlink()
                print("uploaded", name, f"{size / 1e6:.1f} MB", flush=True)
            offline.setdefault(series, {})[vid] = {"url": f"{bp.REPO}/releases/download/{bp.VIDEOS_RELEASE}/{name}",
                                                   "size": size, "sha256": sha256(src)}
    return offline


def publish_packs(offline):
    ensure_release(bp.PACKS_RELEASE, "StepStone lesson packs",
                   "Lesson packs for the StepStone app, built from the study guides. The app reads "
                   "https://luilan.github.io/educational-videos/stepstone/catalog.json to find them.")
    packs = bp.main([], offline)
    have = assets(bp.PACKS_RELEASE)
    for p in packs:
        if p["file"] not in have:
            gh("release", "upload", bp.PACKS_RELEASE, str(bp.DIST / p["file"]))
            print("uploaded", p["file"])
        for name in have:                                         # drop superseded versions of this series
            if name.startswith(p["id"] + "-v") and name != p["file"]:
                gh("release", "delete-asset", bp.PACKS_RELEASE, name, "--yes")
                print("deleted", name)
    dest = bp.ROOT / "docs" / "stepstone" / "catalog.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bp.DIST / "catalog.json", dest)
    print("catalog →", dest.relative_to(bp.ROOT))


if __name__ == "__main__":
    offline = {} if "--no-videos" in sys.argv else publish_videos()
    publish_packs(offline)
