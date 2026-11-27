"""One-off: add the StepStone app line to every published episode and playlist description.

Usage: python youtube/add_stepstone_line.py [--apply]
Without --apply it only lists what would change (read-only API calls, ~10 quota units).
With --apply each changed video costs 50 units (videos.update) and each playlist 50 (playlists.update).
"""
import sys
from pathlib import Path

import httplib2
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "uploader"))
from uploader import load_credentials  # noqa: E402

sys.path.insert(0, str(ROOT / "youtube"))
sys.modules.setdefault("av", type(sys)("av"))  # build_manifests imports av only to read durations
from build_manifests import STEPSTONE_LINE, REPO  # noqa: E402

TOKEN = ROOT / "uploader" / "credentials" / "token.json"
ANCHOR = "All lessons, study guides and source code: "
PLAYLIST_ANCHOR = f"Study guides (PDF) and source code: {REPO}"


def pages(call, **kw):
    token = None
    while True:
        resp = call(pageToken=token, **kw).execute()
        yield from resp.get("items", [])
        token = resp.get("nextPageToken")
        if not token:
            return


def new_video_description(text):
    if "StepStone" in text or ANCHOR not in text:
        return None
    lines = text.split("\n")
    i = next(k for k, line in enumerate(lines) if line.startswith(ANCHOR))
    lines.insert(i + 1, STEPSTONE_LINE)
    return "\n".join(lines)


def new_playlist_description(text):
    if "StepStone" in text or PLAYLIST_ANCHOR not in text:
        return None
    return text.replace(PLAYLIST_ANCHOR, PLAYLIST_ANCHOR + "\n" + STEPSTONE_LINE, 1)


def main(apply):
    http = AuthorizedHttp(load_credentials(TOKEN), http=httplib2.Http(timeout=60))
    yt = build("youtube", "v3", http=http, cache_discovery=False, static_discovery=True)
    uploads = yt.channels().list(part="contentDetails", mine=True).execute()["items"][0][
        "contentDetails"]["relatedPlaylists"]["uploads"]
    ids = [it["contentDetails"]["videoId"]
           for it in pages(yt.playlistItems().list, part="contentDetails", playlistId=uploads, maxResults=50)]
    changes, skipped = [], 0
    for k in range(0, len(ids), 50):
        for v in yt.videos().list(part="snippet", id=",".join(ids[k:k + 50])).execute()["items"]:
            desc = new_video_description(v["snippet"]["description"])
            if desc is None:
                skipped += 1
                continue
            assert len(desc.encode()) <= 5000, v["id"]
            changes.append(("video", v, desc))
    for p in pages(yt.playlists().list, part="snippet", mine=True, maxResults=50):
        desc = new_playlist_description(p["snippet"]["description"])
        if desc is not None:
            changes.append(("playlist", p, desc))
    print(f"videos on channel: {len(ids)}, unchanged: {skipped}, changes: {len(changes)} "
          f"({sum(c[0] == 'playlist' for c in changes)} playlists)")
    if changes:
        print("--- sample ---\n" + changes[0][2][:700] + "\n--------------")
    if not apply:
        return
    done = 0
    for kind, item, desc in changes:
        s = item["snippet"]
        if kind == "video":
            body = {"id": item["id"], "snippet": {
                "title": s["title"], "description": desc, "categoryId": s["categoryId"],
                **{k: s[k] for k in ("tags", "defaultLanguage", "defaultAudioLanguage") if k in s}}}
            yt.videos().update(part="snippet", body=body).execute()
        else:
            body = {"id": item["id"], "snippet": {
                "title": s["title"], "description": desc,
                **{k: s[k] for k in ("defaultLanguage",) if k in s}}}
            yt.playlists().update(part="snippet", body=body).execute()
        done += 1
        print(f"updated {kind} {item['id']} {s['title'][:60]}", flush=True)
    print(f"done: {done}/{len(changes)}")
    yt.close()


if __name__ == "__main__":
    main("--apply" in sys.argv)
