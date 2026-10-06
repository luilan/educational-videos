"""Machine-checkable answer keys for study-guide exercises (used by the StepStone app; the PDFs ignore them).

    python framework/answer_keys.py auto      # add keys where the printed answer is unambiguous
    python framework/answer_keys.py todo      # list exercises that still need a key (JSON)
    python framework/answer_keys.py check     # validate every key against its exercise
    python framework/answer_keys.py apply keys.json   # insert hand-written keys: [{"file", "concept", "exercise", "key"}]

Key format (format version 1), stored as "key" in each exercise dict:
    mc      {"choice": 1}                                  index into options
    tf      {"value": False}
    order   {"items": ["first", "second", ...]}            correct order; the app shuffles them
    number  {"parts": [{"label": None, "value": 0.55, "tol": 0.01, "unit": None}, ...]}
            one part per number the learner must give; |given - value| <= tol is right; unit "%" also accepts
            the fraction (55% == 0.55)
    any     {"self": True}                                 self-graded: the answer can't be checked automatically
short and code exercises are always self-graded and need no key.
"""
import ast
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"(?<![\w.])[−-]?\d[\d,]*(?:\.\d+)?(?![\w])")
GRADED = ("mc", "tf", "order", "number")


def study_files():
    return sorted(glob.glob(str(ROOT / "*/study/*_study.py")))


def load(path):
    spec = importlib.util.spec_from_file_location("study", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def plain(s):
    return re.sub(r"<[^>]+>", "", s).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def to_float(tok):
    return float(tok.replace(",", "").replace("−", "-"))


def tolerance(tok, value, approx):
    d = len(tok.split(".")[1]) if "." in tok else 0
    tol = 0.5 * 10 ** -d
    if approx:
        tol = max(tol, 0.02 * abs(value))
    return clean(tol)


def clean(x):
    """Short, exact-looking numbers in the source: 0.011 not 0.011000000000000001, 12 not 12.0."""
    x = float(f"{x:.6g}")
    return int(x) if x.is_integer() else x


def auto_key(e):
    k, ans = e["kind"], plain(e["answer"]).strip()
    if k == "tf":
        m = re.match(r"(True|False)\b", ans)
        return {"value": m.group(1) == "True"} if m else None
    if k == "mc":
        m = re.match(r"([A-H])\b[.)]?", ans)
        if m and ord(m.group(1)) - 65 < len(e["options"]):
            return {"choice": ord(m.group(1)) - 65}
        hits = [i for i, o in enumerate(e["options"]) if plain(o).strip(" .").lower() == ans.strip(" .").lower()]
        return {"choice": hits[0]} if len(hits) == 1 else None
    if k == "order":
        m = re.search(r"<i>(.*?)</i>", e["q"])
        if not m or "→" not in ans:
            return None
        items = [plain(x).strip() for x in m.group(1).split("·")]
        out = []
        for piece in ans.rstrip(".").split("→"):
            p = re.sub(r"\s*\(.*?\)\s*$", "", piece.strip()).lower()
            hits = [it for it in items if it.lower() == p or it.lower().strip(" .") == p.strip(" .")]
            if len(hits) != 1:
                return None
            out.append(hits[0])
        return {"items": out} if sorted(out) == sorted(items) else None
    if k == "number":
        toks = NUM.findall(ans)
        if len(toks) != 1:
            return None
        v = to_float(toks[0])
        v = int(v) if v.is_integer() else v                         # exactly as printed
        unit = "%" if re.search(re.escape(toks[0]) + r"\s*%", ans) else None
        approx = bool(re.search(r"\babout\b|≈|~|roughly|around", ans, re.I))
        return {"parts": [{"label": None, "value": v, "tol": tolerance(toks[0], v, approx), "unit": unit}]}
    return None


def exercise_nodes(path):
    """(concept index, exercise index, ast.Dict) for every exercise, in source order."""
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CONCEPTS" for t in node.targets):
            for ci, c in enumerate(node.value.elts):
                ex = next(v for kk, v in zip(c.keys, c.values) if getattr(kk, "value", None) == "exercises")
                for ei, d in enumerate(ex.elts):
                    yield ci, ei, d


def insert_keys(path, keys):
    """keys: {(ci, ei): key}; writes '"key": ...' before each exercise dict's closing brace."""
    lines = Path(path).read_text().split("\n")
    spots = []
    for ci, ei, d in exercise_nodes(path):
        if (ci, ei) in keys and not any(getattr(kk, "value", None) == "key" for kk in d.keys):
            spots.append((d.end_lineno - 1, d.end_col_offset - 1, keys[(ci, ei)]))
    for ln, col, key in sorted(spots, reverse=True):
        line = lines[ln]
        col = len(line.encode()[:col].decode())                    # ast offsets are in UTF-8 bytes
        assert line[col] == "}", (path, ln, line)
        before = line[:col].rstrip()
        sep = " " if before.endswith(",") or before.endswith("{") else ", "
        lines[ln] = before + sep + '"key": ' + repr(key) + line[col:]
    Path(path).write_text("\n".join(lines))
    return len(spots)


def each():
    for path in study_files():
        m = load(path)
        for ci, c in enumerate(m.CONCEPTS):
            for ei, e in enumerate(c["exercises"]):
                yield path, ci, ei, e


def check_key(e):
    k, key = e["kind"], e.get("key")
    if k not in GRADED:
        return None if key in (None, {"self": True}) else "short/code exercises take no key"
    if key is None:
        return "missing key"
    if key == {"self": True}:
        return None
    if k == "tf":
        return None if isinstance(key.get("value"), bool) else "tf key needs a bool value"
    if k == "mc":
        c = key.get("choice")
        return None if isinstance(c, int) and 0 <= c < len(e["options"]) else "mc choice out of range"
    if k == "order":
        it = key.get("items")
        return None if isinstance(it, list) and len(it) >= 2 and all(isinstance(x, str) for x in it) else "order needs 2+ items"
    if k == "number":
        parts = key.get("parts")
        if not parts:
            return "number key needs parts"
        ans = plain(e["answer"]).replace(",", "").replace("−", "-")
        nums = [to_float(t) for t in NUM.findall(plain(e["answer"]))]
        for p in parts:
            if not isinstance(p.get("value"), (int, float)) or not isinstance(p.get("tol"), (int, float)) or p["tol"] < 0:
                return "number part needs value and tol >= 0"
            same = lambda a, b: abs(a - b) <= 1e-9 * max(1.0, abs(a))
            if not any(same(n, p["value"]) or same(n / 100, p["value"]) or same(n * 100, p["value"]) for n in nums):
                return f"part value {p['value']} not found in the printed answer {ans!r}"
            if len(parts) > 1 and not p.get("label"):
                return "multi-part keys need a label per part"
    return None


def main(cmd):
    if cmd == "auto":
        per_file, added = {}, 0
        for path, ci, ei, e in each():
            if e["kind"] in GRADED and "key" not in e:
                k = auto_key(e)
                if k:
                    per_file.setdefault(path, {})[(ci, ei)] = k
        for path, keys in per_file.items():
            added += insert_keys(path, keys)
        print(f"added {added} keys")
    elif cmd == "todo":
        todo = [{"file": str(Path(p).relative_to(ROOT)), "concept": ci, "exercise": ei, "kind": e["kind"], "q": e["q"],
                 "answer": e["answer"], **({"options": e["options"]} if "options" in e else {})}
                for p, ci, ei, e in each() if e["kind"] in GRADED and "key" not in e]
        print(json.dumps(todo, ensure_ascii=False, indent=1))
        print(f"{len(todo)} exercises need a key", file=sys.stderr)
    elif cmd == "apply":
        per_file, added = {}, 0
        for row in json.load(open(sys.argv[2])):
            per_file.setdefault(str(ROOT / row["file"]), {})[(row["concept"], row["exercise"])] = row["key"]
        for path, keys in per_file.items():
            added += insert_keys(path, keys)
        print(f"added {added} keys")
    elif cmd == "check":
        bad = [(Path(p).name, ci, ei, err) for p, ci, ei, e in each() if (err := check_key(e))]
        for b in bad:
            print(*b)
        total = sum(1 for _ in each())
        print(f"{total - len(bad)}/{total} exercises OK", file=sys.stderr)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1])
