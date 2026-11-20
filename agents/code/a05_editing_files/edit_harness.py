"""Editing files: the same one-line bug fix, done two ways.

    python3 edit_harness.py rewrite   # edit by writing the whole file again (cat > FILE <<'EOF' ... EOF)
    python3 edit_harness.py patch     # edit with Codex's patch format; the harness intercepts `apply_patch` commands
    python3 edit_harness.py read      # patch, plus: read the lines you will change before editing

demo/invoice.py is 108 lines; the bug is one line in total(). After each run the harness checks the test and compares
the edited file with the original: how many lines changed, and whether anything else got lost on the way.
WARNING: runs the model's commands. Use a container or a throwaway VM.
"""
import difflib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from llm import respond

TASK = ("The test in test_invoice.py fails. Run it with: python3 test_invoice.py. Read invoice.py, find the bug, and "
        "fix it by editing invoice.py with your tool. Then run the test again.")
BASE = "You are a coding agent working in a project folder. Use the shell tool to run commands."
REWRITE = BASE + (" To edit a file, write the complete new file with a heredoc:\ncat > FILE <<'EOF'\n...whole file...\nEOF")
PATCH = BASE + """ To edit a file, never rewrite it: run the apply_patch command with a patch that shows only the lines that
change, with a few unchanged lines around them for context. Format:
apply_patch <<'EOF'
*** Begin Patch
*** Update File: path/to/file.py
@@ def example():
     unchanged line
-    old line
+    new line
     unchanged line
*** End Patch
EOF
Lines starting with a space are context, - removes a line, + adds a line."""
READ_FIRST = PATCH + "\nBefore you edit a file, read the lines you will change with `grep -n` or `sed -n 'START,ENDp' FILE`."
TOOLS = [{"type": "function", "name": "shell", "description": "Run a bash command in the project folder and return its output.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
MAX_STEPS = 10


# ---------------------------------------------------------------- a small apply_patch (Codex: codex-rs/apply-patch)
def seek(lines, pattern, start):
    """Find `pattern` in `lines` from `start`: exact, then ignoring trailing spaces, then ignoring all edge spaces."""
    for norm in (lambda s: s, str.rstrip, str.strip):
        for i in range(start, len(lines) - len(pattern) + 1):
            if all(norm(lines[i + j]) == norm(p) for j, p in enumerate(pattern)):
                return i
    return None


def apply_patch(patch, root):
    body = patch.split("*** Begin Patch", 1)[1].split("*** End Patch", 1)[0].strip("\n").split("\n")
    if not body or not body[0].startswith("*** Update File: "):
        return "apply_patch failed: expected '*** Update File: <path>' after '*** Begin Patch'"
    path = root / body[0][len("*** Update File: "):].strip()
    if not path.resolve().is_relative_to(root.resolve()):
        return "apply_patch failed: path is outside the project folder"
    if not path.exists():
        return f"apply_patch failed: {path.name} does not exist"
    lines = path.read_text().split("\n")
    chunks, cur = [], None
    for ln in body[1:]:
        if ln.startswith("@@"):
            cur = ([], []); chunks.append(cur)
            continue
        if cur is None:
            cur = ([], []); chunks.append(cur)
        tag, text = (ln[:1], ln[1:]) if ln[:1] in "+- " and ln else (" ", ln)
        if tag in " -":
            cur[0].append(text)
        if tag in " +":
            cur[1].append(text)
    pos = 0
    for old, new in chunks:
        i = seek(lines, old, pos)
        if i is None:
            return "apply_patch failed: could not find these lines in the file:\n" + "\n".join(old)
        lines[i:i + len(old)] = new
        pos = i + len(new)
    path.write_text("\n".join(lines))
    return f"Success. Updated the following files:\nM {path.name}"


# ---------------------------------------------------------------- the loop
def run(mode):
    work = Path(tempfile.mkdtemp())
    shutil.copytree(Path(__file__).parent / "demo", work, dirs_exist_ok=True)
    original = (work / "invoice.py").read_text()
    history = [{"role": "user", "content": TASK}]
    edits, recent = [], []
    stop = "no tool call"
    for step in range(1, MAX_STEPS + 1):
        r = respond(history, TOOLS, {"patch": PATCH, "read": READ_FIRST}.get(mode, REWRITE))
        calls = [o for o in r["output"] if o["type"] == "function_call"]
        history += r["output"]
        print(f"step {step}: wrote {r['usage']['output_tokens']:>5} tokens, {r['_seconds']:4.0f} s")
        if not calls:
            answer = "".join(c.get("text", "") for o in r["output"] if o["type"] == "message" for c in o["content"])
            print("   final message (end): " + answer.split("</think>")[-1].strip()[-300:].replace("\n", " "))
            break
        for c in calls:
            cmd = json.loads(c["arguments"]).get("command", "")
            if mode != "rewrite" and cmd.lstrip().startswith("apply_patch") and "*** Begin Patch" in cmd:
                out = apply_patch(cmd, work)                       # intercepted: never reaches the shell
                edits.append(len(c["arguments"]))
            else:
                if "cat >" in cmd or "> invoice.py" in cmd:
                    edits.append(len(c["arguments"]))
                p = subprocess.run(["bash", "-c", cmd], cwd=work, capture_output=True, text=True, timeout=30)
                out = (p.stdout + p.stderr) or "(no output)"
            print(f"   $ {cmd.splitlines()[0][:90] if cmd else ''}{' …' if cmd.count(chr(10)) else ''}   "
                  f"({len(c['arguments']):,} chars)  -> {out.splitlines()[0][:70] if out.strip() else ''}")
            history.append({"type": "function_call_output", "call_id": c["call_id"], "output": out[:4000]})
        recent = (recent + [tuple(sorted({json.loads(c["arguments"]).get("command", "") for c in calls}))])[-3:]
        if len(recent) == 3 and len(set(recent)) == 1:               # episode 2's stuck detector (per reply)
            stop = "stuck: the same command on three steps in a row"
            break
    edited = (work / "invoice.py").read_text()
    diff = [d for d in difflib.unified_diff(original.splitlines(), edited.splitlines(), lineterm="", n=0)
            if d[:1] in "+-" and not d.startswith(("+++", "---"))]
    test = subprocess.run(["python3", "test_invoice.py"], cwd=work, capture_output=True, text=True)
    print(f"\n{mode}: {step} model calls (stopped: {stop if step < MAX_STEPS else 'step limit'}); edit calls: {len(edits)} ({', '.join(f'{e:,}' for e in edits)} chars)")
    print(f"changed lines: {sum(d[0] == '-' for d in diff)} removed, {sum(d[0] == '+' for d in diff)} added")
    for d in diff[:12]:
        print("   " + d[:110])
    print("test:", "PASS" if test.returncode == 0 else "FAIL")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "patch")
