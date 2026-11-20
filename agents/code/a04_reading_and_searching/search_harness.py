"""Finding the right lines in a project too big to read: the same task in two harness configurations.

    python3 search_harness.py naive     # full command output, no guidance
    python3 search_harness.py search    # search-first instructions + Codex-style middle truncation of long output
    python3 search_harness.py context   # search-first, plus: read the lines around each match, follow values back

The project in demo_project/ is 12 Python files, 1,455 lines, about 19,000 tokens (Qwen3 tokenizer).
WARNING: runs the model's commands. Use a container or a throwaway VM.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from llm import respond

TASK = ("In this project, which line decides the file name of a figure image extracted for the study-guide PDF? "
        "Answer with the file path, the line number and the exact format string.")
BASE = "You are a coding agent working in a project folder. Use the shell tool to inspect files."
SEARCH = (BASE + " The project is too big to read whole. Search first: use `grep -rn` to find candidate lines, then read "
          "only the lines you need with `sed -n 'START,ENDp' FILE`. Never print whole files.")
CONTEXT = (SEARCH + " Use `grep -rn -C 3` so you see a few lines around each match. When you find where a value is "
           "used, search again for where it is set.")
TOOLS = [{"type": "function", "name": "shell", "description": "Run a bash command in the project folder and return its output.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
MAX_STEPS = 10
LIMIT = 2000                     # characters kept per command output (search and context modes)


def truncate_middle(text, limit=LIMIT):
    """Keep the beginning and the end, drop the middle (Codex: utils/string/src/truncate.rs)."""
    if len(text) <= limit:
        return text
    head, tail = text[:limit // 2], text[-(limit - limit // 2):]
    return f"{head}\n…{len(text) - limit} chars truncated…\n{tail}"


def run(mode):
    work = Path(tempfile.mkdtemp())
    shutil.copytree(Path(__file__).parent / "demo_project", work, dirs_exist_ok=True)
    history = [{"role": "user", "content": TASK}]
    sent = 0
    answer = ""
    for step in range(1, MAX_STEPS + 1):
        r = respond(history, TOOLS, {"search": SEARCH, "context": CONTEXT}.get(mode, BASE))
        sent += r["usage"]["input_tokens"]
        calls = [o for o in r["output"] if o["type"] == "function_call"]
        history += r["output"]
        print(f"step {step}: sent {r['usage']['input_tokens']:>6} tokens, {r['_seconds']:4.0f} s")
        if not calls:
            answer = "".join(c.get("text", "") for o in r["output"] if o["type"] == "message" for c in o["content"])
            break
        for c in calls:
            cmd = json.loads(c["arguments"]).get("command", "")
            p = subprocess.run(["bash", "-c", cmd], cwd=work, capture_output=True, text=True, timeout=30)
            out = (p.stdout + p.stderr) or "(no output)"
            shown = truncate_middle(out) if mode != "naive" else out
            print(f"   $ {cmd[:100]}   -> {len(out):,} chars of output" + (f", {len(shown):,} shown" if shown != out else ""))
            history.append({"type": "function_call_output", "call_id": c["call_id"], "output": shown})
    tail = answer.split("</think>")[-1].strip()[-500:]
    print(f"\n{mode}: {step} model calls, {sent:,} input tokens in total")
    print("answer (end): " + tail.replace("\n", " "))
    final = answer.split("</think>")[-1]            # judge only the final answer, not the reasoning
    right = "study_guide.py" in final and re.search(r"\b42\b", final) is not None
    print("check:", "RIGHT (framework/study_guide.py, line 42)" if right else "WRONG or incomplete")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "search")
