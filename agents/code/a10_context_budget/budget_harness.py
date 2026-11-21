"""The context budget: a reading task that outgrows a small context window, with and without compaction.

    python3 budget_harness.py unlimited   # no budget: every tool output stays in the conversation
    python3 budget_harness.py compact     # a 4,096-token window; at 90% the harness compacts, the way Codex does
    python3 budget_harness.py capped      # compact, plus episode 4's cap: every tool result cut to 2,000 characters

Compaction (Codex: codex-rs/prompts/templates/compact/): ask the model for a handoff summary with Codex's own prompt
(compact_prompt.md), then continue from just the task plus that summary, introduced by compact_summary_prefix.md.
Three questions, three files to read; the harness checks the three answers at the end.
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

HERE = Path(__file__).parent
TASK = ("Answer three questions about this project. Read the files to find out.\n"
        "1. Which cover image file does the agents series use? (stepstone/build_packs.py)\n"
        "2. What is the value of MIN_CHAPTER? (youtube/build_manifests.py)\n"
        "3. What is the value of MAX_TITLE_WIDTH? (framework/intro.py)")
EXPECTED = [r"AgentsCover\.jpg", r"\b10\b", r"\b12\.8\b"]
BASE = "You are a coding agent working in a project folder. Use the shell tool to run commands."
TOOLS = [{"type": "function", "name": "shell", "description": "Run a bash command in the project folder and return its output.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
WINDOW = 4096
COMPACT_AT = WINDOW * 9 // 10        # Codex: auto-compact at 90% of the context window
MAX_STEPS = 12
LIMIT = 2000                         # characters kept per tool result in "capped" mode (episode 4)


def truncate_middle(text, limit=LIMIT):
    if len(text) <= limit:
        return text
    return f"{text[:limit // 2]}\n…{len(text) - limit} chars truncated…\n{text[-(limit - limit // 2):]}"


def text_of(r):
    return "".join(c.get("text", "") for o in r["output"] if o["type"] == "message" for c in o["content"])


def compact(history):
    r = respond(history + [{"role": "user", "content": (HERE / "compact_prompt.md").read_text()}])
    summary = text_of(r).split("</think>")[-1].strip()
    print(f"   [compacted: {r['usage']['input_tokens']:,} tokens -> summary of {r['usage']['output_tokens']:,} written, "
          f"{len(summary):,} chars kept]")
    return [{"role": "user", "content": TASK},
            {"role": "user", "content": (HERE / "compact_summary_prefix.md").read_text() + "\n" + summary}], summary


def run(mode):
    work = Path(tempfile.mkdtemp(prefix="budget-"))
    shutil.copytree(HERE / "demo_project", work, dirs_exist_ok=True)
    history = [{"role": "user", "content": TASK}]
    sizes, compactions, answer = [], 0, ""
    recent = []
    for step in range(1, MAX_STEPS + 1):
        r = respond(history, TOOLS, BASE)
        used = r["usage"]["input_tokens"]
        sizes.append(used)
        calls = [o for o in r["output"] if o["type"] == "function_call"]
        history += r["output"]
        print(f"step {step}: sent {used:>6,} tokens{'  (over the 4,096 window!)' if used > WINDOW else ''}")
        if not calls:
            answer = text_of(r)
            break
        for c in calls:
            cmd = json.loads(c["arguments"]).get("command", "")
            p = subprocess.run(["bash", "-c", cmd], cwd=work, capture_output=True, text=True, timeout=30)
            out = (p.stdout + p.stderr) or "(no output)"
            shown = truncate_middle(out) if mode == "capped" else out
            print(f"   $ {cmd[:90]}   -> {len(out):,} chars" + (f", {len(shown):,} shown" if shown != out else ""))
            out = shown
            history.append({"type": "function_call_output", "call_id": c["call_id"], "output": out})
        sig = tuple(sorted({c["arguments"] for c in calls}))
        recent = (recent + [sig])[-3:]
        if len(recent) == 3 and len(set(recent)) == 1:           # episode 2's stuck detector
            print("   stopped: the same tool calls on three steps in a row")
            break
        if mode in ("compact", "capped") and used + sum(len(h.get("output", "")) for h in history[-len(calls):]) // 4 > COMPACT_AT:
            history, _ = compact(history)
            compactions += 1
    final = answer.split("</think>")[-1]
    right = [re.search(e, final) is not None for e in EXPECTED]
    print(f"\n{mode}: {step} model calls, {compactions} compactions, largest request {max(sizes):,} tokens")
    print("tokens sent per call:", ", ".join(f"{s:,}" for s in sizes))
    print("answers right:", sum(right), "of 3", right)
    print("answer (end): " + final.strip()[-300:].replace("\n", " "))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "compact")
