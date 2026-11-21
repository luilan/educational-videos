"""Instructions and memory: does the agent follow the project's own rules? One bug fix, three setups.

    python3 instructions_harness.py none       # no AGENTS.md in the project
    python3 instructions_harness.py file       # AGENTS.md is in the folder, but the harness does not send it
    python3 instructions_harness.py injected   # the harness sends AGENTS.md, the way Codex does (a user message)
    python3 instructions_harness.py system     # the harness puts AGENTS.md into the system instructions instead
    python3 instructions_harness.py hooks      # injected, plus a Stop check: the harness refuses to finish until the
                                               # CHANGELOG line exists and run_tests.py passes (at most twice)

demo/AGENTS.md asks for three things: run tests with run_tests.py, add a line to CHANGELOG.md, keep the change small.
The harness checks all three afterwards.
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

TASK = "The tests fail. Read invoice.py, find the bug, and fix it by editing invoice.py with your tool."
BASE = "You are a coding agent working in a project folder. Use the shell tool to run commands and edit files."
TOOLS = [{"type": "function", "name": "shell", "description": "Run a bash command in the project folder and return its output.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
MAX_STEPS = 12


def agents_md_message(folder, text):
    """Codex sends AGENTS.md as a user message in this shape (codex-rs/core/src/agents_md.rs)."""
    return {"role": "user", "content": f"# AGENTS.md instructions for {folder}\n\n<INSTRUCTIONS>\n{text}\n</INSTRUCTIONS>"}



def stop_check(work, changelog_before):
    """The harness's own check of AGENTS.md's rules when the model wants to stop. Returns a reason, or None."""
    problems = []
    if (work / "CHANGELOG.md").read_text() == changelog_before:
        problems.append('no line was added to CHANGELOG.md under "Unreleased"')
    if subprocess.run(["python3", "run_tests.py"], cwd=work, capture_output=True).returncode != 0:
        problems.append("python3 run_tests.py fails")
    return "; ".join(problems) or None

def run(mode):
    work = Path(tempfile.mkdtemp(prefix="instr-"))
    shutil.copytree(Path(__file__).parent / "demo", work, dirs_exist_ok=True)
    rules = (work / "AGENTS.md").read_text()
    if mode == "none":
        (work / "AGENTS.md").unlink()
    original = (work / "invoice.py").read_text()
    changelog = (work / "CHANGELOG.md").read_text()
    history = [agents_md_message(work, rules)] if mode in ("injected", "hooks") else []
    history.append({"role": "user", "content": TASK})
    commands, blocks = [], 0
    recent = []
    for step in range(1, MAX_STEPS + 1):
        r = respond(history, TOOLS, BASE + "\n\n" + rules if mode == "system" else BASE)
        calls = [o for o in r["output"] if o["type"] == "function_call"]
        history += r["output"]
        print(f"step {step}: sent {r['usage']['input_tokens']:>5} tokens, {r['_seconds']:4.0f} s")
        if not calls:
            reason = stop_check(work, changelog) if mode == "hooks" and blocks < 2 else None
            if reason:                                  # like Codex's Stop hook: block the end of the turn
                blocks += 1
                print(f"   Stop blocked: {reason}")
                history.append({"role": "user", "content": f"Stop hook blocked the end of the turn: {reason}"})
                continue
            break
        for c in calls:
            cmd = json.loads(c["arguments"]).get("command", "")
            commands.append(cmd)
            p = subprocess.run(["bash", "-c", cmd], cwd=work, capture_output=True, text=True, timeout=30)
            out = f"exit code {p.returncode}\n{p.stdout}{p.stderr}"
            print(f"   $ {cmd.splitlines()[0][:90] if cmd else ''}{' …' if cmd.count(chr(10)) else ''}   -> exit {p.returncode}")
            history.append({"type": "function_call_output", "call_id": c["call_id"], "output": out[-4000:]})
        sig = tuple(sorted({c["arguments"] for c in calls}))
        recent = (recent + [sig])[-3:]
        if len(recent) == 3 and len(set(recent)) == 1:           # episode 2's stuck detector
            print("   stopped: the same tool calls on three steps in a row")
            break
    diff = [d for d in difflib.unified_diff(original.splitlines(), (work / "invoice.py").read_text().splitlines(),
                                            lineterm="", n=0) if d[:1] in "+-" and not d.startswith(("+++", "---"))]
    new_log = (work / "CHANGELOG.md").read_text()
    test = subprocess.run(["python3", "run_tests.py"], cwd=work, capture_output=True, text=True)
    print(f"\n{mode}: {step} model calls, {len(commands)} commands, {blocks} stops blocked")
    print("read AGENTS.md itself:", any("AGENTS.md" in c for c in commands))
    print("used run_tests.py:", any("run_tests.py" in c for c in commands))
    print("CHANGELOG line added:", new_log != changelog and "Fixed" in new_log.split("## 1.3.0")[0])
    print(f"lines changed in invoice.py: {len(diff)}")
    print("tests pass:", test.returncode == 0)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "injected")
