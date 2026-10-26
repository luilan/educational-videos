"""The agent loop with its stop conditions made explicit.

    python3 agent_loop.py fix      # fix the bug in demo/ (the loop should end when the model answers in words)
    python3 agent_loop.py wait     # an impossible task: wait for a file that never appears

Stops when: the model answers in words (done) · the same command is requested 3 times in a row (stuck) ·
MAX_STEPS model calls (step limit). Each step prints the tokens sent: the whole conversation goes back every time.
WARNING: runs the model's commands. Use a container or a throwaway VM.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from llm import respond

TASKS = {
    "fix": "The test in test_stats.py fails. Run it with: python3 test_stats.py. Find the bug in stats.py, fix it, "
           "and run the test again to confirm it passes.",
    "wait": "Another process will create the file ready.txt in this folder. Wait until it exists, then print its "
            "contents.",
}
INSTRUCTIONS = "You are a coding agent working in a project folder. Use the shell tool to inspect, edit and test files."
TOOLS = [{"type": "function", "name": "shell", "description": "Run a bash command in the project folder and return its output.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
MAX_STEPS, MAX_REPEATS = 8, 3


def run(task):
    work = Path(tempfile.mkdtemp())
    for f in (Path(__file__).parent / "demo").iterdir():
        shutil.copy(f, work / f.name)
    history = [{"role": "user", "content": TASKS[task]}]
    last_cmds, reason = [], "step limit"
    for step in range(1, MAX_STEPS + 1):
        r = respond(history, TOOLS, INSTRUCTIONS)
        u = r["usage"]
        calls = [o for o in r["output"] if o["type"] == "function_call"]
        history += r["output"]
        print(f"step {step}: sent {u['input_tokens']:>5} tokens, got {u['output_tokens']:>4}, {r['_seconds']:4.0f} s, "
              f"{len(calls)} tool call(s)")
        if not calls:
            text = "".join(c.get("text", "") for o in r["output"] if o["type"] == "message" for c in o["content"])
            print("   answer: " + text.strip().replace("\n", " ")[:300])
            reason = "done: answered in words"
            break
        for c in calls:
            cmd = json.loads(c["arguments"]).get("command", "")
            try:
                p = subprocess.run(["bash", "-c", cmd], cwd=work, capture_output=True, text=True, timeout=30)
                out, code = (p.stdout + p.stderr)[-2000:] or "(no output)", p.returncode
            except subprocess.TimeoutExpired:
                out, code = "(timed out after 30 s)", -1
            print(f"   $ {cmd[:110]}   -> exit {code}")
            history.append({"type": "function_call_output", "call_id": c["call_id"], "output": f"exit code {code}\n{out}"})
            last_cmds.append(cmd)
        if len(last_cmds) >= MAX_REPEATS and len(set(last_cmds[-MAX_REPEATS:])) == 1:
            reason = f"stuck: the same command {MAX_REPEATS} times in a row"
            break
    print(f"stopped after {step} model call(s): {reason}")
    if task == "fix":
        ok = subprocess.run([sys.executable, "test_stats.py"], cwd=work, capture_output=True).returncode == 0
        print("independent check:", "test passes" if ok else "test still FAILS")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "fix")
