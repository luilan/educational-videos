"""A whole agent harness in one loop: the model asks for a shell command, the harness runs it, and feeds back the output.

WARNING: this runs whatever command the model writes. Run it only inside a container or a throwaway VM.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from llm import respond

TASK = ("The test in test_stats.py fails. Run it with: python3 test_stats.py. "
        "Find the bug in stats.py, fix it, and run the test again to confirm it passes.")
INSTRUCTIONS = "You are a coding agent working in a project folder. Use the shell tool to inspect, edit and test files."
TOOLS = [{"type": "function", "name": "shell", "description": "Run a bash command in the project folder and return its output.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]

work = Path(tempfile.mkdtemp())                       # a fresh copy of the demo project
for f in ("stats.py", "test_stats.py"):
    shutil.copy(Path(__file__).parent / "demo" / f, work / f)

history = [{"role": "user", "content": TASK}]
total_in = total_out = 0
for step in range(1, 11):                             # never loop forever
    r = respond(history, TOOLS, INSTRUCTIONS)
    total_in += r["usage"]["input_tokens"]; total_out += r["usage"]["output_tokens"]
    calls = [o for o in r["output"] if o["type"] == "function_call"]
    history += r["output"]                            # the model's turn goes back into the conversation
    if not calls:                                     # plain text means: done
        text = "".join(c.get("text", "") for o in r["output"] if o["type"] == "message" for c in o["content"])
        print(f"step {step}: model answers ({r['_seconds']:.0f} s)\n{text.strip()[:600]}")
        break
    for c in calls:
        cmd = json.loads(c["arguments"])["command"]
        p = subprocess.run(["bash", "-c", cmd], cwd=work, capture_output=True, text=True, timeout=60)
        out = (p.stdout + p.stderr)[-2000:] or "(no output)"
        print(f"step {step}: $ {cmd}   [exit {p.returncode}, {r['_seconds']:.0f} s]\n    " + out.strip().replace("\n", "\n    ")[:400])
        history.append({"type": "function_call_output", "call_id": c["call_id"], "output": f"exit code {p.returncode}\n{out}"})

final = subprocess.run([sys.executable, "test_stats.py"], cwd=work, capture_output=True, text=True)
print(f"\n{step} model calls, {total_in} tokens in, {total_out} out")
print("independent check:", "test passes" if final.returncode == 0 else "test still FAILS")
print("stats.py is now:\n" + (work / "stats.py").read_text())
