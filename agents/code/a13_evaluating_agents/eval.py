"""Evaluating agents: four tasks from earlier episodes, each run k times, graded automatically.

    python3 eval.py 3          # k = 3 runs per task; writes results_<date>.json

The agent is the episode-2 harness: one shell tool, a 30-second timeout per command, at most 10 model calls.
Graders check the outcome, not the transcript: tests must pass AND the test files must be unchanged (an agent can
"pass" by editing the test), answers must contain the right facts.
Reported per task: pass rate (pass@1), whether all k runs passed (pass^k), tokens sent and time.
WARNING: runs the model's commands. Use a container or a throwaway VM.
"""
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from llm import respond

HERE = Path(__file__).parent / "tasks"
BASE = "You are a coding agent working in a project folder. Use the shell tool to run commands and edit files."
TOOLS = [{"type": "function", "name": "shell", "description": "Run a bash command in the project folder and return its output.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
MAX_STEPS = 10



def kill_matching(word):
    """Stop leftover processes whose command line contains `word` (like pkill -f, which slim images lack)."""
    for d in os.listdir("/proc"):
        try:
            if d.isdigit() and int(d) != os.getpid() and word in open(f"/proc/{d}/cmdline").read():
                os.kill(int(d), 9)
        except (OSError, ProcessLookupError):
            pass

def tests_pass_unchanged(test_file):
    def grade(work, answer):
        same = (work / test_file).read_text() == (HERE / work.name.split("-")[0] / test_file).read_text()
        ok = subprocess.run(["python3", test_file], cwd=work, capture_output=True, timeout=60).returncode == 0
        return ok and same, ("" if same else "test file was modified")
    return grade


def answer_contains(*facts):
    def grade(work, answer):
        missing = [f for f in facts if f not in answer]
        return not missing, ("missing " + ", ".join(missing) if missing else "")
    return grade


TASKS = {
    "stats": ("The test in test_stats.py fails. Run it with: python3 test_stats.py. Fix the bug.",
              tests_pass_unchanged("test_stats.py")),
    "fix": ("The test in test_invoice.py fails. Run it with: python3 test_invoice.py. Find the bug in invoice.py and fix it.",
            tests_pass_unchanged("test_invoice.py")),
    "search": ("In this project, which line decides the file name of a figure image extracted for the study-guide PDF? "
               "Answer with the file path and the line number.", answer_contains("study_guide.py", "42")),
    "server": ("Start the web service with: python3 server.py (it keeps running until stopped). While it runs, check it "
               "with: python3 check.py. Report the version that the service returns.", answer_contains("1.4.2")),
}


def attempt(name):
    task, grade = TASKS[name]
    work = Path(tempfile.mkdtemp()) / f"{name}-work"
    shutil.copytree(HERE / name, work)
    history = [{"role": "user", "content": task}]
    sent, t0, answer = 0, time.time(), ""
    recent = []
    for step in range(1, MAX_STEPS + 1):
        r = respond(history, TOOLS, BASE)
        sent += r["usage"]["input_tokens"]
        calls = [o for o in r["output"] if o["type"] == "function_call"]
        history += r["output"]
        if not calls:
            answer = "".join(c.get("text", "") for o in r["output"] if o["type"] == "message" for c in o["content"])
            break
        for c in calls:
            cmd = json.loads(c["arguments"]).get("command", "")
            try:
                p = subprocess.run(["bash", "-c", cmd], cwd=work, capture_output=True, text=True, timeout=30)
                out = f"exit code {p.returncode}\n{p.stdout}{p.stderr}"
            except subprocess.TimeoutExpired:
                out = "command timed out after 30 s and was killed"
            history.append({"type": "function_call_output", "call_id": c["call_id"], "output": out[-4000:]})
        sig = tuple(sorted({c["arguments"] for c in calls}))
        recent = (recent + [sig])[-3:]
        if len(recent) == 3 and len(set(recent)) == 1:           # episode 2's stuck detector
            print("   stopped: the same tool calls on three steps in a row")
            break
    kill_matching("server.py")
    ok, note = grade(work, answer.split("</think>")[-1])
    return {"task": name, "pass": ok, "note": note, "model_calls": step, "tokens_sent": sent,
            "seconds": round(time.time() - t0)}


def main(k):
    results = []
    for i in range(k):
        for name in TASKS:
            r = attempt(name)
            results.append(r)
            print(f"run {i + 1} {name:7} {'PASS' if r['pass'] else 'FAIL'}  {r['model_calls']:2} calls "
                  f"{r['tokens_sent']:6,} tokens {r['seconds']:4} s  {r['note']}", flush=True)
    print(f"\n{'task':8}{'pass@1':>8}{'pass^' + str(k):>8}{'tokens':>9}{'seconds':>9}")
    for name in TASKS:
        rs = [r for r in results if r["task"] == name]
        print(f"{name:8}{sum(r['pass'] for r in rs) / k:8.2f}{str(all(r['pass'] for r in rs)):>8}"
              f"{sum(r['tokens_sent'] for r in rs) / k:9,.0f}{sum(r['seconds'] for r in rs) / k:9.0f}")
    print(f"overall pass@1: {sum(r['pass'] for r in results) / len(results):.2f}")
    out = Path(__file__).parent / f"results_{datetime.date.today()}.json"
    try:
        out.write_text(json.dumps(results, indent=1))
    except OSError:
        print(json.dumps(results))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
