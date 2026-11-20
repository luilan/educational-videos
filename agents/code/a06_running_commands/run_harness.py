"""Running commands: a task that needs a process to keep running, in two harness designs.

    python3 run_harness.py blocking   # every command runs to the end (killed after 30 s), like episodes 1-5
    python3 run_harness.py sessions   # Codex-style: a command returns after a short wait; if it is still running it
                                      # keeps a session ID, and write_stdin can read more output or type into it

Task: start demo/server.py (it never exits), then query it with demo/check.py.
WARNING: runs the model's commands. Use a container or a throwaway VM.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from llm import respond

TASK = ("Start the web service with: python3 server.py (it keeps running until stopped). While it runs, check it with: "
        "python3 check.py. Report the version that the service returns.")
BASE = "You are a coding agent working in a project folder. Use your tools to run commands."
SHELL = [{"type": "function", "name": "shell", "description": "Run a bash command in the project folder and return its output.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
SESSIONS = [
    {"type": "function", "name": "exec_command",
     "description": "Run a command. Returns its output after yield_time_ms (default 10000) or when it exits, whichever "
                    "comes first. If the command is still running, the result gives a session_id: use write_stdin "
                    "to read more output or send input.",
     "parameters": {"type": "object", "properties": {"cmd": {"type": "string"},
                                                     "yield_time_ms": {"type": "integer"}}, "required": ["cmd"]}},
    {"type": "function", "name": "write_stdin",
     "description": "Write characters to a running session's stdin (may be empty, to just read new output). Returns the "
                    "output produced after yield_time_ms (default 250).",
     "parameters": {"type": "object", "properties": {"session_id": {"type": "integer"}, "chars": {"type": "string"},
                                                     "yield_time_ms": {"type": "integer"}}, "required": ["session_id"]}},
]
MAX_STEPS = 10
TIMEOUT = 30


class Session:
    """A process that keeps running between tool calls; a thread collects its output."""
    def __init__(self, cmd, cwd):
        self.p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        self.buf, self.lock = [], threading.Lock()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            with self.lock:
                self.buf.append(line)

    def collect(self, yield_ms):
        end = time.time() + yield_ms / 1000
        while time.time() < end and self.p.poll() is None:
            time.sleep(0.05)
        time.sleep(0.05)
        with self.lock:
            out, self.buf = "".join(self.buf), []
        return out



def kill_matching(word):
    """Stop leftover processes whose command line contains `word` (like pkill -f, which slim images lack)."""
    for d in os.listdir("/proc"):
        try:
            if d.isdigit() and int(d) != os.getpid() and word in open(f"/proc/{d}/cmdline").read():
                os.kill(int(d), 9)
        except (OSError, ProcessLookupError):
            pass

def run(mode):
    work = Path(tempfile.mkdtemp())
    shutil.copytree(Path(__file__).parent / "demo", work, dirs_exist_ok=True)
    history = [{"role": "user", "content": TASK}]
    sessions, waited = {}, 0.0
    t_start = time.time()
    recent = []
    answer = ""
    for step in range(1, MAX_STEPS + 1):
        r = respond(history, SESSIONS if mode == "sessions" else SHELL, BASE)
        calls = [o for o in r["output"] if o["type"] == "function_call"]
        history += r["output"]
        print(f"step {step}: model {r['_seconds']:4.0f} s")
        if not calls:
            answer = "".join(c.get("text", "") for o in r["output"] if o["type"] == "message" for c in o["content"])
            break
        for c in calls:
            a = json.loads(c["arguments"])
            t0 = time.time()
            if c["name"] == "shell":
                try:
                    p = subprocess.run(["bash", "-c", a.get("command", "")], cwd=work, capture_output=True, text=True,
                                       timeout=TIMEOUT)
                    out = f"exit code {p.returncode}\n{p.stdout}{p.stderr}"
                except subprocess.TimeoutExpired as e:
                    out = f"command timed out after {TIMEOUT} s and was killed\n{(e.stdout or b'').decode(errors='replace')}"
                shown = a.get("command", "")
            elif c["name"] == "exec_command":
                s = Session(a.get("cmd", ""), work)
                text = s.collect(min(max(a.get("yield_time_ms", 10000), 250), 30000))
                if s.p.poll() is None:
                    sid = len(sessions) + 1
                    sessions[sid] = s
                    out = f"Process running with session ID {sid}\nOutput:\n{text}"
                else:
                    out = f"Process exited with code {s.p.returncode}\nOutput:\n{text}"
                shown = a.get("cmd", "")
            else:  # write_stdin
                s = sessions.get(a.get("session_id"))
                if s is None:
                    out = f"unknown session_id {a.get('session_id')}"
                else:
                    if a.get("chars"):
                        s.p.stdin.write(a["chars"]); s.p.stdin.flush()
                    text = s.collect(min(max(a.get("yield_time_ms", 250), 250), 30000))
                    state = "still running" if s.p.poll() is None else f"exited with code {s.p.returncode}"
                    out = f"Process {state}\nOutput:\n{text}"
                shown = f"write_stdin {a}"
            waited += time.time() - t0
            print(f"   {c['name']}: {shown[:80]}   [{time.time() - t0:4.1f} s]  -> {out.splitlines()[0][:60]}")
            history.append({"type": "function_call_output", "call_id": c["call_id"], "output": out[:4000]})
        sig = tuple(sorted({c["arguments"] for c in calls}))
        recent = (recent + [sig])[-3:]
        if len(recent) == 3 and len(set(recent)) == 1:           # episode 2's stuck detector
            print("   stopped: the same tool calls on three steps in a row")
            break
    else:
        answer = ""
    for s in sessions.values():
        if s.p.poll() is None:
            os.killpg(s.p.pid, 9)
    kill_matching("server.py")
    final = answer.split("</think>")[-1]
    print(f"\n{mode}: {step} model calls, {waited:.0f} s waiting for commands, {time.time() - t_start:.0f} s total")
    print("answer (end): " + final.strip()[-300:].replace("\n", " "))
    print("check:", "RIGHT (1.4.2)" if "1.4.2" in final else "WRONG or incomplete")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "sessions")
