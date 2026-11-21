"""A tiny coding agent with every part from the series, in one file (Python standard library only).

    python3 tiny_agent.py "Ticket T-42 describes a bug. Fix it."      # works in ./demo (a copy of it)

Parts, and the episode that explained each one:
  agent loop with stop conditions ............. 2    tool calls, errors returned as results ........ 3
  search-first instructions, output truncation  4    apply_patch edits (intercepted) ............... 5
  exec_command / write_stdin sessions ......... 6    bubblewrap sandbox (Linux) .................... 7
  rules + on-request approvals ................ 8    AGENTS.md instructions ........................ 9
  compaction at 90% of the context window .... 10    MCP servers ................................... 11
  parallel tool calls ........................ 12
Needs Linux with bubblewrap (bwrap) and a local Ollama with qwen3:4b (OLLAMA_URL, MODEL).
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm import respond

HERE = Path(__file__).parent
INSTRUCTIONS = """You are a coding agent working in a project folder.
- Search before reading: `grep -rn PATTERN .`, then `sed -n 'START,ENDp' FILE`. Never print whole large files.
- Edit files only with apply_patch (run it with exec_command):
apply_patch <<'EOF'
*** Begin Patch
*** Update File: path/to/file.py
@@ def example():
     unchanged line
-    old line
+    new line
*** End Patch
EOF
- Commands run in a sandbox: no network, write access only to the project folder. If a command needs more, run it
  again with sandbox_permissions "require_escalated" and a justification; the user will be asked.
- Independent tool calls can go in one reply; they run at the same time."""
TOOLS = [
    {"type": "function", "name": "exec_command",
     "description": "Run a command. Returns its output after yield_time_ms (default 10000) or when it exits. If it is "
                    "still running, the result gives a session_id for write_stdin.",
     "parameters": {"type": "object", "properties": {
         "cmd": {"type": "string"}, "yield_time_ms": {"type": "integer"},
         "sandbox_permissions": {"type": "string", "enum": ["use_default", "require_escalated"]},
         "justification": {"type": "string"}}, "required": ["cmd"]}},
    {"type": "function", "name": "write_stdin",
     "description": "Send characters to a running session (may be empty, to read new output).",
     "parameters": {"type": "object", "properties": {"session_id": {"type": "integer"}, "chars": {"type": "string"},
                                                     "yield_time_ms": {"type": "integer"}}, "required": ["session_id"]}},
]
RULES = [(["rm", "-rf"], "forbidden"), (["pip", "install"], "prompt"), (["git", "push"], "prompt")]   # episode 8
WINDOW, MAX_STEPS, OUTPUT_LIMIT = 32768, 20, 4000


# ------------------------------------------------------------------ episode 4: keep the head and the tail
def truncate_middle(text, limit=OUTPUT_LIMIT):
    if len(text) <= limit:
        return text
    return f"{text[:limit // 2]}\n…{len(text) - limit} chars truncated…\n{text[-(limit - limit // 2):]}"


# ------------------------------------------------------------------ episode 5: apply_patch
def apply_patch(cmd, root):
    body = cmd.split("*** Begin Patch", 1)[1].split("*** End Patch", 1)[0].strip("\n").split("\n")
    if not body[0].startswith("*** Update File: "):
        return "apply_patch failed: expected '*** Update File: <path>'"
    path = root / body[0][len("*** Update File: "):].strip()
    if not path.resolve().is_relative_to(root.resolve()):           # the patch tool runs outside the sandbox
        return "apply_patch failed: path is outside the project folder"
    if not path.is_file():
        return f"apply_patch failed: {path.name} does not exist"
    lines, chunks = path.read_text().split("\n"), [([], [])]
    for ln in body[1:]:
        if ln.startswith("@@"):
            chunks.append(([], []))
            continue
        tag, text = (ln[:1], ln[1:]) if ln[:1] in "+- " and ln else (" ", ln)
        if tag in " -":
            chunks[-1][0].append(text)
        if tag in " +":
            chunks[-1][1].append(text)
    pos = 0
    for old, new in (c for c in chunks if c != ([], [])):
        for norm in (lambda s: s, str.rstrip, str.strip):
            i = next((i for i in range(pos, len(lines) - len(old) + 1)
                      if all(norm(lines[i + j]) == norm(o) for j, o in enumerate(old))), None)
            if i is not None:
                break
        if i is None:
            return "apply_patch failed: could not find these lines:\n" + "\n".join(old)
        lines[i:i + len(old)] = new
        pos = i + len(new)
    path.write_text("\n".join(lines))
    return f"Success. Updated the following files:\nM {path.name}"


# ------------------------------------------------------------------ episode 7: the sandbox
def bwrap(work, home, network=False):
    args = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp"] + ([] if network else ["--unshare-net"])
    for d in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(d).exists():
            args += ["--ro-bind", d, d]
    return args + ["--ro-bind", str(home), "/home/agent", "--setenv", "HOME", "/home/agent",
                   "--bind", str(work), str(work), "--chdir", str(work)]


# ------------------------------------------------------------------ episode 8: rules, then policy, then the user
def decide(cmd, escalated):
    try:
        words = shlex.split(cmd)
    except ValueError:
        words = cmd.split()
    for prefix, decision in RULES:
        if words[:len(prefix)] == prefix:
            return decision
    return "prompt" if escalated else "allow"


def ask_user(cmd, justification):
    if not sys.stdin.isatty():
        return False                                         # nobody to ask: deny
    return input(f"\nApprove `{cmd}`? ({justification}) [y/N] ").strip().lower() == "y"


# ------------------------------------------------------------------ episode 6: sessions
class Session:
    def __init__(self, argv, cwd):
        self.p = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        self.buf, self.lock = [], threading.Lock()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            with self.lock:
                self.buf.append(line)

    def collect(self, ms):
        end = time.time() + min(max(ms, 250), 30000) / 1000
        while time.time() < end and self.p.poll() is None:
            time.sleep(0.05)
        time.sleep(0.05)
        with self.lock:
            out, self.buf = "".join(self.buf), []
        return out


# ------------------------------------------------------------------ episode 11: MCP over stdio
class MCPClient:
    def __init__(self, name, argv):
        self.name, self.n = name, 0
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "tiny-agent", "version": "1.0"}})
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")

    def request(self, method, params=None):
        self.n += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())["result"]

    def tools(self):
        return [{"type": "function", "name": f"mcp__{self.name}__{t['name']}", "description": t.get("description", ""),
                 "parameters": t["inputSchema"]} for t in self.request("tools/list")["tools"]]

    def call(self, tool, args):
        return "\n".join(c.get("text", "") for c in self.request("tools/call", {"name": tool, "arguments": args})["content"])


# ------------------------------------------------------------------ the agent
class Agent:
    def __init__(self, work):
        self.work = work
        self.home = Path(tempfile.mkdtemp(prefix="agent-home-"))
        self.sessions = {}
        self.mcp = {"tickets": MCPClient("tickets", [sys.executable, str(HERE / "tickets_mcp.py")])}
        self.tools = TOOLS + [t for m in self.mcp.values() for t in m.tools()]
        self.log = {"commands": 0, "patches": 0, "blocked": 0, "asked": 0, "compactions": 0, "parallel": 0}

    def history_start(self, task):
        msgs = []
        agents_md = self.work / "AGENTS.md"                                  # episode 9
        if agents_md.exists():
            msgs.append({"role": "user", "content": f"# AGENTS.md instructions for {self.work}\n\n<INSTRUCTIONS>\n"
                                                    f"{agents_md.read_text()[:32 * 1024]}\n</INSTRUCTIONS>"})
        return msgs + [{"role": "user", "content": task}]

    def call_tool(self, c):                                                   # episode 3: errors become results
        try:
            a = json.loads(c["arguments"])
        except json.JSONDecodeError as e:
            return f"failed to parse function arguments: {e}"
        if c["name"].startswith("mcp__"):
            _, server, tool = c["name"].split("__", 2)
            return self.mcp[server].call(tool, a)
        if c["name"] == "write_stdin":
            s = self.sessions.get(a.get("session_id"))
            if s is None:
                return f"unknown session_id {a.get('session_id')}"
            if a.get("chars"):
                s.p.stdin.write(a["chars"]); s.p.stdin.flush()
            out = s.collect(a.get("yield_time_ms", 250))
            return f"Process {'still running' if s.p.poll() is None else f'exited with code {s.p.returncode}'}\n{out}"
        cmd = a.get("cmd", "")
        self.log["commands"] += 1
        if cmd.lstrip().startswith("apply_patch") and "*** Begin Patch" in cmd:
            self.log["patches"] += 1
            return apply_patch(cmd, self.work)
        esc = a.get("sandbox_permissions") == "require_escalated"
        d = decide(cmd, esc)
        if d == "forbidden":
            self.log["blocked"] += 1
            return "forbidden by policy"
        if d == "prompt":
            self.log["asked"] += 1
            if not ask_user(cmd, a.get("justification", "")):
                return "the user denied this command"
        s = Session(bwrap(self.work, self.home, network=(d == "prompt")) + ["bash", "-c", cmd], self.work)
        out = s.collect(a.get("yield_time_ms", 10000))
        if s.p.poll() is None:
            sid = len(self.sessions) + 1
            self.sessions[sid] = s
            return f"Process running with session ID {sid}\nOutput:\n{truncate_middle(out)}"
        if any(m in out for m in ("Read-only file system", "Temporary failure in name resolution")):
            self.log["blocked"] += 1
        return f"Process exited with code {s.p.returncode}\nOutput:\n{truncate_middle(out)}"

    def compact(self, history, task):                                         # episode 10
        r = respond(history + [{"role": "user", "content": (HERE / "compact_prompt.md").read_text()}])
        summary = "".join(c.get("text", "") for o in r["output"] if o["type"] == "message"
                          for c in o["content"]).split("</think>")[-1].strip()
        self.log["compactions"] += 1
        return self.history_start(task) + [
            {"role": "user", "content": (HERE / "compact_summary_prefix.md").read_text() + "\n" + summary}]

    def run(self, task):                                                      # episode 2: the loop
        history, sent = self.history_start(task), []
        recent = []
        for step in range(1, MAX_STEPS + 1):
            r = respond(history, self.tools, INSTRUCTIONS)
            sent.append(r["usage"]["input_tokens"])
            calls = [o for o in r["output"] if o["type"] == "function_call"]
            history += r["output"]
            print(f"step {step}: sent {sent[-1]:>6,} tokens, {len(calls)} tool call(s)")
            if not calls:                                                     # no tool call: the turn is over
                text = "".join(c.get("text", "") for o in r["output"] if o["type"] == "message" for c in o["content"])
                return text.split("</think>")[-1].strip(), step, sent
            if len(calls) > 1:
                self.log["parallel"] += 1
            with ThreadPoolExecutor() as pool:                                # episode 12
                outs = list(pool.map(self.call_tool, calls))
            for c, out in zip(calls, outs):
                first = json.loads(c["arguments"]).get("cmd", c["arguments"]) if c["arguments"].startswith("{") else ""
                print(f"   {c['name']}: {str(first).splitlines()[0][:90] if first else ''}  -> {out.splitlines()[0][:60] if out else ''}")
                history.append({"type": "function_call_output", "call_id": c["call_id"], "output": out})
            sig = tuple(sorted({c["arguments"] for c in calls}))
            recent = (recent + [sig])[-3:]
            if len(recent) == 3 and len(set(recent)) == 1:           # episode 2's stuck detector
                print("   stopped: the same tool calls on three steps in a row")
                break
            if sent[-1] > WINDOW * 9 // 10:
                history = self.compact(history, task)
        return "(stopped: step limit)", MAX_STEPS, sent

    def close(self):
        for s in self.sessions.values():
            if s.p.poll() is None:
                os.killpg(s.p.pid, 9)


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "Ticket T-42 describes a bug in this project. Fix it."
    work = Path(tempfile.mkdtemp(prefix="tiny-agent-"))
    shutil.copytree(HERE / "demo", work, dirs_exist_ok=True)
    agent = Agent(work)
    t0 = time.time()
    answer, steps, sent = agent.run(task)
    agent.close()
    tests = subprocess.run([sys.executable, "run_tests.py"], cwd=work, capture_output=True, text=True)
    changelog = re.search(r"## Unreleased\s+- Fixed", (work / "CHANGELOG.md").read_text()) is not None
    print(f"\n{steps} model calls, {sum(sent):,} tokens sent, {time.time() - t0:.0f} s; " +
          ", ".join(f"{k} {v}" for k, v in agent.log.items()))
    print(f"tests pass: {tests.returncode == 0}; CHANGELOG line: {changelog}")
    print("answer (end): " + answer[-400:].replace("\n", " "))
