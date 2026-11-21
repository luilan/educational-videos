"""Extending a harness: plug in an MCP server, and its tools appear next to the built-in ones.

    python3 mcp_harness.py            # starts tickets_mcp.py, lists its tools, gives them to the model

The model gets `shell` (built in) plus `mcp__tickets__get_ticket` (from the server; Codex uses the same
mcp__<server>__<tool> naming). Calls to an MCP tool are forwarded to the server as JSON-RPC tools/call.
WARNING: runs the model's commands. Use a container or a throwaway VM.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from llm import respond

HERE = Path(__file__).parent
TASK = "Ticket T-42 describes a bug in this project. Read the ticket, fix the bug, and run python3 test_invoice.py."
BASE = "You are a coding agent working in a project folder. Use your tools."
SHELL = {"type": "function", "name": "shell", "description": "Run a bash command in the project folder and return its output.",
         "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}
MAX_STEPS = 10


class MCPClient:
    """Talks JSON-RPC 2.0 to an MCP server over its stdin/stdout."""
    def __init__(self, name, argv):
        self.name, self.n = name, 0
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "tiny-harness", "version": "0.1"}})
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")

    def request(self, method, params=None):
        self.n += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())["result"]

    def tools(self):
        """MCP tool definitions, renamed and reshaped into the model's function-tool format."""
        return [{"type": "function", "name": f"mcp__{self.name}__{t['name']}", "description": t.get("description", ""),
                 "parameters": t["inputSchema"]} for t in self.request("tools/list")["tools"]]

    def call(self, tool, arguments):
        r = self.request("tools/call", {"name": tool, "arguments": arguments})
        return "\n".join(c.get("text", "") for c in r["content"])


def run():
    work = Path(tempfile.mkdtemp(prefix="mcp-"))
    shutil.copytree(HERE / "demo", work, dirs_exist_ok=True)
    servers = {"tickets": MCPClient("tickets", [sys.executable, str(HERE / "tickets_mcp.py")])}
    tools = [SHELL] + [t for s in servers.values() for t in s.tools()]
    print("tools offered to the model:", ", ".join(t["name"] for t in tools))
    print(f"definition sizes (chars): " + ", ".join(f"{t['name']} {len(json.dumps(t))}" for t in tools))
    history = [{"role": "user", "content": TASK}]
    recent = []
    for step in range(1, MAX_STEPS + 1):
        r = respond(history, tools, BASE)
        calls = [o for o in r["output"] if o["type"] == "function_call"]
        history += r["output"]
        print(f"step {step}: model {r['_seconds']:4.0f} s")
        if not calls:
            break
        for c in calls:
            args = json.loads(c["arguments"])
            if c["name"].startswith("mcp__"):
                _, server, tool = c["name"].split("__", 2)
                out = servers[server].call(tool, args)                     # forwarded over JSON-RPC
                print(f"   MCP {server}.{tool}({args})   -> {out[:70]}…")
            else:
                cmd = args.get("command", "")
                p = subprocess.run(["bash", "-c", cmd], cwd=work, capture_output=True, text=True, timeout=30)
                out = f"exit code {p.returncode}\n{p.stdout}{p.stderr}"
                print(f"   $ {cmd.splitlines()[0][:80] if cmd else ''}{' …' if cmd.count(chr(10)) else ''}   -> exit {p.returncode}")
            history.append({"type": "function_call_output", "call_id": c["call_id"], "output": out[-4000:]})
        sig = tuple(sorted({c["arguments"] for c in calls}))
        recent = (recent + [sig])[-3:]
        if len(recent) == 3 and len(set(recent)) == 1:           # episode 2's stuck detector
            print("   stopped: the same tool calls on three steps in a row")
            break
    test = subprocess.run(["python3", "test_invoice.py"], cwd=work, capture_output=True, text=True)
    print(f"\n{step} model calls; tests pass: {test.returncode == 0}")


if __name__ == "__main__":
    run()
