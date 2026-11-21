"""Sub-agents and parallel tool calls.

    python3 parallel_harness.py serial     # three slow checks; the model may ask for all three at once; run one by one
    python3 parallel_harness.py parallel   # same, but calls from one reply run at the same time (threads)
    python3 parallel_harness.py single     # the episode-4 search task, done by one agent
    python3 parallel_harness.py subagent   # the parent gets a spawn_agent tool; a child with a fresh context searches,
                                           # and only its final answer comes back to the parent

WARNING: runs the model's commands. Use a container or a throwaway VM.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm import respond

HERE = Path(__file__).parent
CHECKS_TASK = ("Run all three checks and report their results: python3 check.py lint, python3 check.py types and "
               "python3 check.py tests. They are independent: request all three tool calls in one reply.")
SEARCH_TASK = ("In this project, which line decides the file name of a figure image extracted for the study-guide PDF? "
               "Answer with the file path, the line number and the exact format string.")
BASE = "You are a coding agent working in a project folder. Use your tools."
SEARCH = (BASE + " The project is too big to read whole. Search first with `grep -rn`, then read only the lines you need "
          "with `sed -n 'START,ENDp' FILE`.")
PARENT = (BASE + " Delegate searching to a sub-agent with spawn_agent: give it a complete, self-contained task. Its "
          "final answer comes back to you as the tool result.")
SHELL = {"type": "function", "name": "shell", "description": "Run a bash command in the project folder and return its output.",
         "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}
SPAWN = {"type": "function", "name": "spawn_agent",
         "description": "Start a sub-agent with its own fresh conversation and the shell tool. It works on `message` "
                        "until done; its final answer is returned to you.",
         "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}
MAX_STEPS = 10


def shell(cmd, work):
    p = subprocess.run(["bash", "-c", cmd], cwd=work, capture_output=True, text=True, timeout=60)
    return (p.stdout + p.stderr) or "(no output)"


def agent(task, tools, instructions, work, name, parallel=False, depth=0):
    """One agent loop. Returns (final answer, tokens sent in total, model calls)."""
    history = [{"role": "user", "content": task}]
    sent = 0
    pad = "      " * depth
    recent = []
    for step in range(1, MAX_STEPS + 1):
        r = respond(history, tools, instructions)
        sent += r["usage"]["input_tokens"]
        calls = [o for o in r["output"] if o["type"] == "function_call"]
        history += r["output"]
        print(f"{pad}{name} step {step}: sent {r['usage']['input_tokens']:>5} tokens, {len(calls)} tool calls in the reply")
        if not calls:
            text = "".join(c.get("text", "") for o in r["output"] if o["type"] == "message" for c in o["content"])
            return text.split("</think>")[-1].strip(), sent, step

        def do(c):
            a = json.loads(c["arguments"])
            if c["name"] == "spawn_agent":
                print(f"{pad}   spawn_agent: {a.get('message', '')[:100]!r}")
                ans, child_sent, child_steps = agent(a.get("message", ""), [SHELL], SEARCH, work, "child", depth=depth + 1)
                print(f"{pad}   child finished: {child_steps} model calls, {child_sent:,} tokens sent (not in our context)")
                return ans or "(the sub-agent gave no answer)"
            return shell(a.get("command", ""), work)

        t0 = time.time()
        if parallel:
            with ThreadPoolExecutor() as pool:
                outs = list(pool.map(do, calls))
        else:
            outs = [do(c) for c in calls]
        for c, out in zip(calls, outs):
            if c["name"] == "shell":
                print(f"{pad}   $ {json.loads(c['arguments']).get('command', '')[:80]}")
            history.append({"type": "function_call_output", "call_id": c["call_id"], "output": out[-4000:]})
        sig = tuple(sorted({c["arguments"] for c in calls}))
        recent = (recent + [sig])[-3:]
        if len(recent) == 3 and len(set(recent)) == 1:           # episode 2's stuck detector
            print("   stopped: the same tool calls on three steps in a row")
            break
        print(f"{pad}   tools took {time.time() - t0:.1f} s")
    return "", sent, MAX_STEPS


def run(mode):
    work = Path(tempfile.mkdtemp(prefix="par-"))
    t0 = time.time()
    if mode in ("serial", "parallel"):
        shutil.copytree(HERE / "demo_checks", work, dirs_exist_ok=True)
        ans, sent, steps = agent(CHECKS_TASK, [SHELL], BASE, work, "agent", parallel=(mode == "parallel"))
        ok = all(s in ans for s in ("0 problems", "0 errors", "14 passed"))
    else:
        shutil.copytree(HERE / "demo_project", work, dirs_exist_ok=True)
        if mode == "single":
            ans, sent, steps = agent(SEARCH_TASK, [SHELL], SEARCH, work, "agent")
        else:
            ans, sent, steps = agent(SEARCH_TASK, [SPAWN], PARENT, work, "parent")
        ok = "study_guide.py" in ans and "42" in ans
    print(f"\n{mode}: {steps} model calls, {sent:,} tokens sent by the top agent, {time.time() - t0:.0f} s in total")
    print("answer (end): " + ans[-300:].replace("\n", " "))
    print("check:", "RIGHT" if ok else "WRONG or incomplete")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "parallel")
