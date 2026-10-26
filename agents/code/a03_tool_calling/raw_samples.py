"""What the model actually writes: send the rendered prompt to Ollama in raw mode (no server-side parsing) and parse
the <tool_call> ourselves, the way a harness or model server does.

    python3 show_prompt.py > /dev/null   # needs the prompt file written below; or pass your own
    python3 raw_samples.py [n_samples]
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
PROMPT = (Path(__file__).parent / "prompt.txt").read_text()


def parse_tool_call(text):
    """Return (call, error): the JSON inside <tool_call> tags, or why there is none."""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if not m:
        return None, "no tool call"
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        return None, f"failed to parse function arguments: {e}"   # a harness sends this back to the model
    if not isinstance(call.get("arguments"), dict):
        return None, "arguments must be a JSON object"
    return call, None


n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
valid = 0
for i in range(n):
    body = {"model": os.environ.get("MODEL", "qwen3:4b"), "prompt": PROMPT, "raw": True, "stream": False,
            "options": {"temperature": 0.7, "num_predict": 1500, "seed": i}}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
    call, err = parse_tool_call(r["response"])
    valid += call is not None
    start = r["response"].find("<tool_call>")
    print(f"sample {i}: {r['eval_count']} tokens out, {start if start >= 0 else len(r['response'])} characters before the call, "
          f"{time.time() - t:.0f} s -> {call or err}")
print(f"valid tool calls: {valid}/{n}")
