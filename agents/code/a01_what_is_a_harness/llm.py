"""One call to a local model through the OpenAI Responses API, as served by Ollama."""
import json
import os
import time
import urllib.error
import urllib.request

URL = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/v1/responses"
MODEL = os.environ.get("MODEL", "qwen3:4b")


def respond(input_items, tools=None, instructions=None):
    body = {"model": MODEL, "input": input_items, "reasoning": {"effort": "none"}}
    if tools:
        body["tools"] = tools
    if instructions:
        body["instructions"] = instructions
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    for attempt in range(3):                          # servers fail now and then: retry, like real harnesses do
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                out = json.load(r)
            break
        except urllib.error.HTTPError as e:
            print(f"  (model server error {e.code}: {e.read()[:200]!r}; retrying)")
            if attempt == 2:
                raise
            time.sleep(5)
    out["_seconds"] = time.time() - t0
    return out
