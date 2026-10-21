"""The model on its own: same task, no tools. It can only answer with text."""
from llm import respond

TASK = ("The test in test_stats.py fails. Run it with: python3 test_stats.py. "
        "Find the bug in stats.py, fix it, and run the test again to confirm it passes.")

r = respond(TASK)
text = "".join(c.get("text", "") for o in r["output"] if o["type"] == "message" for c in o["content"])
print(f"{r['usage']['input_tokens']} tokens in, {r['usage']['output_tokens']} out, {r['_seconds']:.0f} s\n")
print(text)
