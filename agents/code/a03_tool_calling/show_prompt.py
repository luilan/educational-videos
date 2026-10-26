"""What the model actually reads: render a tool definition, a call and its result through Qwen3's chat template.

    pip install transformers        # only the tokenizer is downloaded (a few MB)
    python3 show_prompt.py
"""
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
TOOLS = [{"type": "function", "function": {"name": "shell", "description": "Run a bash command in the project folder and return its output.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
SYSTEM = {"role": "system", "content": "You are a coding agent working in a project folder. Use the shell tool to inspect, edit and test files."}
USER = {"role": "user", "content": "The test in test_stats.py fails. Run it with: python3 test_stats.py."}
CALL = {"name": "shell", "arguments": {"command": "python3 test_stats.py"}}


def render(msgs, tools=None):
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def n(text):
    return len(tok(text).input_ids)


first = render([SYSTEM, USER], TOOLS)
print("=== what the model reads on the first call ===\n" + first)
print(f"{n(first)} tokens with the tool, {n(render([SYSTEM, USER]))} without it")

rest = [SYSTEM, USER, {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": CALL}]},
        {"role": "tool", "content": "exit code 1\nAssertionError"}]
second = render(rest, TOOLS)
print("\n=== what the model reads on the second call (new part only) ===\n" + second[len(first) - 40:])
print(f"the call and its result add {n(second) - n(first)} tokens; the call's JSON alone is {n(json.dumps(CALL))}")
