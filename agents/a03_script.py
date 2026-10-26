"""How AI Agents Work: Inside the Harness, episode 3 — Tool Calling Under the Hood. One entry per narration section.

Numbers come from code/a03_tool_calling (2026-10-26: Qwen3's chat template via its tokenizer; 10 raw samples of Qwen3 4B
on Ollama; a Codex CLI 0.163.0 request captured through the logging proxy) and openai/codex @ 89461ea.
"""
TITLE = "Tool Calling Under the Hood"
TAGLINE = "What the model reads, and what it writes, when it calls a tool"
NEXT = "Reading and Searching Code"
SECTIONS = [
    # 1. Hook
    "In episode one, the model called a tool. But a model only writes text. So what does a tool call actually look "
    "like, from the model's side?",
    # 2. What the model reads
    "First, what it reads. The chat template turns our tool definition into instructions inside the system message: "
    "here are the function signatures, and to call one, write a JSON object between tool call tags. Our one small "
    "tool adds a hundred and thirty-nine tokens.",
    # 3. What the model writes
    "Now, what it writes. We sampled the model ten times, with no server in between. All ten wrote the same call: "
    "shell, with the command to run the test. About twenty-five tokens. But first, two hundred and fifty to five "
    "hundred tokens of its own reasoning, even with thinking switched off.",
    # 4. Parsing
    "Something has to find the call in that text. The model server, or the harness, looks for the tool call tags, "
    "parses the JSON inside, and hands over a structured call: a name, the arguments, and a call ID.",
    # 5. The result goes back
    "The result goes back as text too: a new turn, wrapped in tool response tags. The call and its result add just "
    "forty-three tokens. The model's reasoning isn't kept, only the call.",
    # 6. Codex's tool
    "Codex's main tool is exec underscore command, short for execute command. It has ten parameters, though only "
    "the command is required: a working "
    "directory, how long to wait for output, an output budget, a way to ask to run outside the sandbox. This one "
    "definition is four hundred and twenty-one tokens. Ours is fifty-eight.",
    # 7. Codex's result
    "Codex's result carries the same call ID, so the model can match it to its call, plus a header: the exit code, "
    "the wall time, and how long the output was, before the output itself.",
    # 8. When it goes wrong
    "And if the arguments aren't valid JSON? Codex doesn't crash. It sends the error back as the tool's result: "
    "failed to parse function arguments. The model reads it, and can try again.",
    # 9. Code
    "In code, parsing is a search for the tags and a JSON parse. If either fails, the error becomes the result.",
    # 10. Outro
    "Next: reading and searching code. How an agent finds the right lines in a project too big to read.",
]
