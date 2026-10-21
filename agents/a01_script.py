"""How AI Agents Work: Inside the Harness, episode 1 — What a Harness Is. One entry per narration section. DRAFT for review.

Every number comes from runs on 2026-10-19 (code/a01_what_is_a_harness, and Codex CLI 0.163.0 traced through a logging
proxy), all with Qwen3 4B served by Ollama 0.40.6 on a CPU, inside containers on a network with no internet access.
"""
TITLE = "What a Harness Is"
TAGLINE = "Everything an agent does happens outside the model"
NEXT = "The Agent Loop"
SECTIONS = [
    # 1. Hook
    "A language model can't open a file, run a command, or check its own work. It reads text and writes text. "
    "Everything an AI agent does happens in the program around the model: the harness.",
    # 2. The model alone
    "Here is a failing test, and a small model, Kwen three, four billion parameters. Ask it to fix the bug, with no "
    "tools. It invents a median function that doesn't exist, fixes a bug that isn't there, and says the test now "
    "passes. It never saw a single file.",
    # 3. The trick
    "The trick: describe a tool to the model. When it answers with a tool call instead of prose, the harness runs "
    "it and pastes the result back into the conversation. Then it asks again, until the model answers in plain "
    "words.",
    # 4. A tiny harness
    "Here is a whole harness: forty lines of Python, one tool, a shell. Three model calls. The model runs the test, "
    "sees the assertion fail, and rewrites stats dot py. The harness checks for itself: the test passes. Notice "
    "that it never read the file before rewriting it. We'll come back to that.",
    # 5. A real harness
    "Now a real one: Codex CLI, open source. Same task. Your hundred and forty-seven characters reach the model "
    "wrapped in about eight thousand tokens: instructions, a list of skills, the environment, and nine tool "
    "definitions.",
    # 6. Who does what
    "The harness owns the loop, the tools, the sandbox and the approvals, the instructions, the context window, "
    "and the record of what happened. The model owns one thing: choosing the next words.",
    # 7. A warning from the trace
    "And the details matter. Our local model server silently dropped Codex's instructions: the model saw less "
    "than half of what was sent. With them restored, it wrote its fix as a patch, in plain text, instead of "
    "calling a tool. Nothing ran. Text only becomes an action when the harness recognizes it.",
    # 8. Code
    "In code, the core is one loop: while the model asks for a tool, run it, and add the result to the "
    "conversation.",
    # 9. Outro
    "Next: the agent loop, step by step, and how a harness decides when to stop.",
]
