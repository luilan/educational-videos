"""How AI Agents Work: Inside the Harness, episode 2 — The Agent Loop. One entry per narration section.

Numbers come from code/a02_the_agent_loop/agent_loop.py (2026-10-21, Qwen3 4B on Ollama, CPU, in a container) and from
the Codex CLI 0.163.0 trace of episode 1 (5 model calls, usage reported by Codex).
"""
TITLE = "The Agent Loop"
TAGLINE = "What happens each time around, and what makes it stop"
NEXT = "Tool Calling Under the Hood"
SECTIONS = [
    # 1. Hook
    "Every agent is one loop. The interesting questions are what happens each time around, and what makes it stop.",
    # 2. One step
    "One step: send the conversation to the model. If the reply contains tool calls, run them, add each result to "
    "the conversation, and call the model again. If it doesn't, the turn is over.",
    # 3. A real run
    "Here is a real run with our harness and Kwen three. Five model calls: run the test, and it fails. Read stats "
    "dot py. Rewrite it. Run the test again, and it passes. Then answer in words. This time the model read the file "
    "first, and checked its own work.",
    # 4. It grows
    "Notice what gets sent. The model remembers nothing between calls, so every step sends the whole conversation "
    "again: one hundred ninety-seven tokens, then three hundred twenty-two, up to four hundred ninety-two. In Codex, "
    "the same five steps grew from three thousand seven hundred to almost eight thousand tokens. Eighty-six percent "
    "came from the server's cache.",
    # 5. How Codex stops
    "How does Codex decide to stop? Every tool call sets a flag: needs follow-up. When a reply has no tool call, the "
    "flag stays false and the turn ends, after the stop hooks run. A user interrupt, an error, or a budget can end it "
    "too. There is no fixed step limit.",
    # 6. A task that never ends
    "So what if the task can never finish? We asked the model to wait for a file that never appears. It wrote a "
    "shell loop that waits forever. The harness killed the command after thirty seconds, and on the next call the "
    "model gave up and explained. The safety net was the timeout, not the model.",
    # 7. Our safeguards
    "Our tiny harness adds three guards: a timeout on every command, a limit of eight model calls, and a stuck "
    "detector, the same command three times in a row. In this run, only the timeout was needed.",
    # 8. Code
    "In code, each guard is one line in the loop, and every exit records why the loop stopped.",
    # 9. Outro
    "Next: tool calling under the hood. What the model actually sees, and what it writes, when it calls a tool.",
]
