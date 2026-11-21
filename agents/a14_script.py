"""How AI Agents Work: Inside the Harness, episode 14 — Putting It Together. One entry per narration section.

Numbers come from code/a14_putting_it_together (2026-11-21: tiny_agent.py, Qwen3 4B on Ollama, CPU; commands confined by
bubblewrap) and the earlier episodes' runs; Codex first-request sizes from episodes 4 and 9.
"""
TITLE = "Putting It Together"
TAGLINE = "One tiny agent with every part from this series"
NEXT = None  # series finale
SECTIONS = [
    # 1. Hook
    "Thirteen episodes built a harness piece by piece. Here is all of it in one file: about three hundred lines of "
    "Python, standard library only.",
    # 2. The parts
    "The loop and its stop conditions. Tool calls, with errors returned as results. Search-first instructions and "
    "an output cap. Patches. Sessions for commands that keep running. A sandbox, rules and approvals. The project's "
    "AGENTS file. Compaction. MCP servers. And parallel calls.",
    # 3. The task
    "The task: ticket T forty-two describes a bug in this project. Fix it. The ticket lives in an MCP server. The "
    "project's AGENTS file asks for a changelog line, and every command runs inside the sandbox.",
    # 4. The first request
    "The first request is about six hundred tokens: our instructions, the tools, and the AGENTS file. Codex sends "
    "about eight thousand before the model has done anything.",
    # 5. Run 1
    "In the first run, the model did everything right at the start. It read the ticket, searched, and read the file. "
    "Then it wrapped its patch in the Unix patch command instead of apply patch. The harness didn't recognize it, "
    "patch rejected it, and the model spent its last reply reasoning about line numbers.",
    # 6. Run 2
    "In the second and third runs, it read the ticket, searched, and then replied with nothing. In all three runs the "
    "tests still failed, and nothing outside the project was touched.",
    # 7. What the parts did
    "Every part did its job. The ticket arrived through MCP. The search found the file. The sandbox held. The harness "
    "can make an agent safe, cheap and checkable. It cannot make a four billion parameter model careful.",
    # 8. What we saw
    "That is the pattern of this whole series. The model guessed line numbers, rewrote files it had never read, "
    "wrote tool calls as plain text, and knew rules it did not follow. Each time, a part of the harness either caught "
    "it, or showed us exactly where it went wrong.",
    # 9. Code
    "In code, the whole agent is the same loop as episode two: send the conversation, run the tool calls, append the "
    "results, and stop when the model stops calling tools.",
    # 10. End
    "That's the end of How AI Agents Work. Every number came from code you can run, with a model small enough for a "
    "laptop. Thanks for watching.",
]
