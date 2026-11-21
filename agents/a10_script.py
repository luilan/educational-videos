"""How AI Agents Work: Inside the Harness, episode 10 — The Context Budget. One entry per narration section.

Numbers come from code/a10_context_budget (2026-11-21: budget_harness.py, Qwen3 4B on Ollama, CPU, in containers without
internet; a simulated 4,096-token window on a server that allows 32,768) and openai/codex @ 89461ea
(prompts/templates/compact/, protocol/src/openai_models.rs, models-manager/src/model_info.rs).
"""
TITLE = "The Context Budget"
TAGLINE = "What happens when the conversation no longer fits"
NEXT = "Extending a Harness"
SECTIONS = [
    # 1. Hook
    "Everything the model reads has to fit in its context window: the instructions, the task, and every tool result "
    "so far. When the conversation grows past it, something has to go.",
    # 2. Codex's numbers
    "Codex assumes two hundred and seventy-two thousand tokens for a model it doesn't know, counts ninety-five "
    "percent of that as usable, and compacts automatically at ninety percent of the window.",
    # 3. Our setup
    "We gave our harness a much smaller window: four thousand and ninety-six tokens. The task: three questions about "
    "three files in the project.",
    # 4. Unlimited
    "With no budget, the model printed all three files at once. Its next request was seven thousand eight hundred "
    "tokens, almost twice the window. It answered all three questions, but only because our server had room to spare.",
    # 5. Compaction
    "So, compaction. When a request passes ninety percent of the window, the harness asks the model for a handoff "
    "summary, with Codex's own prompt, then starts over from just the task and that summary.",
    # 6. Compact runs
    "In three runs, the summary kept about eleven hundred characters. The next request was around five hundred and "
    "fifty tokens, and every answer was still right.",
    # 7. The catch
    "But look at the compaction request itself: seven thousand eight hundred tokens, over the window again. To "
    "summarize, the model has to read everything first. That is why Codex compacts early, and caps every command's "
    "output.",
    # 8. Capped
    "So we also capped every tool result at two thousand characters, as in episode four. Now the largest request was "
    "eighteen hundred tokens, and no compaction was needed. But the line with the cover image sat in the middle of its "
    "file, in the part that was cut. In both runs, the model answered cover dot J P G, from the top of the file. Wrong.",
    # 9. Lesson
    "A budget forces choices. A summary can lose details. A cap cuts the middle. Searching, as in episode four, reads "
    "only the lines that matter, and needs neither.",
    # 10. Code
    "In code, compaction is one more model call, and a much shorter conversation after it.",
    # 11. Outro
    "Next: extending a harness. How new tools plug in, without changing the harness.",
]
