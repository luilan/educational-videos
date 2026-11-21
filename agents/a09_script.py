"""How AI Agents Work: Inside the Harness, episode 9 — Instructions and Memory. One entry per narration section.

Numbers come from code/a09_instructions_and_memory (2026-11-20/21: instructions_harness.py, Qwen3 4B on Ollama, CPU, in
containers without internet; a Codex CLI 0.163.0 request captured through the logging proxy) and openai/codex @ 89461ea
(core/src/agents_md.rs, config/src/config_toml.rs, hooks/src/lib.rs, core/src/hook_runtime.rs).
"""
TITLE = "Instructions and Memory"
TAGLINE = "How a project tells the agent its own rules"
NEXT = "The Context Budget"
SECTIONS = [
    # 1. Hook
    "Every project has its own rules. How to run the tests. What goes in the changelog. What not to touch. An agent "
    "starts every session knowing none of them. Its memory of the project is a file: AGENTS dot M D.",
    # 2. How Codex loads it
    "Codex looks for AGENTS dot M D from the root of the project down to the working folder, joins what it finds, up "
    "to thirty-two kilobytes, and skips it entirely for projects you haven't trusted.",
    # 3. Where it goes
    "Then it sends it as a user message, before your task, wrapped in instructions tags. In our captured request, "
    "Codex's own instructions come first, seventeen thousand characters, then a list of skills, then the project's "
    "three rules, and then the task.",
    # 4. Our rules
    "Our AGENTS dot M D has three rules. Run the tests with run tests dot P Y. Add a line to the changelog for every "
    "fix. Keep the change small.",
    # 5. The experiment
    "We gave the model the same bug fix, in four setups: no AGENTS file. The file in the folder, but not sent. Sent "
    "the Codex way. And put in the system instructions. Nine runs in all.",
    # 6. Results
    "Sending the file cost about a hundred tokens per call. In nine runs, the model never used run tests dot P Y. "
    "The changelog line appeared in only three runs, all with the file sent the Codex way, and always appended at the "
    "end of the file, not under Unreleased. From the system instructions: neither rule, in both runs. Left in the folder, the file "
    "was never opened.",
    # 7. Knowing is not doing
    "Yet when Codex asked the same model which command runs the tests, it answered run tests dot P Y, quoting the "
    "file. The model can read the rules. Following them, step by step, is another matter.",
    # 8. Hooks
    "So a harness enforces what matters. Codex has hooks: a stop hook can refuse to end the turn, and a pre tool use "
    "hook can block a command. Either way, the reason goes back to the model. Our stop check fired once: no changelog "
    "line. The model then added one. But our check only tested that the changelog had changed, so a line in the wrong "
    "place passed. A check is only as good as what it tests.",
    # 9. Code
    "In code, loading the project's instructions is one message, built from a file.",
    # 10. Outro
    "Next: the context budget. What happens when the conversation no longer fits.",
]
