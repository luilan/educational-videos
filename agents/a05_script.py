"""How AI Agents Work: Inside the Harness, episode 5 — Editing Files. One entry per narration section.

Numbers come from code/a05_editing_files (2026-11-20: edit_harness.py, Qwen3 4B on Ollama, CPU, in containers without
internet; 3 rewrite runs, 3 patch runs, 2 read-first runs) and openai/codex @ 89461ea (apply-patch/, the apply_patch
grammar, core/src/tools/handlers/apply_patch.rs, protocol/src/prompts/base_instructions/default.md).
"""
TITLE = "Editing Files"
TAGLINE = "How an agent changes one line without rewriting the whole file"
NEXT = "Running Commands"
SECTIONS = [
    # 1. Hook
    "The bug is one line in a hundred-and-eight-line file. The total adds the discount back. How does an agent change "
    "that one line?",
    # 2. Rewrite
    "The simplest way: write the whole file again. We told the model to edit with cat and a heredoc, which replaces "
    "the file with whatever it types. In the first run, it never read the file. It replaced the hundred and eight "
    "lines with a dozen, most of them made up.",
    # 3. Rewrite, after reading
    "In the third run, it did read the file first. Then it typed all four thousand characters back, four times. "
    "Along the way, it dropped the dataclass decorators and broke a class definition. The file no longer even ran.",
    # 4. The cost
    "A rewrite costs the whole file in output tokens, on every edit, and every line is a chance to copy something "
    "wrong. The fix itself is one line.",
    # 5. The patch format
    "So Codex edits with patches. A patch names the file, then shows only what changes: lines starting with minus "
    "are removed, plus lines are added, and a few unchanged lines around them say where.",
    # 6. Intercepted
    "For models like ours, Codex has no separate edit tool. The model runs a command called apply patch, and Codex "
    "catches it before it reaches the shell, checks it, and applies it itself. Our harness does the same.",
    # 7. Matching
    "The unchanged lines must match the file. Codex tries an exact match, then ignores spaces at line ends, then "
    "around the whole line, then fancy punctuation. If nothing matches, the edit is refused, and the error goes back "
    "to the model.",
    # 8. Patch runs
    "In our five patch runs, the model never broke the file. Three times it drafted a patch for code it hadn't read, "
    "for a line that says return sum of items plus ten. That line doesn't exist, so the patch would have been refused.",
    # 9. The right patch
    "In one run, it read the file and worked out the right fix, but its reasoning ran past our token cap before it "
    "sent the call. Sent as a patch, that fix is two hundred and seventy-seven characters, and the test passes.",
    # 10. Code
    "In code, applying a patch is a search: find the old lines, from strict to loose, and replace them. If they "
    "aren't there, say so.",
    # 11. Outro
    "Next: running commands. What a harness does with a command that never ends.",
]
