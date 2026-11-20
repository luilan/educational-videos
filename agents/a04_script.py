"""How AI Agents Work: Inside the Harness, episode 4 — Reading and Searching Code. One entry per narration section.

Numbers come from code/a04_reading_and_searching (2026-11-20: search_harness.py, 3 runs naive and search, 2 context, Qwen3 4B on Ollama,
CPU, in containers without internet; count_tokens.py with the Qwen3 tokenizer) and openai/codex @ 89461ea
(prompts base_instructions/default.md; utils/string/src/truncate.rs; core/src/unified_exec/mod.rs).
"""
TITLE = "Reading and Searching Code"
TAGLINE = "How an agent finds the right lines in a project too big to read"
NEXT = "Editing Files"
SECTIONS = [
    # 1. Hook
    "Our demo project has twelve files and fourteen hundred lines: about nineteen thousand tokens. The question: "
    "which line names the figure images in the study guide? Every file the agent prints stays in the conversation, "
    "and is sent again on every call.",
    # 2. Naive
    "First, with no guidance. In two runs out of three, the model listed folders, then printed two whole files. "
    "Five calls, almost seven thousand tokens sent. Its answer had the right format string, but from the wrong file.",
    # 3. Invented line numbers
    "And the line number was made up: one hundred and eight in one run, one hundred and seventy-seven in the other. "
    "Cat prints no line numbers, so the model guessed. In the third run, it opened nothing, and named a file that "
    "doesn't exist.",
    # 4. Search first
    "Now the same task, with two sentences of instructions: search first with grep, then print only the lines you "
    "need with sed. Grep shows the file and the line number of every match. Sed shows just a range of lines.",
    # 5. Search results
    "The model searched for the word figure, and got back a list of matching lines instead of whole files. Around "
    "two thousand tokens in total, under a third of the naive runs. And every line number it quoted was real.",
    # 6. The near miss
    "But all three runs gave the same wrong answer: line two twenty-four, where the name is used. The line that makes "
    "the name is forty-two, right below a match the search had already found.",
    # 7. Context
    "So, two more sentences: show a few lines around each match, and when you find where a value is used, search "
    "for where it is set. In one run, the model searched again for fig file. In the other, the context lines were "
    "enough. Both answered line forty-two: right, for two to three thousand tokens.",
    # 8. Output budget
    "Searches can return too much, too: with context lines, one grep returned over five thousand characters. So "
    "harnesses cap every result: keep the beginning and the end, cut the middle, and say how much was cut. Ours "
    "keeps two thousand characters. Codex keeps up to ten thousand tokens per command by default.",
    # 9. Codex
    "Codex's own instructions say: to search for text or files, prefer ripgrep, because it is much faster than grep. "
    "But with our small model, Codex listed two folders and printed a whole file, like our naive runs. Instructions "
    "only help as much as the model follows them.",
    # 10. Code
    "In code, the cap is a few lines: if the output is too long, keep the head and the tail, and put a note in "
    "the middle.",
    # 11. Outro
    "Next: editing files. How an agent changes three lines without rewriting the whole file.",
]
