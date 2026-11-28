"""The Frontier: Self-Improvement and Memory, episode 4 — A Model That Edits Itself. One entry per narration section.

Every number comes from code/x04_a_model_that_edits_itself (self_edit.py, 2026-11-28): Qwen3.5-0.8B proposes 100
edits to its own RMSNorm weights over 25 rounds, against 100 random edits in the same loop; plus the Self-WSP
preprint (arXiv 2611.28833) as quoted from its abstract.
"""
TITLE = "A Model That Edits Itself"
TAGLINE = "The editor and the edited become one network"
NEXT = "Diagnosing Its Own Mistakes"
SECTIONS = [
    # 1. Hook
    "In November, a preprint called Self W S P reported something strange. A small model, Kwen three point five, point "
    "eight billion parameters, reads its own weights, edits them, and keeps going as the edited model. On school maths "
    "it went from twelve percent to sixty-two, sometimes by changing only sixteen numbers.",
    # 2. Setup
    "We tried it at our scale, with the same model on a C P U. The task: add three three-digit numbers. It gets "
    "twenty-three percent of forty-eight practice problems right, and twenty-six percent of two hundred held-out ones.",
    # 3. The loop
    "Each round, it sees a summary of its own normalization weights and its wrong answers, and writes four edits. "
    "We test each one on the practice problems and keep the best, if it helps. Then the edited model writes the next "
    "edits. Twenty-five rounds, a hundred edits.",
    # 4. What it writes
    "It can do it. Ninety-five of its hundred replies were valid edits. But they look alike: mostly scaling a "
    "whole vector up by ten to fifty percent, and a quarter of them on the very last layer.",
    # 5. Result
    "Three edits were kept. Practice score: from twenty-three percent to thirty-five. On the held-out problems: from "
    "twenty-six percent to twenty-six.",
    # 6. The strange edit
    "One kept edit set a single number in its last layer to minus thirty-one point six. That flips one channel and "
    "makes it seven times bigger. It was kept because one more practice problem came out right.",
    # 7. Random baseline
    "And the random baseline? The same loop with a hundred random edits: thirty-one percent on practice, twenty-seven "
    "and a half on the held-out set. Only three of the model's edits beat the model it already was. For random edits, five.",
    # 8. Our mistake
    "We made a mistake too. Our first run described these weights wrongly to the model. We re-ran with the right "
    "description, and the picture didn't change.",
    # 9. What it shows
    "So at our scale, self-editing worked as a search that the practice set could fool, no better than dice. The "
    "preprint's own abstract says random search is sometimes comparable, and the model shows little understanding of "
    "its edits.",
    # 10. Code
    "In code: the model writes the edit, we apply it, score it, and keep it only if the practice score goes up.",
    # 11. Next
    "Next: can it diagnose its own mistakes first, and edit the weights behind them?",
]
