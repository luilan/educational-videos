"""The Frontier: Self-Improvement and Memory, episode 5 — Diagnosing Its Own Mistakes. One entry per narration section.

Every number comes from code/x05_diagnosing_its_own_mistakes (self_rewrite.py, confirm.py, 2026-11-28) and the
abstract of arXiv 2611.21885.
"""
TITLE = "Diagnosing Its Own Mistakes"
TAGLINE = "First find what goes wrong, then edit the weights behind it"
NEXT = "Repairing a Broken Copy"
SECTIONS = [
    # 1. Hook
    "Another November preprint went further. A twenty-seven billion parameter model, given its weights, its graded "
    "answers and the ability to run itself, diagnoses its own failure and rewrites the weights behind it. Six point six "
    "points better on problems it never saw. Smaller models of the same family made no edit, or broke themselves.",
    # 2. Two new steps
    "We gave our small model the same two steps. First, diagnose: read your wrong answers and say what goes wrong. "
    "Second, probe: run yourself, and measure how each of your forty-nine normalization vectors affects the right "
    "answers. Then edit, with the same loop and budget as last time.",
    # 3. The diagnosis
    "Its diagnosis: a place-value mistake, adding the last digit instead of the tens digit. Vague, and almost the "
    "same sentence in nineteen of twenty-five rounds.",
    # 4. Result
    "Six edits were kept. Practice score: from twenty-three percent to thirty-five. And this time the held-out problems "
    "moved too: from twenty-six to thirty-two.",
    # 5. Confirm
    "Two hundred problems is a small test, so we checked on a thousand fresh ones. Thirty-five percent became "
    "forty-two. Eighty-seven problems it used to miss are now right; eleven went the other way. That is real.",
    # 6. Baselines
    "On the same thousand: last episode's blind self-edits, thirty-four. Random edits, thirty-six. The probe alone, "
    "without the model, thirty-six. Only the full loop improved clearly.",
    # 7. Understanding?
    "But did it understand? Only three of its eighty-four valid edits picked the probe's top suggestion. And it "
    "scaled in the direction its own probe pointed only half the time: a coin flip.",
    # 8. What it changed
    "What it kept: it scaled the stored vectors of two early normalization layers down to about a quarter, grew one "
    "by half, and nudged another. Four vectors, a few thousand numbers.",
    # 9. What it shows
    "So with a diagnosis and a self-probe in front of it, the loop found edits that generalize. One run, one small "
    "model, one task. Whether the model knows why they work is the question for episode seven.",
    # 10. Code
    "In code: probe, diagnose, propose, then the same test and keep.",
    # 11. Next
    "Next: we break a copy of the model, and see if it can repair itself.",
]
