"""The Frontier: Self-Improvement and Memory, episode 3 — The Random Baseline. One entry per narration section.

Every number comes from code/x03_the_random_baseline (random_search.py, 2026-11-28): episode 2's propose-test-keep
loop with random edits (3 seeds) and random nudges (1 seed), up to 10,000 tries per damage; Claude's numbers are
episode 2's.
"""
TITLE = "The Random Baseline"
TAGLINE = "Is it the reasoning, or just the loop?"
NEXT = "A Model That Edits Itself"
SECTIONS = [
    # 1. Hook
    "Last time, Claude repaired our broken addition model by reading its weights. But our harness only kept edits "
    "that raised the score. Maybe the loop did the work, and any guesser would do. So today we replace Claude with "
    "dice.",
    # 2. The random editors
    "The random editor uses exactly Claude's edits: set a number, scale a row, swap two rows, replace a row. But it "
    "picks everything at random: which weights, which row, which value. A second one just nudges every weight by a "
    "little noise. Same rule: keep an edit only if the practice score goes up. Up to ten thousand tries.",
    # 3. Equal tries
    "First, a fair fight. Each of Claude's successful repairs took one try. Give random search four tries, and it fixes "
    "nothing. "
    "All three damaged models stay exactly where they were.",
    # 4. The easy case
    "But random tries are cheap. Each one takes about forty-five milliseconds. For the zeroed number, random edits "
    "reached ninety-nine percent on practice in two of three runs, after seven hundred and seventy-four and fifteen "
    "hundred tries. Claude changed one number. Random search changed three hundred and thirty-one, and eighteen hundred.",
    # 5. Overfitting
    "And those patches fit the practice questions, not the task. On the hidden test, they score ninety-eight and "
    "ninety-seven. The nudges reach ninety-nine on practice, but only ninety-two on the test.",
    # 6. The hard cases
    "Now the swapped digits and the flipped sign. After ten thousand tries, random edits reach sixty to ninety-three "
    "percent, and sixty-six to sixty-nine, by piling up patches. They never find the real fix. The exact swap is about "
    "one chance in nine thousand per try. The nudges never keep a single edit.",
    # 7. Lesson
    "So the loop alone can fix small damage, if you can afford thousands of tries, and accept a messier model. "
    "Structural damage needs the right edit in a huge space, and that's where reading the weights pays. Results like "
    "the Neural Thickets paper show random search is a strong rival, so any self-improvement claim must beat it.",
    # 8. Code
    "In code, the random editor is a few lines: pick a tensor, pick an operation, pick its numbers, and hand it to the "
    "same loop.",
    # 9. Next
    "Next: a model that edits itself. The editor and the edited become the same network.",
]
