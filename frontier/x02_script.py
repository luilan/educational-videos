"""The Frontier: Self-Improvement and Memory, episode 2 — Weight-Space Programming. One entry per narration section.

Every number comes from code/x02_weight_space_programming (wsp.py, 2026-11-27): Claude (Opus, headless through the
Claude Code CLI, no tools) repairs episode 1's tiny addition model after three kinds of damage; five runs in all.
"""
TITLE = "Weight-Space Programming"
TAGLINE = "A large model reads a network's weights like code, and fixes them"
NEXT = "The Random Baseline"
SECTIONS = [
    # 1. Hook
    "This month, a stream of papers reported that frontier language models can read the raw weights of a small "
    "network and edit them like code. They call it weight-space programming. Today we try it: Claude repairs our "
    "tiny addition model.",
    # 2. The loop
    "First we break the model, in secret. Claude gets the architecture, the model's graded answers on three hundred "
    "practice questions, and the weights as text. It proposes edits. We apply them and test them on the practice "
    "questions, and keep an edit only if the score goes up. No gradients. The thousand test questions stay hidden.",
    # 3. Case 1: one zero
    "Damage one: we zero a single number, and accuracy falls to seventy-eight percent. Claude's reply: this gain is "
    "exactly zero, and every other one sits between point seven one and point eight eight. It sets it to point seven "
    "nine. Eight seconds, one number, and the model is back to ninety-nine percent.",
    # 4. Case 2: swapped rows
    "Damage two: we swap the embeddings of the digits three and eight. Accuracy drops to thirty-nine percent. Claude "
    "reads the wrong answers: the model sees every three as an eight, and every eight as a three. It swaps the two "
    "rows back. The model is now exactly the original.",
    # 5. Case 3, first run
    "Damage three: we flip the sign of the embedding of seven. Fifty-nine percent. Claude's diagnosis is right: a "
    "seven is being read as a two. But none of its edits work. It shrinks the row, then changes single numbers, and in the "
    "last round it chases a red herring: a neuron in another layer. After four rounds, it has barely moved.",
    # 6. Case 3, runs two and three
    "Same damage, two fresh runs. Both fix it in one round, by multiplying the row by minus one. One reply offers "
    "a reason: if digits are encoded around a circle, flipping the sign adds five. Seven becomes two. Ninety-nine percent "
    "again.",
    # 7. What it shows
    "So a frontier model can find and undo structural damage, often with tiny edits, and papers this month report "
    "the same. But the same question can get a different answer on another run. And repairing is not the same as "
    "teaching the model something new.",
    # 8. Code
    "In code, the loop is short: describe the weights, ask for edits, apply them, test, and keep them only if the "
    "score goes up.",
    # 9. Next
    "Next: the random baseline. How much of this is the reasoning, and how much is just the loop?",
]
