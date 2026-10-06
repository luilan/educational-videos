"""How LLMs Work: Deep Dive, episode 45 — Circuits. One entry per narration section."""
TITLE = "Circuits"
TAGLINE = "Tracing an algorithm inside GPT-2"
NEXT = None  # series finale
SECTIONS = [
    # 1. Hook
    "In this last episode, we trace a whole algorithm inside a model: which parts do what, and in what order. The "
    "classic example is called indirect object identification.",
    # 2. Task
    "When Kate and Peter went to the store, Peter gave a drink to. The answer is Kate: the name that wasn't repeated. "
    "Over ninety prompts like this, GPT-2 prefers the right name every time, with a logit difference of two point six "
    "eight.",
    # 3. Patching
    "The tool is activation patching. Make a corrupted prompt where the second name is swapped, so the answer flips: "
    "the difference becomes minus three point three. Run the corrupted prompt, copy in one piece of the clean run, and "
    "measure how much of the right answer comes back.",
    # 4. Residual
    "Patch the residual stream at each layer and position. Through layer six, everything that matters sits at the "
    "repeated name. At layers seven and eight, it moves to the last position, where the answer is written. "
    "Information flows across the sentence, at a specific depth.",
    # 5. Heads
    "Now patch single heads at the last position. A handful matter: eight point ten, eight point six, seven point nine "
    "and nine point nine recover a quarter each. One head, ten point seven, pushes the wrong way: it argues against "
    "the answer.",
    # 6. Circuit
    "The published analysis of this circuit tells a story. Early heads notice that Peter appears twice. Middle heads "
    "pass on: don't say the repeated name. Late heads, the name movers, copy the remaining name to the output.",
    # 7. Knockout
    "Knock out the top six heads, and the logit difference drops from two point six eight to one point six four; six "
    "random heads barely matter. But GPT-2 still answers right ninety-three percent of the time: backup heads take "
    "over. Real circuits are redundant.",
    # 8. Code
    "In code, patching is a hook: during the corrupted run, overwrite one activation with its value from the clean run.",
    # 9. Outro
    "That's the end of the Deep Dive. From bytes and tokens to attention, training, scaling, alignment and "
    "interpretability, every number came from code you can run. Thanks for watching.",
]
