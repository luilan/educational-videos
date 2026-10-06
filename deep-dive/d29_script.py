"""How LLMs Work: Deep Dive, episode 29 — State-Space Models. One entry per narration section."""
TITLE = "State-Space Models"
TAGLINE = "Replacing attention with a running memory"
NEXT = "Multimodal: Images as Tokens"
SECTIONS = [
    # 1. Hook
    "Attention compares every new token with every earlier one, and keeps a cache that grows with the text. State-"
    "space models take another route: a fixed-size memory, updated once per token.",
    # 2. Recurrence
    "The simplest version: for every channel, the new state is a times the old state, plus one minus a times the new "
    "input. A running, decaying average of the past. With a fixed a per channel, some channels remember for a long "
    "time, others only a few steps.",
    # 3. Selective
    "The key idea behind Mamba-style models is to make a depend on the token itself. Then the model can choose, at "
    "every step, what to keep and what to forget. That's called a selective state space.",
    # 4. Experiment
    "We replace attention in our tiny model with each version. No position embedding is needed: a recurrence knows the "
    "order by itself. Attention: one point six four four. Fixed decay: one point six four eight, about the same. "
    "Selective decay: one point five nine one. Better than attention, here.",
    # 5. Caveat
    "Here means texts of sixty-four characters, where nearby characters matter most. A fixed-size state must squeeze "
    "the whole past into a few numbers, and research has found that exact recall from far back is where attention "
    "keeps its edge.",
    # 6. Memory
    "The big win is at inference. Per layer, attention's cache grows with every token: about one megabyte after a "
    "thousand tokens, ninety-eight after a hundred thousand. The recurrent state stays at half a kilobyte, forever.",
    # 7. Time
    "And time. Processing one new token in one layer: attention takes thirty-eight microseconds with a thousand tokens "
    "of context, three hundred and thirteen with ten thousand, and eleven milliseconds with a hundred thousand. The "
    "recurrence: six microseconds, every time.",
    # 8. Hybrids
    "So some recent models mix the two. AI21's Jamba, for example, interleaves Mamba layers with a few attention "
    "layers: cheap memory for most of the work, exact lookup where it's needed.",
    # 9. Code
    "In code, the recurrence is one line in a loop: h equals a times h, plus one minus a, times u.",
    # 10. Outro
    "That's state-space models. Next: how a language model learns to read images.",
]
