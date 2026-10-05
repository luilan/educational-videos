"""How LLMs Work: Deep Dive, episode 27 — Sharding the Optimizer State. One entry per narration section."""
TITLE = "Sharding the Optimizer State"
TAGLINE = "Stop storing the same numbers on every worker"
NEXT = "Mixture of Experts"
SECTIONS = [
    # 1. Hook
    "Data parallelism has a wasteful habit: every worker stores the same Adam state, the same two numbers for every "
    "parameter, copied on every machine. Sharding fixes that: each worker keeps only its own slice.",
    # 2. How
    "After the gradients are averaged, each worker updates only its slice of the weights, using only its slice of Adam's "
    "state. Then the workers share the updated slices, an operation called all-gather, so everyone ends the step with "
    "the full, identical model.",
    # 3. Real run
    "Let's run it: four processes again, once with a plain optimizer, and once with PyTorch's zero redundancy optimizer. "
    "After a hundred steps, the losses are identical, to every digit: two point four four six seven in both runs.",
    # 4. Memory
    "But the memory is not. With the plain optimizer, each worker holds six point two four megabytes of Adam state. "
    "Sharded: one point five six. Exactly a quarter, with four workers.",
    # 5. Scale
    "At scale, this is what makes training possible. A seven billion parameter model on sixty-four workers: everything "
    "copied needs a hundred and twelve gigabytes per worker. Shard Adam's state, and it's fifty-seven. Shard the "
    "weights and gradients too, and each worker needs less than two gigabytes.",
    # 6. Cost
    "Nothing is free. Sharded weights must be gathered just before each layer uses them, and released afterwards. That's "
    "more communication, traded for memory.",
    # 7. Names
    "You'll meet these ideas as ZeRO, from Microsoft's DeepSpeed, and as fully sharded data parallel, or FSDP, in "
    "PyTorch. Same principle: no worker stores what another already holds.",
    # 8. Code
    "In code, the optimizer sharding is one line: wrap AdamW in the zero redundancy optimizer. The training loop doesn't "
    "change.",
    # 9. Outro
    "That completes training at scale. Next, part seven: architecture variants, starting with mixture of experts.",
]
