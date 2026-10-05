"""How LLMs Work: Deep Dive, episode 25 — Data Parallelism. One entry per narration section."""
TITLE = "Data Parallelism"
TAGLINE = "Many workers, one model, the same step"
NEXT = "Tensor and Pipeline Parallelism"
SECTIONS = [
    # 1. Hook
    "Large models are trained on thousands of GPUs at once. Part six of the deep dive is about how. The simplest way "
    "to use many workers is data parallelism.",
    # 2. Idea
    "Every worker holds a full copy of the model, and takes its own slice of the batch. Each one computes gradients on "
    "its slice. Then the workers average their gradients, an operation called all-reduce, so every copy takes exactly "
    "the same step, and they all stay identical.",
    # 3. Real run
    "Let's run it for real, on this machine: four worker processes, talking through PyTorch's distributed library. "
    "Each takes eight sequences. We compare them with a single process that takes all thirty-two.",
    # 4. Result
    "The first gradient matches to within two hundred-millionths. And over two hundred training steps, the losses never "
    "differ by more than five ten-millionths: four point three three three five at the start, two point three zero "
    "zero six at the end, in both runs.",
    # 5. Why exact
    "Why exact? The batch loss is an average. The average of four averages over equal slices is the average over all "
    "thirty-two. Splitting the work changes nothing in the math.",
    # 6. Cost
    "The cost is communication. Every step, every worker must exchange all its gradients. For our tiny model, that's "
    "three megabytes. For GPT-2: half a gigabyte, every step. For a seven billion parameter model: twenty-six "
    "gigabytes.",
    # 7. Ring
    "The usual way is a ring all-reduce: workers pass chunks to their neighbor around a ring, and each one sends about "
    "twice the gradient size, however many workers there are. Libraries also start sending early, while the backward "
    "pass is still running, to hide the wait.",
    # 8. Limit
    "The catch: every worker needs the whole model. With Adam, in thirty-two bits, that's sixteen bytes per parameter: "
    "the weights, the gradients, and Adam's two averages. For seven billion parameters, a hundred and twelve gigabytes, "
    "more than one GPU holds. The next episodes split the model itself.",
    # 9. Code
    "In code, it's two lines: start the process group, and wrap the model in distributed data parallel. The training "
    "loop doesn't change.",
    # 10. Outro
    "That's data parallelism. Next: splitting a single model across workers, with tensor and pipeline parallelism.",
]
