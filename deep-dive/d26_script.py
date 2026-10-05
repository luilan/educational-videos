"""How LLMs Work: Deep Dive, episode 26 — Tensor and Pipeline Parallelism. One entry per narration section."""
TITLE = "Tensor and Pipeline Parallelism"
TAGLINE = "Splitting one model across workers"
NEXT = "Sharding the Optimizer State"
SECTIONS = [
    # 1. Hook
    "Last episode, a seven billion parameter model needed a hundred and twelve gigabytes on every worker. Too much. "
    "So we split the model itself. There are two classic ways.",
    # 2. Tensor parallelism
    "The first is tensor parallelism: split each weight matrix. Take an M L P going from a hundred and twenty-eight "
    "numbers to five hundred and twelve, and back. Worker zero keeps the first half of the columns of the first "
    "matrix, so it computes half the hidden units. The GELU works on each unit separately, so that's fine. Then it "
    "keeps the matching half of the rows of the second matrix, and gets a partial output.",
    # 3. Check
    "Worker one does the same with the other halves. Add the two partial outputs, with an all-reduce, and you get "
    "exactly the full M L P's output. In a real run with two processes: a difference of one millionth, just rounding. "
    "Each worker holds half the parameters.",
    # 4. Cost
    "The price: an all-reduce of the layer's output in every layer, and attention is split the same way, by heads. "
    "That's a lot of traffic, every step, so tensor parallelism usually stays inside one machine, where the links "
    "between GPUs are fastest.",
    # 5. Pipeline
    "The second way is pipeline parallelism: split the layers. Worker zero runs layers zero and one, then sends its "
    "activations to worker one, which runs layers two and three. Only one tensor crosses between them. The output "
    "matches the full model exactly.",
    # 6. Bubble
    "But a pipeline has a bubble. With one batch, worker one waits while worker zero works, then worker zero waits: "
    "half the time idle. Split the batch into micro-batches, and both stay busy. Two stages, four micro-batches: twenty "
    "percent idle. Sixteen: six percent. With eight stages, you need many more micro-batches to shrink the bubble.",
    # 7. Combine
    "Real training runs combine all three: tensor parallelism inside a machine, pipeline parallelism across machines, "
    "and data parallelism on top, over many copies of the whole arrangement.",
    # 8. Code
    "In code, tensor parallelism is a slice of each matrix and one all-reduce. Pipeline parallelism is a send and a "
    "receive between stages.",
    # 9. Outro
    "That's splitting the model. Next: splitting the optimizer state, so data parallel workers stop storing copies of "
    "the same numbers.",
]
