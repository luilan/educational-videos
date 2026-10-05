"""LLMs in Practice, episode 9 — Fine-Tuning vs Prompting, and LoRA in One Picture. One entry per narration section."""
TITLE = "Fine-Tuning and LoRA"
TAGLINE = "When prompting isn't enough"
NEXT = "Quantization: Shrinking a Model to Fit a Laptop"
SECTIONS = [
    # 1. Hook
    "So far we've changed what goes into the model: prompts, documents, tool results. The model itself never changed. "
    "But sometimes you want it to behave differently every time, without saying so in every prompt. "
    "That's fine-tuning.",
    # 2. Prompting vs fine-tuning
    "Prompting is cheap and instant, and it's the first thing to try. RAG adds knowledge. "
    "Fine-tuning changes the weights: it's for a style, a format, or a skill that's hard to describe in words. "
    "It costs training time, and you need examples.",
    # 3. The experiment
    "Let's try it on the tiny GPT from How LLMs Work. It was trained on Shakespeare. Ask it for a recipe, "
    "and it writes Shakespeare. On our made-up recipes, its loss is two point seven seven.",
    # 4. Full fine-tuning
    "Option one: full fine-tuning. Keep training on six hundred steps of recipes, updating all eight hundred "
    "thousand weights. Recipe loss drops to zero point two five, and it writes a perfect recipe. "
    "But its Shakespeare loss jumps from one point six to four point four. It forgot. "
    "And you now have a whole new copy of the model.",
    # 5. LoRA in one picture
    "Option two: Lora, low-rank adaptation. Freeze every original weight. Next to each weight matrix, "
    "add two thin matrices: matrix A, and matrix B. Their product is a small correction, added to the frozen output. "
    "Matrix B starts at zero, so at first nothing changes. Only matrix A and matrix B are trained.",
    # 6. The numbers
    "With rank four, that's thirty-two thousand, seven hundred and sixty-eight trainable numbers: "
    "four percent of the model. The adapter is a one hundred and twenty-eight kilobyte file. "
    "And the result? Recipe loss: zero point two seven. Almost the same as full fine-tuning.",
    # 7. Switch it off
    "Here's the best part. Switch the adapter off, and the Shakespeare loss is back to exactly one point six. "
    "The original weights never changed. Keep one base model, and swap small adapters: one for recipes, "
    "one for legal letters, one per customer.",
    # 8. Code
    "In code, Lora is a few lines. Wrap a linear layer. Freeze it. Add matrix A, random, and matrix B, zero. "
    "The output is the frozen layer, plus x times matrix A, times matrix B, times a scale.",
    # 9. When to use what
    "So: start with prompting. Add RAG for knowledge, especially facts that change. "
    "Fine-tune, with Lora, for style, format, and narrow skills, when you have hundreds of good examples.",
    # 10. Outro
    "Next up: making models smaller, so they run on a laptop. Quantization.",
]
