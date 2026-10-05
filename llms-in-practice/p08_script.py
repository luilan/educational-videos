"""LLMs in Practice, episode 8 — Agents: Think, Act, Observe. One entry per narration section."""
TITLE = "Agents"
TAGLINE = "Think, act, observe, repeat"
NEXT = "Fine-Tuning vs Prompting, and LoRA in One Picture"
SECTIONS = [
    # 1. Hook
    "Last time: one question, one tool call. But real tasks take several steps, and each step depends on the last. "
    "A model that keeps choosing tools, in a loop, until a goal is reached, is called an agent.",
    # 2. The loop
    "The loop has three parts. Think: the model decides the next action. Act: the program runs the tool. "
    "Observe: the result goes back into the context. Then repeat, until the model gives a final answer, "
    "or we hit a step limit.",
    # 3. The task
    "Our task: I want to buy gluten-free bread today. Where can I get it, and how long is the walk? "
    "The agent has three made-up tools: get today's date, find shops, and get the walking time. "
    "And it must use them in order: first the day, then the shops, then the walk.",
    # 4. First try
    "First try, with a three billion parameter model. It calls find shops with the day set to: today. "
    "Not a weekday name, so the tool finds nothing. Then it asks for the walk to: shop for gluten-free bread. "
    "Its final answer: no shop sells it today. Wrong. Pane Vivo does, on Wednesdays. Confident, and wrong.",
    # 5. Smaller model
    "An even smaller model did worse. After getting the day, it simply invented two shops, with addresses.",
    # 6. The fix
    "The fix isn't in the model. It's in the tools. Check every input, and when it's wrong, say why, "
    "and what to do instead. Day must be a weekday name: call get today first. Unknown place: use a shop name "
    "from find shops.",
    # 7. Second try
    "Second try, same model. Step one: the same two mistakes, but now both come back as clear errors. "
    "Step two: it gets the day, Wednesday, but repeats a mistake. Step three: it checks the day again. "
    "Step four: Wednesday, and the shop list says Pane Vivo. Step five: eighteen minutes. "
    "Step six: the right answer. Six steps, three errors, each one fixed by reading an error message.",
    # 8. Cost
    "Notice the context. Every step resends everything so far: four hundred and twenty-two tokens at the start, "
    "eight hundred and fourteen at the end. Agents are slow and costly, and long runs can overflow the window.",
    # 9. Code
    "In code, it's the tool loop with a limit. For each step: generate. No tool call? That's the final answer. "
    "Otherwise, run every call, append the results, and go around again.",
    # 10. Lessons
    "So, to build agents that work: set a step limit. Make tools check their inputs, with helpful errors. "
    "Keep the toolset small and clear. Log every step. Ask a human before any action with consequences. "
    "And use a model that's capable enough for the job.",
    # 11. Outro
    "Next up: when prompting isn't enough. Fine-tuning, and Lora, in one picture.",
]
