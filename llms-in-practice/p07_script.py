"""LLMs in Practice, episode 7 — Tool Use: How a Model Calls a Function. One entry per narration section."""
TITLE = "Tool Use"
TAGLINE = "How a model calls a function"
NEXT = "Agents: Think, Act, Observe"
SECTIONS = [
    # 1. Hook
    "Ask a model about today's weather. It can't know. It has no window, no internet, no clock: "
    "just text in, and text out. Yet assistants check the weather, search the web, and run code. How?",
    # 2. The trick
    "The model never runs anything itself. It writes a request, in a format it was trained on, "
    "and your program does the work.",
    # 3. Describe the tools
    "Step one: describe the tools. A name, what it does, and its parameters, written as JSON. "
    "The chat template pastes this into the system prompt. Here's the real one: you may call functions, "
    "here are their signatures, and here's how to write a call. It's just more text in the context.",
    # 4. The model writes a call
    "Step two: we ask, what's the weather like in Milan right now? Do I need an umbrella? "
    "And the reply isn't an answer. It's a request: tool call, get weather, with the city set to Milan.",
    # 5. The app runs it
    "Step three: our program spots the tool call, reads the JSON, and runs the real function. "
    "Here, a made-up weather service: eighteen degrees, and light rain. "
    "This is the only moment anything happens in the world.",
    # 6. Back into the context
    "Step four: the result goes back into the context, as a tool response, and the model continues. "
    "Now it reads two hundred and sixty-four tokens, including the weather, and answers: "
    "eighteen degrees and light rain, so you might want an umbrella.",
    # 7. Failure
    "But ask more vaguely: do I need an umbrella in Milan right now? The same small model didn't call the tool at all. "
    "It asked for your location, even though you'd said Milan. Larger models call tools more reliably, "
    "and clear descriptions help. Your code must handle no call, and broken JSON.",
    # 8. Code
    "In code, it's a loop. Generate. If there's no tool call, you're done. Otherwise, parse it, "
    "run the function, append the result, and generate again.",
    # 9. Safety
    "The model chooses what to call, but you choose what it can do. Reading the weather is harmless. "
    "Sending an email or making a payment is not: ask the user to confirm, check the arguments, "
    "and never run code from the model without limits.",
    # 10. Outro
    "Give a model several tools and a goal, and run this loop until the job is done. That's an agent. "
    "Next up.",
]
