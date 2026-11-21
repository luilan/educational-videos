"""How AI Agents Work: Inside the Harness, episode 12 — Sub-agents and Parallel Tool Calls. One entry per narration section.

Numbers come from code/a12_subagents_and_parallel (2026-11-21: parallel_harness.py, Qwen3 4B on Ollama, CPU, in
containers without internet) and openai/codex @ 89461ea (core/src/tools/parallel.rs, core/src/tools/handlers/
multi_agents_spec.rs, core/src/config/mod.rs).
"""
TITLE = "Sub-agents and Parallel Tool Calls"
TAGLINE = "How one agent splits its work"
NEXT = "Evaluating Agents"
SECTIONS = [
    # 1. Hook
    "Two ways for an agent to do more at once. Several tool calls in one reply, run at the same time. And sub-agents: "
    "a fresh agent, with its own conversation, for one piece of the job.",
    # 2. Parallel calls
    "A model's reply can hold more than one tool call. The harness can run them one after another, or all at once.",
    # 3. The task
    "The task: run three independent checks, lint, types, and tests. Each takes three seconds. In all four runs, the "
    "model asked for all three in a single reply.",
    # 4. Results
    "Run one after another, the tools took nine seconds. Run at the same time, three. Same answer, same tokens.",
    # 5. Codex
    "Codex marks each tool as safe to run in parallel or not. Safe ones share a read lock and run together. The others "
    "take a write lock, and wait their turn. Running a command counts as safe.",
    # 6. Sub-agents
    "Now sub-agents. The parent gets a spawn agent tool. It writes a task for the child. The child starts with a fresh "
    "conversation and its own tools, and only its final answer comes back.",
    # 7. Context
    "On our search task from episode four, a single agent sent about eleven hundred tokens. With a sub-agent, the "
    "parent sent six hundred or so. The child's searches, up to twenty-two hundred tokens, never entered the "
    "parent's conversation.",
    # 8. Same answer
    "But all three runs gave the same answer: line two twenty-four, the near miss from episode four. A sub-agent keeps "
    "the parent's context small. It doesn't make the search smarter. And the parent's one-line task left out part of "
    "the question.",
    # 9. Codex agents
    "Codex has tools to spawn an agent, send it input, wait for it, and close it. By default, at most six agent "
    "threads, and only one level deep: a sub-agent cannot spawn its own.",
    # 10. Code
    "In code, parallel calls are a thread pool over the reply's calls. A sub-agent is the same agent loop, called with "
    "a new conversation.",
    # 11. Outro
    "Next: evaluating agents. How to tell if a change made the agent better.",
]
