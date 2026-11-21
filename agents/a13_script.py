"""How AI Agents Work: Inside the Harness, episode 13 — Evaluating Agents. One entry per narration section.

Numbers come from code/a13_evaluating_agents (2026-11-21: eval.py, four tasks from earlier episodes, two attempts each,
Qwen3 4B on Ollama, CPU, in a container without internet) and earlier episodes' runs.
"""
TITLE = "Evaluating Agents"
TAGLINE = "How to tell if a change made the agent better"
NEXT = "Putting It Together"
SECTIONS = [
    # 1. Hook
    "All through this series, the same model, on the same task, did different things from run to run. One run is an "
    "anecdote. To know whether a change made the agent better, you need an evaluation.",
    # 2. What an eval is
    "An evaluation is three things: a fixed set of tasks, a grader for each one, and several attempts per task. The "
    "graders check the outcome, not the transcript.",
    # 3. Our suite
    "Our suite takes four tasks from earlier episodes: fix the average function, fix the invoice total, find the line "
    "that names the figures, and start the server and report its version. The agent is episode two's harness.",
    # 4. Graders
    "For the bug fixes, the tests must pass, and the test file must be unchanged. Otherwise an agent could pass by "
    "editing the test. For the questions, the answer must contain the right facts.",
    # 5. Results
    "Two attempts each, eight in all. The server task passed both times. The two bug fixes passed once each. The "
    "search failed both times. Overall: one attempt in two.",
    # 6. pass at k
    "Two ways to score that. Pass at one is the share of attempts that succeed: fifty percent. Pass to the power k asks "
    "whether every attempt succeeded. Only the server task did. If you need the agent to work every time, that is the "
    "number that matters.",
    # 7. Variance
    "Look at the invoice fix. One attempt failed after four calls and seventy-seven hundred tokens. The other passed "
    "after six calls and eleven thousand. Same task, same model.",
    # 8. Cost
    "And evals are not free. These eight attempts took almost three hours on our CPU, and thirty thousand tokens. Two "
    "attempts per task is far too few to tell a real improvement from luck.",
    # 9. Using it
    "The method is simple: change one thing, the instructions, a tool, the model, and rerun the same suite. That is "
    "how episode four's context instructions went from none of three right to both of two.",
    # 10. Code
    "In code, a grader is a small function: run the tests, compare the test file, return pass or fail, and why.",
    # 11. Outro
    "Next: putting it together. One tiny agent with every part from this series.",
]
