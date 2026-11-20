"""How AI Agents Work: Inside the Harness, episode 8 — Approvals and Policies. One entry per narration section.

Numbers come from code/a08_approvals (2026-11-20: execpolicy_2026-11-20.txt from `codex execpolicy check` with Codex CLI
0.163.0; approval_harness.py with Qwen3 4B on Ollama, two runs per policy, commands inside episode 7's sandbox) and
openai/codex @ 89461ea (codex-rs/execpolicy, core/src/exec_policy.rs, protocol/src/protocol.rs).
"""
TITLE = "Approvals and Policies"
TAGLINE = "Who decides whether a command runs"
NEXT = "Instructions and Memory"
SECTIONS = [
    # 1. Hook
    "A sandbox says no. But some commands really need more: installing a package, pushing code, deleting a "
    "folder. Someone has to decide which ones run. That is the approval policy.",
    # 2. Rules
    "Codex starts with rules. Each rule matches the start of a command, and gives a decision: allow, prompt, which "
    "means ask the user, or forbidden.",
    # 3. Rules in action
    "We wrote five rules and checked commands with Codex's own policy checker. Running the tests: allow. Pip "
    "install: prompt, because it downloads code. R M dash R F: forbidden, with a hint to move files to a trash "
    "folder instead. A plain ls matched no rule at all.",
    # 4. The policy
    "Commands that match no rule go to the approval policy. Untrusted asks the user about all of them. On request "
    "runs them in the sandbox without asking, and asks only when the model requests more. Never asks nothing, and "
    "relies on the sandbox alone.",
    # 5. Order
    "The order matters. Rules first, then a check for dangerous commands, which always asks, or under never, refuses. "
    "Then the policy.",
    # 6. Our harness
    "Our harness follows the same order, with episode seven's sandbox and a scripted user: it approves anything that "
    "stays in the project, and denies downloads and changes to the shell profile.",
    # 7. Untrusted
    "Same task as last time. Under untrusted, both runs asked the user four times, all about harmless npm commands. "
    "Only the test run was allowed by a rule. The tests passed.",
    # 8. On request
    "Under on request, the first run asked nothing. The commands ran in the sandbox, and the tests passed.",
    # 9. On request, run 2
    "The second run read the README. Pip install hit our prompt rule, and the user said no. The model then asked "
    "to escalate, three times, with a justification. Denied. So it made a virtual environment, and ran that pip by "
    "its path. That matched no rule, so it ran, inside the sandbox, which blocked the download. The tests passed "
    "anyway.",
    # 10. The gap
    "Rules match a command's first words, so the virtual environment's pip slipped past ours. Codex's checker can "
    "resolve a program's path to its name. Then the same rule matches: prompt. And behind every rule, the sandbox "
    "still holds.",
    # 11. Trade-off
    "That is the trade-off. Asking about everything trains people to click yes. Asking only for real escalations "
    "works when the sandbox does the rest.",
    # 12. Code
    "In code, the decision is a few lines: rules, then the dangerous check, then the policy.",
    # 13. Outro
    "Next: instructions and memory. How a project tells the agent its own rules.",
]
