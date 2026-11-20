"""How AI Agents Work: Inside the Harness, episode 6 — Running Commands. One entry per narration section.

Numbers come from code/a06_running_commands (2026-11-20: run_harness.py, Qwen3 4B on Ollama, CPU, in containers
without internet) and openai/codex @ 89461ea (core/src/tools/handlers/unified_exec.rs, core/src/unified_exec/,
core/src/tools/context.rs).
"""
TITLE = "Running Commands"
TAGLINE = "What a harness does with a command that never ends"
NEXT = "Sandboxing"
SECTIONS = [
    # 1. Hook
    "Some commands never end on their own: a web server, a file watcher, a program waiting for input. Our harness "
    "so far waits for every command to finish, and kills it after thirty seconds.",
    # 2. The task
    "The task: start a small web service, which keeps running until stopped, then check it with a second script, "
    "and report the version it returns.",
    # 3. Blocking, run 1
    "In two runs, the model started the server in the background, with an ampersand. The harness still waited the "
    "full thirty seconds: it captures the command's output, and the server was holding that output open. Then the "
    "timeout killed the shell, but not the server. So the check worked: the right answer, by accident, from a "
    "server nobody was tracking.",
    # 4. Blocking, run 2
    "In another, the model never started the server. It ran the check, got connection refused, and told the user "
    "to start the server themselves.",
    # 5. Sessions
    "Codex runs commands differently. Exec command waits a short while, ten seconds by default, then returns "
    "whatever output there is. If the process is still running, the result says so, with a session ID.",
    # 6. Write stdin
    "A second tool, write standard in, takes that ID. It can type into the process, or send nothing, just to read the new "
    "output since last time.",
    # 7. Sessions run
    "With these two tools, the model started the server. After ten seconds, the result said: process running, "
    "session one. It ran the check, and answered version one point four point two. Two model calls. In the other "
    "sessions run, the model's first reply was empty, and the run ended there.",
    # 8. Limits
    "A harness has to keep these processes in check. Codex keeps at most sixty-four, closing old ones when it needs "
    "room, and every result still has an output cap. And when the task ends, someone has to stop the server.",
    # 9. Code
    "In code, a session is a process, plus a thread that collects its output. Each call waits a little, then hands "
    "over what came in.",
    # 10. Outro
    "Next: sandboxing. What a command is allowed to touch.",
]
