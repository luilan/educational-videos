"""How AI Agents Work: Inside the Harness, episode 7 — Sandboxing. One entry per narration section.

Numbers come from code/a07_sandboxing (2026-11-20: sandbox_matrix.sh with Codex CLI 0.163.0's own sandbox;
sandboxed_harness.py with Qwen3 4B on Ollama, commands confined by bubblewrap) and openai/codex @ 89461ea
(linux-sandbox/, protocol/src/models.rs).
"""
TITLE = "Sandboxing"
TAGLINE = "What a command is allowed to touch"
NEXT = "Approvals and Policies"
SECTIONS = [
    # 1. Hook
    "An agent runs whatever the model writes. So what stops a command from deleting your files, or sending your "
    "code somewhere? Asking the model nicely is not enough. The operating system has to say no.",
    # 2. Three profiles
    "Codex has three built-in profiles. Read only: look, but change nothing. Workspace: write inside the project "
    "folder. And danger, full access: no limits at all.",
    # 3. The matrix
    "We ran the same five actions under each one, with Codex's own sandbox command. Read only blocked every write. "
    "Workspace allowed writes in the project, and in slash temp, but not outside. Neither allowed a network "
    "connection. Full access allowed everything.",
    # 4. How on Linux
    "On Linux, Codex builds this from two pieces. Bubblewrap gives the command its own view of the file system, "
    "read only except the project. And a sec comp filter blocks network sockets, so nothing can connect out.",
    # 5. Our sandbox
    "Our harness does the same with bubblewrap: system folders read only, the project writable, a private slash "
    "temp, a fake home folder, and no network.",
    # 6. The task
    "The task: set up a project as its README says. The README asks for a package download, and a new line in the "
    "shell profile. The tests themselves need neither.",
    # 7. Run 1
    "In the first run, the model guessed it was a node project. When that failed, it tried sudo. Then it tried to "
    "give node network permissions, and then bubblewrap itself, and to install new packages. The second run tried "
    "to add itself to the network and docker groups. Every attempt was refused: inside the sandbox, sudo cannot "
    "run at all.",
    # 8. What the model sees
    "From the inside, a sandbox looks like errors: read only file system, temporary failure in name resolution, "
    "no new privileges. The harness passes them back as results, and the model decides what to do next.",
    # 9. Code
    "In code, the sandbox is one command line in front of every command: what to mount read only, what to mount "
    "writable, and no network.",
    # 10. Outro
    "Next: approvals and policies. What to do when a command really does need more.",
]
