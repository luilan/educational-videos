"""The bubblewrap sandbox from episode 7 (sandboxed_harness.py)."""
from pathlib import Path


def bwrap(work, home):
    """The sandbox, as a bubblewrap command line (Codex: codex-rs/linux-sandbox, plus seccomp for sockets)."""
    args = ["bwrap", "--die-with-parent", "--new-session", "--unshare-net", "--unshare-pid",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for d in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(d).exists():
            args += ["--ro-bind", d, d]
    args += ["--ro-bind", str(home), "/home/agent", "--setenv", "HOME", "/home/agent",
             "--setenv", "PIP_BREAK_SYSTEM_PACKAGES", "1",      # let pip get as far as the network, as in a virtualenv
             "--bind", str(work), str(work), "--chdir", str(work)]
    return args
