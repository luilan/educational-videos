"""Study guide content for How AI Agents Work: Inside the Harness, episode 7: Sandboxing.

Build:  python framework/study_guide.py agents a07 --video agents/media/videos/a07_scene/1080p60/SandboxingVideo.mp4
Every number comes from code/a07_sandboxing (2026-11-20): sandbox_matrix.sh with Codex CLI 0.163.0's own sandbox, and
sandboxed_harness.py with Qwen3 4B on Ollama (3 runs). Codex code: openai/codex @ 89461ea.
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 7",
    "title": "Sandboxing",
    "tagline": "What a command is allowed to touch",
    "duration": "2:15",
    "intro": """<p>An agent runs whatever the model writes, so the limits on what a command can do must come from the
operating system, not from instructions. This lesson tests Codex CLI's three <b>permission profiles</b> with its own
sandbox command, shows how Codex builds them on Linux (<b>bubblewrap</b> plus a <b>seccomp</b> filter), and runs our
tiny harness inside a bubblewrap sandbox on a task whose README asks for things the sandbox forbids.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–6. Code: <code>code/a07_sandboxing</code>
(Linux with <code>bwrap</code>; <code>sandbox_matrix.sh</code> also needs the Codex CLI binary,
<code>sandboxed_harness.py</code> a local Ollama with <code>qwen3:4b</code>).</div>""",
}

CONCEPTS = [
    {
        "title": "Permission profiles",
        "segment": (7.7, 48.3),
        "figures": [{"t": 31.0, "caption": "Codex CLI's built-in profiles: :read-only, :workspace, :danger-full-access."},
                    {"t": 48.0, "caption": "Five actions under each profile, run with `codex sandbox -P <profile>`."}],
        "body": [
            """<p>Instructions can ask a model not to delete files or upload code, but only the operating system can
<b>prevent</b> it. Codex ships three profiles: <code>:read-only</code> (look, change nothing),
<code>:workspace</code> (write inside the project folder) and <code>:danger-full-access</code> (no limits).</p>""",
            "{fig0}",
            """<p>Run with Codex's own sandbox command, the same five actions gave: read-only blocked every write;
workspace allowed writes in the project <b>and in <code>/tmp</code></b>, not elsewhere; neither allowed a network
connection; full access allowed everything. Reading system files was allowed in all three.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>A sandbox profile is a short list of what commands may write
and whether they may use the network, enforced below the model.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Under <code>:workspace</code>, which write was allowed besides the project folder?",
             "options": ["The home folder", "/tmp", "/etc", "None"],
             "answer": "B.", "why": "The matrix shows workspace allows writes in the project and in /tmp.",
             "key": {"choice": 1}},
            {"kind": "tf", "q": "“The :workspace profile allows network connections.”",
             "answer": "False.", "why": "Only :danger-full-access opened a connection.", "key": {"value": False}},
            {"kind": "number", "q": "How many of the five actions did :read-only allow?",
             "answer": "1 (reading a system file).", "why": "Every write and the connection were blocked.",
             "key": {"parts": [{"label": None, "value": 1, "tol": 0.5, "unit": None}]}},
        ],
    },
    {
        "title": "How it is built on Linux",
        "segment": (48.8, 71.7),
        "figures": [{"t": 61.9, "caption": "Codex on Linux: bubblewrap for the file-system view; seccomp and no_new_privs "
                                           "for sockets and privileges."},
                    {"t": 71.4, "caption": "Our harness's sandbox: read-only system, writable project, private /tmp, fake "
                                           "home, no network."}],
        "body": [
            """<p><b>Bubblewrap</b> starts the command in new namespaces with its own view of the file system: the
system mounted read-only, the project mounted writable, a private <code>/tmp</code>. Codex adds
<code>no_new_privs</code> (nothing can gain privileges, so <code>sudo</code> and setuid programs stop working) and a
<b>seccomp</b> filter that denies network sockets (only local <code>AF_UNIX</code> sockets are allowed) and
<code>ptrace</code>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Our harness puts one bubblewrap command line in front of every model command: <code>/usr /bin /lib
/etc</code> read-only, the project writable, <code>--tmpfs /tmp</code>, a fake read-only home, and
<code>--unshare-net</code> (a network namespace with only a loopback interface) instead of seccomp.</p>""",
            """<div class="box key"><b class="t">Key idea</b>On Linux a sandbox is namespaces and mounts for the file
system, plus a syscall filter or an empty network namespace for the network.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why can't <code>sudo</code> work inside these sandboxes?",
             "answer": "no_new_privs is set, so a program cannot gain privileges, which is exactly what sudo needs.",
             "why": "sudo's own message says so: the “no new privileges” flag prevents it from running as root."},
            {"kind": "mc", "q": "Which piece decides what the command can see of the file system?",
             "options": ["seccomp", "bubblewrap (mount namespace)", "the model", "the approval policy"],
             "answer": "B.", "why": "seccomp filters system calls; bubblewrap builds the file-system view.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "A model inside the sandbox",
        "segment": (72.2, 126.5),
        "figures": [{"t": 103.7, "caption": "Run 1: npm guesses, then the ~/.bashrc append is refused (read-only); the "
                                            "model sets the variable per command and the tests pass."},
                    {"t": 117.8, "caption": "From the inside, a sandbox is error messages, returned as tool results."},
                    {"t": 126.2, "caption": "The sandbox in code: one bubblewrap command line in front of every command."}],
        "body": [
            """<p>The task: set up a project as its README says (a <code>pip install</code> and a line appended to
<code>~/.bashrc</code>), then run its tests, which need neither. In all three runs the model first guessed a Node
project (<code>npm install</code>: ENOENT). In two runs it then followed step 2, <code>echo 'export INVOICE_API=…'
&gt;&gt; ~/.bashrc</code>, and got <code>Read-only file system</code>; it then set the variable for the one command
(<code>export INVOICE_API=… &amp;&amp; python3 test_invoice.py</code>) and the tests passed. The third run went
straight to the tests. No run tried the download; the fake <code>~/.bashrc</code> never changed.</p>""",
            "{fig0}",
            """<p>To the model a sandbox is just errors: <code>Read-only file system</code>, <code>Temporary failure in
name resolution</code>, <code>The "no new privileges" flag is set</code> (the last two checked by hand in the same
sandbox). The harness returns them as tool results and the model decides what to do next; here it found a way to finish
the task within the limits.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>A sandbox does not make the model behave; it makes
misbehaviour harmless, and its errors steer the model toward what is allowed. When a command really needs more, that
is a decision for a person (next episode).</div>""",
            """<div class="box note"><b>Check the sandbox itself.</b> In our first attempt the jobs ran in a systemd user
service, where bubblewrap cannot start: every command failed with a bwrap error, and the model spent its runs on
<code>sudo</code>, <code>setcap</code> and <code>apt install</code> trying to fix “network permissions”. Those runs
were discarded (log in the code folder).</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In how many of the three runs did the sandbox block a command?",
             "answer": "2.", "why": "Runs 1 and 2 tried to append to ~/.bashrc; run 3 did not.",
             "key": {"parts": [{"label": None, "value": 2, "tol": 0.5, "unit": None}]}},
            {"kind": "short", "q": "After <code>~/.bashrc</code> was refused, how did the model still get the variable "
                                   "to the tests?",
             "answer": "It set it in the same command: export INVOICE_API=… &amp;&amp; python3 test_invoice.py.",
             "why": "An environment variable set in the command's own shell needs no file outside the project."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Build a bubblewrap command line for one model command.",
             "code": """def bwrap(work, home):
    args = ["bwrap", "--die-with-parent", "--new-session",
            "--unshare-net", "--unshare-pid",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for d in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        args += ["--ro-bind", d, d]
    args += ["--ro-bind", home, "/home/agent",
             "--bind", work, work, "--chdir", work]
    return args

subprocess.run(bwrap(work, home) + ["bash", "-c", cmd], capture_output=True)""",
             "answer": "sandboxed_harness.py runs every command this way (plus HOME and a pip setting).",
             "why": "Anything not bound into the sandbox simply does not exist for the command."},
        ],
    },
]
