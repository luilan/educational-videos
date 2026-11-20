"""Study guide content for How AI Agents Work: Inside the Harness, episode 6: Running Commands.

Build:  python framework/study_guide.py agents a06 --video agents/media/videos/a06_scene/1080p60/RunningCommandsVideo.mp4
Every number comes from code/a06_running_commands (2026-11-20): run_harness.py with Qwen3 4B on Ollama (3 blocking runs,
2 sessions runs). Codex code: openai/codex @ 89461ea.
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 6",
    "title": "Running Commands",
    "tagline": "What a harness does with a command that never ends",
    "duration": "2:06",
    "intro": """<p>Servers, watchers and prompts waiting for input never finish on their own. A harness that waits for
every command to end, with a timeout, gets stuck on them, or loses track of them. This lesson runs one task (start a
web service, query it, report its version) through a <b>blocking</b> harness and through Codex-style <b>sessions</b>:
<code>exec_command</code> returns after a short wait with a session ID if the process is still running, and
<code>write_stdin</code> types into it or reads its new output.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–5. Code: <code>code/a06_running_commands</code>
(<code>run_harness.py</code> needs a local Ollama with <code>qwen3:4b</code>; run it in a container, it executes the
model's commands and starts a server on port 8000).</div>""",
}

CONCEPTS = [
    {
        "title": "Commands that never end",
        "segment": (7.7, 57.8),
        "figures": [{"t": 19.2, "caption": "Commands that never end on their own, and the harness of episodes 1–5: run "
                                           "to the end, killed after 30 s."},
                    {"t": 48.9, "caption": "Blocking runs 1 and 3: server.py & still blocked for 30 s; the server "
                                           "outlived the timeout, untracked."},
                    {"t": 57.5, "caption": "Blocking run 2: the model never started the server and asked the user to."}],
        "body": [
            """<p>The task: start <code>server.py</code> (it runs until stopped), check it with <code>check.py</code>,
report the version. With a blocking <code>shell</code> tool, <code>subprocess.run(…, capture_output=True,
timeout=30)</code>, two of three runs started the server in the background with <code>&amp;</code>, and the harness
still waited the full 30 seconds.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Why: capturing output means holding pipes to the command, and the background server inherits them,
so the end of output never comes. When the timeout killed the shell, the server survived: <code>check.py</code> then
worked and run 3 reported <b>1.4.2</b>, by accident, from a process the harness no longer knew about. In the other run
the model never started the server, got <i>Connection refused</i>, and told the user to start it.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>“Run it and wait” breaks on long-running processes: the
harness either blocks until a timeout or leaves processes running out of its control.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why did <code>python3 server.py &amp;</code> still make the harness wait 30 seconds?",
             "answer": "The harness captures output through pipes; the background server inherited them and kept them "
                       "open, so the command's output never ended.",
             "why": "subprocess.run waits for end-of-output, not just for the shell to exit."},
            {"kind": "tf", "q": "“When the harness's timeout killed the shell, the server stopped too.”",
             "answer": "False.", "why": "The server kept running; that is why check.py worked afterwards.",
             "key": {"value": False}},
            {"kind": "number", "q": "Of the 3 blocking runs, how many reported the right version?",
             "answer": "1 (run 3), by accident.", "why": "Run 2 was wrong; run 1 crashed before answering.",
             "key": {"parts": [{"label": None, "value": 1, "tol": 0.5, "unit": None}]}},
        ],
    },
    {
        "title": "Sessions: exec_command and write_stdin",
        "segment": (58.3, 80.6),
        "figures": [{"t": 71.1, "caption": "exec_command returns after yield_time_ms (10 s by default); a running process "
                                           "keeps a session ID."},
                    {"t": 80.3, "caption": "write_stdin types into a session, or reads its new output when sent nothing."}],
        "body": [
            """<p>Codex's <code>exec_command</code> starts the command, waits <code>yield_time_ms</code> (default
10,000 ms, clamped to 250–30,000), and returns whatever output has arrived. If the process has exited, the result
gives its exit code; if not, it says <code>Process running with session ID …</code>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The second tool, <code>write_stdin(session_id, chars)</code>, writes characters to that process's
input (for example <code>"y\\n"</code> for a yes/no prompt) or, with an empty string, just returns the output produced
since the last call, after a short wait (250 ms by default).</p>""",
            """<div class="box key"><b class="t">Key idea</b>A session turns a long-running process into something the
model can come back to: start it, get control back, then read from it or type into it.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A model calls exec_command with yield_time_ms = 60,000. How long does Codex wait at "
                                    "most before returning, in seconds?",
             "answer": "30.", "why": "yield_time_ms is clamped to at most 30,000 ms.",
             "key": {"parts": [{"label": None, "value": 30, "tol": 0.5, "unit": "s"}]}},
            {"kind": "mc", "q": "What does <code>write_stdin(1, \"\")</code> do?",
             "options": ["Stops session 1", "Returns session 1's new output", "Restarts session 1", "Clears its output"],
             "answer": "B.", "why": "Sending nothing just reads what the process printed since the last call.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "Sessions in practice, and their limits",
        "segment": (81.1, 120.7),
        "figures": [{"t": 98.3, "caption": "Sessions run 2: server running as session 1 after 10.1 s, check exits 0, "
                                           "version 1.4.2 in 2 model calls."},
                    {"t": 112.3, "caption": "Limits: at most 64 processes in Codex, an output cap on every result, and "
                                            "cleanup at the end."},
                    {"t": 120.4, "caption": "A session: a process plus a thread that collects its output; each call waits "
                                            "a little and hands over what came in."}],
        "body": [
            """<p>With the two session tools, run 2 started the server (<i>Process running with session ID 1</i> after
10.1 s), ran <code>check.py</code> (exit 0) and answered <b>version 1.4.2</b> in two model calls, with 10 seconds of
waiting for commands. In run 3 the model's first reply was empty and the run ended without an answer: the tools fix the
harness's problem, not the model's.</p>""",
            "{fig0}",
            """<p>Running processes need limits. Codex keeps at most 64; when the store is full it prunes old ones,
exited ones first. Every result still has an output cap (<code>max_output_tokens</code>), and at the end someone has
to stop what is still running: our harness starts each session in its own process group and kills the group.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>Sessions keep long-running processes under the harness's
control: bounded in number, capped in output, and stopped when the work is done.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Sessions run 2 waited 10.1 s for the server and 0.2 s for the check. How many "
                                    "seconds of command waiting in total, to one decimal?",
             "answer": "10.3.", "why": "10.1 + 0.2 = 10.3 (the log rounds to 10 s).",
             "key": {"parts": [{"label": None, "value": 10.3, "tol": 0.05, "unit": "s"}]}},
            {"kind": "tf", "q": "“With sessions, every run in our test reported the right version.”",
             "answer": "False.", "why": "Run 3's first reply was empty; it ended without an answer.",
             "key": {"value": False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Collect a running process's output without blocking.",
             "code": """class Session:
    def __init__(self, cmd, cwd):
        self.p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, stdin=PIPE,
                                  stdout=PIPE, stderr=STDOUT, text=True)
        self.buf = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.buf.append(line)

    def collect(self, yield_ms):
        end = time.time() + yield_ms / 1000
        while time.time() < end and self.p.poll() is None:
            time.sleep(0.05)
        out, self.buf = "".join(self.buf), []
        return out""",
             "answer": "run_harness.py's Session adds a lock around buf and a process group for cleanup.",
             "why": "The thread reads continuously, so the harness never blocks on the pipe."},
        ],
    },
]
