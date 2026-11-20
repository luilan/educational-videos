"""How AI Agents Work: Inside the Harness, episode 6 — Running Commands.

Render from the repo root:  ./render.sh agents a06
Every command, wait and answer is from code/a06_running_commands/run_harness.py (2026-11-20, Qwen3 4B on Ollama; logs
in run_2026-11-20.txt). Codex: openai/codex @ 89461ea (core/src/tools/handlers/unified_exec.rs: yield_time_ms default
10,000, write_stdin default 250; core/src/unified_exec/mod.rs: 250-30,000 ms, 64 processes; tools/context.rs).
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from a06_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 6"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
CODE = """class Session:
    def __init__(self, cmd, cwd):
        self.p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd,
                                  stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        self.buf = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):                  # runs in the background
        for line in self.p.stdout:
            self.buf.append(line)

    def collect(self, yield_ms):      # wait a little, return what came in
        end = time.time() + yield_ms / 1000
        while time.time() < end and self.p.poll() is None:
            time.sleep(0.05)
        out, self.buf = "".join(self.buf), []
        return out"""


def card(title, body, color=GREY_B, fs=17, title_fs=17):
    t = Text(body, font=MONO, font_size=fs, line_spacing=0.85)
    hd = Text(title, font_size=title_fs, color=color)
    g = VGroup(hd, t).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    r = RoundedRectangle(width=g.width + 0.45, height=g.height + 0.35, corner_radius=0.12, color=color, fill_opacity=0.07)
    return VGroup(r, g.move_to(r))


def row(cmd, wait, result, color=WHITE):
    return VGroup(Text(cmd, font=MONO, font_size=19, color=color), Text(wait, font=MONO, font_size=17, color=GREY_B),
                  Text(result, font_size=18, color=GREY_A))


def table(rows, x_cols=(-6.3, -0.6, 0.9)):
    for i, r in enumerate(rows):
        for m, x in zip(r, x_cols):
            m.move_to([x, 0, 0], aligned_edge=LEFT)
    g = VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
    for r in rows:
        for m, x in zip(r, x_cols):
            m.align_to([x, 0, 0], LEFT)
    return g


class RunningCommandsVideo(VoicedScene):
    VIDEO = "a06"

    def construct(self):
        play_token_intro(self, TITLE, 6, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.task()        # 2
        self.blocking1()   # 3
        self.blocking2()   # 4
        self.sessions()    # 5
        self.stdin()       # 6
        self.session_run() # 7
        self.limits()      # 8
        self.code()        # 9
        self.outro()       # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1
    def hook(self):
        self.section(1)
        items = VGroup(*[Text(t, font=MONO, font_size=24) for t in ("python3 server.py", "npm run watch", "rm -i old.log   (waits for y/n)")])
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([-2.5, 1.0, 0])
        inf = VGroup(*[Text("never ends on its own", font_size=20, color=YELLOW).next_to(m, RIGHT, buff=0.6) for m in items])
        inf.align_to(items, RIGHT).shift(RIGHT * 4.0)
        self.at("server")
        self.play(FadeIn(items[0]), run_time=0.3)
        self.at("watcher")
        self.play(FadeIn(items[1]), run_time=0.3)
        self.at("input")
        self.play(FadeIn(items[2]), FadeIn(inf), run_time=0.4)
        rule = card("our harness so far (episodes 1–5)", "subprocess.run(cmd, capture_output=True, timeout=30)",
                    color=HARNESS_C, fs=20).move_to([0, -1.9, 0])
        self.at("waits")
        self.play(FadeIn(rule), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2
    def task(self):
        self.section(2)
        self.clear_stage()
        t = card("the task", "Start the web service with: python3 server.py (it keeps running until\n"
                 "stopped). While it runs, check it with: python3 check.py.\nReport the version that the service returns.",
                 color=YELLOW, fs=19).move_to([0, 1.75, 0])
        self.at("task")
        self.play(FadeIn(t), run_time=0.5)
        s = card("server.py", 'print("Serving on http://localhost:8000")\nHTTPServer(("localhost", 8000), Handler).serve_forever()',
                 fs=17).move_to([0, -0.75, 0])
        c = card("check.py", 'print(urlopen(".../health").read())\n→ {"status": "ok", "version": "1.4.2"}', fs=17).move_to([0, -2.55, 0])
        self.at("service")
        self.play(FadeIn(s), run_time=0.4)
        self.at("second")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3
    def blocking1(self):
        self.section(3)
        self.clear_stage()
        h = Text("blocking harness, runs 1 and 3", font_size=26).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        r1 = row("$ python3 server.py &", "30.0 s", "timed out, killed")
        r2 = row("$ python3 check.py", " 0.1 s", "exit 0")
        tb = table([r1, r2]).move_to([0, 1.0, 0])
        self.at("background")
        self.play(FadeIn(r1[0]), run_time=0.3)
        bar = Rectangle(width=0.1, height=0.35, stroke_width=0, fill_color=RED_C, fill_opacity=0.9).move_to([-6.25, -0.6, 0], aligned_edge=LEFT)
        bl = Text("harness waiting", font_size=18, color=RED_B).next_to(bar, UP, buff=0.12).align_to(bar, LEFT)
        self.at("waited")
        self.play(FadeIn(bl), FadeIn(bar), run_time=0.2)
        self.play(bar.animate.stretch_to_fit_width(9.0, about_edge=LEFT), FadeIn(r1[1]), FadeIn(r1[2]), run_time=1.6)
        why = Text("capture_output keeps a pipe to the server's output;\nthe background server keeps it open, so “done” never comes",
                   font_size=21, color=GREY_A, line_spacing=0.9).move_to([0, -1.7, 0])
        self.at("captures")
        self.play(FadeIn(why), run_time=0.5)
        self.at("killed")
        self.play(FadeIn(r2), run_time=0.4)
        acc = Text("right answer (run 3), by accident: the server outlived the timeout, untracked", font_size=21,
                   color=YELLOW).to_edge(DOWN, buff=0.4)
        self.at("accident")
        self.play(FadeIn(acc), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4
    def blocking2(self):
        self.section(4)
        self.clear_stage()
        h = Text("blocking harness, run 2", font_size=26).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        r = row("$ python3 check.py", " 0.1 s", "exit 1: Connection refused")
        tb = table([r]).move_to([0, 1.2, 0])
        self.at("check")
        self.play(FadeIn(tb), run_time=0.4)
        ans = card("final answer", "The server is not running yet. Please start the web service\nwith `python3 server.py` first, "
                   "then run `python3 check.py` ...", color=RED_B, fs=18).move_to([0, -0.9, 0])
        self.at("told")
        self.play(FadeIn(ans), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5
    def sessions(self):
        self.section(5)
        self.clear_stage()
        h = Text("Codex: exec_command", font_size=28).to_edge(UP, buff=0.5)
        self.at("codex")
        self.play(FadeIn(h), run_time=0.3)
        t0 = Line([-5.5, 0.9, 0], [5.5, 0.9, 0], color=GREY_D)
        start = Dot([-5.5, 0.9, 0], color=TOOL_C)
        y = Dot([-0.5, 0.9, 0], color=HARNESS_C)
        lab = VGroup(Text("command starts", font_size=18).next_to(start, UP, buff=0.2),
                     Text("yield_time_ms: 10 s by default\n(250 ms – 30 s)", font_size=18, color=HARNESS_C, line_spacing=0.9).next_to(y, UP, buff=0.2))
        self.play(Create(t0), FadeIn(start), FadeIn(lab[0]), run_time=0.4)
        self.at("10", "ten")
        self.play(FadeIn(y), FadeIn(lab[1]), run_time=0.4)
        res = card("our harness's result, while the server runs", "Process running with session ID 1\nOutput:\nServing on http://localhost:8000", color=HARNESS_C, fs=18).move_to([0, -1.6, 0])
        self.at("returns")
        self.play(FadeIn(res), run_time=0.5)
        self.at("session")
        self.play(Indicate(res[1][1], color=HARNESS_C, scale_factor=1.0), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 6
    def stdin(self):
        self.section(6)
        self.clear_stage()
        h = Text("write_stdin(session_id, chars)", font=MONO, font_size=28).to_edge(UP, buff=0.5)
        self.at("second", "write")
        self.play(FadeIn(h), run_time=0.3)
        a = card("type into it", 'write_stdin(1, "y\\n")  →  the answer to a y/n prompt', color=TOOL_C, fs=19).move_to([0, 0.8, 0])
        b = card("or just read", 'write_stdin(1, "")  →  new output since the last call\n(waits 250 ms by default)', color=TOOL_C, fs=19).move_to([0, -1.2, 0])
        self.at("type")
        self.play(FadeIn(a), run_time=0.4)
        self.at("nothing")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7
    def session_run(self):
        self.section(7)
        self.clear_stage()
        h = Text("sessions harness (exec_command + write_stdin), run 2", font_size=26).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        r1 = row("exec_command: python3 server.py", "10.1 s", "Process running with session ID 1")
        r2 = row("exec_command: python3 check.py", " 0.2 s", "Process exited with code 0")
        tb = table([r1, r2], x_cols=(-6.3, -0.4, 1.0)).move_to([0, 1.0, 0])
        self.at("started")
        self.play(FadeIn(r1[0]), run_time=0.3)
        self.at("session")
        self.play(FadeIn(r1[1]), FadeIn(r1[2]), run_time=0.4)
        self.at("check")
        self.play(FadeIn(r2), run_time=0.4)
        ans = Text("answer: “The web service reports version 1.4.2.”", font_size=26, color=GREEN_B).move_to([0, -1.0, 0])
        self.at("answered")
        self.play(FadeIn(ans), run_time=0.4)
        n = Text("2 model calls · 10 s waiting for commands", font_size=22, color=GREY_A).next_to(ans, DOWN, buff=0.4)
        self.at("two")
        self.play(FadeIn(n), run_time=0.3)
        other = Text("sessions run 3: an empty first reply, no answer", font_size=22, color=RED_B).next_to(n, DOWN, buff=0.5)
        self.at("other")
        self.play(FadeIn(other), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8
    def limits(self):
        self.section(8)
        self.clear_stage()
        h = Text("keeping running processes in check", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        pts = VGroup(Text("Codex: at most 64 processes; when full, old ones are closed (exited first)", font_size=22),
                     Text("every result still has an output cap (max_output_tokens)", font_size=22),
                     Text("at the end, the harness stops what is still running", font_size=22)).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        pts.move_to([0, 0.2, 0])
        self.at("sixty", "64", "most")
        self.play(FadeIn(pts[0]), run_time=0.4)
        self.at("cap")
        self.play(FadeIn(pts[1]), run_time=0.4)
        self.at("stop")
        self.play(FadeIn(pts[2]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.1, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("process")
        self.play(Create(hl), run_time=0.3)
        self.at("thread")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.at("waits")
        self.play(highlight(hl, code, 13), run_time=0.3)
        self.at("hands")
        self.play(highlight(hl, code, 16), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10
    def outro(self):
        self.section(10)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
