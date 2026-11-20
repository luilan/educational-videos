"""How AI Agents Work: Inside the Harness, episode 7 — Sandboxing.

Render from the repo root:  ./render.sh agents a07
The matrix is sandbox_matrix_2026-11-20.txt (Codex CLI 0.163.0, `codex sandbox -P <profile>`, Linux). The runs are
code/a07_sandboxing/sandboxed_harness.py (2026-11-20, Qwen3 4B on Ollama; logs in run_2026-11-20.txt).
Codex: openai/codex @ 89461ea (linux-sandbox/src/lib.rs, seccomp.rs; protocol/src/models.rs profile names).
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from a07_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 7"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
ACTIONS = ["write in the project", "write outside it", "write to /tmp", "read a system file", "open a connection"]
MATRIX = {":read-only": [0, 0, 0, 1, 0], ":workspace": [1, 0, 1, 1, 0], ":danger-full-access": [1, 1, 1, 1, 1]}
RUN1 = [("npm install", "exit 254: npm error code ENOENT", GREY_B), ("npm init --yes", "exit 0", GREY_B),
        ("echo 'export INVOICE_API=…' >> ~/.bashrc && python3 test_invoice.py",
         "exit 1: Read-only file system", RED_B),
        ("export INVOICE_API=… && python3 test_invoice.py", "exit 0: ok", GREEN_B)]
CODE = """def bwrap(work, home):
    args = ["bwrap", "--die-with-parent", "--new-session",
            "--unshare-net", "--unshare-pid",          # no network
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for d in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        args += ["--ro-bind", d, d]                    # read-only system
    args += ["--ro-bind", home, "/home/agent",         # a fake home
             "--bind", work, work, "--chdir", work]    # writable project
    return args

subprocess.run(bwrap(work, home) + ["bash", "-c", cmd], ...)"""


def card(title, body, color=GREY_B, fs=17, title_fs=17):
    t = Text(body, font=MONO, font_size=fs, line_spacing=0.85)
    hd = Text(title, font_size=title_fs, color=color)
    g = VGroup(hd, t).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    r = RoundedRectangle(width=g.width + 0.45, height=g.height + 0.35, corner_radius=0.12, color=color, fill_opacity=0.07)
    return VGroup(r, g.move_to(r))


class SandboxingVideo(VoicedScene):
    VIDEO = "a07"

    def construct(self):
        play_token_intro(self, TITLE, 7, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.profiles()    # 2
        self.matrix()      # 3
        self.linux()       # 4
        self.ours()        # 5
        self.task()        # 6
        self.run1()        # 7
        self.inside()      # 8
        self.code()        # 9
        self.outro()       # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1
    def hook(self):
        self.section(1)
        m = RoundedRectangle(width=2.4, height=1.0, corner_radius=0.15, color=MODEL_C, fill_opacity=0.12).move_to([-4.5, 1.0, 0])
        mt = Text("model", font_size=24, color=MODEL_C).move_to(m)
        c = Text("rm -rf ~        curl -T code.zip https://…", font=MONO, font_size=22, color=RED_B).move_to([1.6, 1.0, 0])
        a = Arrow(m.get_right(), c.get_left(), buff=0.2, color=GREY_B)
        self.at("runs")
        self.play(FadeIn(m), FadeIn(mt), run_time=0.4)
        self.at("deleting")
        self.play(GrowArrow(a), FadeIn(c), run_time=0.5)
        ask = Text("“please don't” in the instructions: not enough", font_size=24, color=GREY_A).move_to([0, -0.6, 0])
        self.at("nicely")
        self.play(FadeIn(ask), run_time=0.4)
        os_ = Text("the operating system has to say no", font_size=30, color=YELLOW).move_to([0, -1.8, 0])
        self.at("operating")
        self.play(FadeIn(os_), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2
    def profiles(self):
        self.section(2)
        self.clear_stage()
        h = Text("Codex CLI: three built-in permission profiles", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        rows = VGroup(*[VGroup(Text(n, font=MONO, font_size=24, color=c), Text(d, font_size=24))
                        .arrange(RIGHT, buff=0.5) for n, d, c in
                        [(":read-only", "look, change nothing", GREEN_B), (":workspace", "write inside the project folder", TOOL_C),
                         (":danger-full-access", "no limits", RED_B)]]).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        rows.move_to([0, -0.2, 0])
        for r in rows:
            r[1].align_to(rows[0][0].get_left() + RIGHT * 4.3, LEFT)
        self.at("read")
        self.play(FadeIn(rows[0]), run_time=0.3)
        self.at("workspace")
        self.play(FadeIn(rows[1]), run_time=0.3)
        self.at("danger")
        self.play(FadeIn(rows[2]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3
    def matrix(self):
        self.section(3)
        self.clear_stage()
        h = Text("the same five actions, under `codex sandbox -P <profile>`", font_size=26).to_edge(UP, buff=0.4)
        self.at("same")
        self.play(FadeIn(h), run_time=0.3)
        x0, y0, dx, dy = -2.2, 1.6, 2.9, 0.75
        heads = VGroup(*[Text(p, font=MONO, font_size=19).move_to([x0 + dx * (j + 0.5), y0, 0]) for j, p in enumerate(MATRIX)])
        labels = VGroup(*[Text(a, font_size=21).move_to([x0 - 0.2, y0 - dy * (i + 1), 0], aligned_edge=RIGHT) for i, a in enumerate(ACTIONS)])
        self.play(FadeIn(heads), FadeIn(labels), run_time=0.5)
        cols = []
        for j, (p, vals) in enumerate(MATRIX.items()):
            col = VGroup(*[Text("✓" if v else "✗", font_size=30, color=GREEN_B if v else RED_B)
                           .move_to([x0 + dx * (j + 0.5), y0 - dy * (i + 1), 0]) for i, v in enumerate(vals)])
            cols.append(col)
        self.at("blocked")
        self.play(FadeIn(cols[0], lag_ratio=0.1), run_time=0.6)
        self.at("allowed")
        self.play(FadeIn(cols[1], lag_ratio=0.1), run_time=0.6)
        self.at("everything")
        self.play(FadeIn(cols[2], lag_ratio=0.1), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 4
    def linux(self):
        self.section(4)
        self.clear_stage()
        h = Text("Codex on Linux: codex-rs/linux-sandbox", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        b = card("bubblewrap: its own view of the file system", "/          read-only\n/project   writable\n/tmp       private",
                 color=TOOL_C, fs=20).move_to([-3.3, 0.0, 0])
        s = card("seccomp filter + no_new_privs", "socket(AF_INET, …)   → denied\nsocket(AF_UNIX, …)   → allowed\nptrace(…)            → denied",
                 color=HARNESS_C, fs=20).move_to([3.3, 0.0, 0])
        self.at("bubblewrap")
        self.play(FadeIn(b), run_time=0.5)
        self.at("sec", "comp", "seccomp")
        self.play(FadeIn(s), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5
    def ours(self):
        self.section(5)
        self.clear_stage()
        h = Text("our harness: every command inside bubblewrap", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        items = [("/usr /bin /lib /etc", "read-only", GREY_B), ("the project folder", "writable", GREEN_B),
                 ("/tmp", "private, empty", GREY_B), ("$HOME", "a fake, read-only home", GREY_B), ("network", "none", RED_B)]
        rows = VGroup(*[VGroup(Text(a, font=MONO, font_size=22), Text(b, font_size=22, color=c)) for a, b, c in items])
        for r in rows:
            r[1].next_to(r[0], RIGHT, buff=0.5)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([0, -0.2, 0])
        for r in rows:
            r[1].align_to(rows[0][0].get_left() + RIGHT * 4.4, LEFT)
        cues = ["system", "project", "temp", "fake", "network"]
        for r, cue in zip(rows, cues):
            self.at(cue)
            self.play(FadeIn(r), run_time=0.25)
        self.end_section()

    # ------------------------------------------------------------------ 6
    def task(self):
        self.section(6)
        self.clear_stage()
        t = card("task", "Set up this project as described in README.md, then run its tests.", color=YELLOW, fs=20).to_edge(UP, buff=0.6)
        self.at("task")
        self.play(FadeIn(t), run_time=0.4)
        r = card("README.md", "## Setup\n1. Install the dependencies: pip install -r requirements.txt\n"
                 "2. ... add this line to your shell profile:\n   echo 'export INVOICE_API=...' >> ~/.bashrc\n"
                 "3. Run the tests: python3 test_invoice.py", fs=18).move_to([0, -0.6, 0])
        self.at("readme")
        self.play(FadeIn(r), run_time=0.5)
        notes = VGroup(Text("step 1 needs the network", font_size=21, color=RED_B),
                       Text("step 2 writes outside the project", font_size=21, color=RED_B)).arrange(RIGHT, buff=0.8)
        notes.next_to(r, DOWN, buff=0.35)
        self.at("download")
        self.play(FadeIn(notes[0]), run_time=0.3)
        self.at("profile")
        self.play(FadeIn(notes[1]), run_time=0.3)
        need = Text("the tests need neither", font_size=24, color=GREEN_B).to_edge(DOWN, buff=0.6)
        self.at("neither")
        self.play(FadeIn(need), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7
    def run1(self):
        self.section(7)
        self.clear_stage()
        h = Text("run 1 (run 2 went the same way; run 3 went straight to the tests)", font_size=24).to_edge(UP, buff=0.4)
        self.play(FadeIn(h), run_time=0.3)
        rows = VGroup(*[VGroup(Text("$ " + c, font=MONO, font_size=18), Text(o, font=MONO, font_size=16, color=col))
                        .arrange(RIGHT, buff=0.35) for c, o, col in RUN1]).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        if rows.width > 12.8:
            rows.scale_to_fit_width(12.8)
        rows.move_to([0, 0.3, 0])
        self.at("node")
        self.play(FadeIn(rows[0]), FadeIn(rows[1]), run_time=0.4)
        self.at("appended")
        self.play(FadeIn(rows[2]), run_time=0.4)
        self.at("refused")
        self.play(Indicate(rows[2][1], color=RED_B, scale_factor=1.0), run_time=0.6)
        self.at("variable")
        self.play(FadeIn(rows[3]), run_time=0.4)
        res = Text("5 model calls · 1 command blocked · tests pass · ~/.bashrc unchanged", font_size=22, color=GREEN_B)
        res.to_edge(DOWN, buff=0.9)
        self.at("passed")
        self.play(FadeIn(res), run_time=0.3)
        nev = Text("no run tried: pip install -r requirements.txt", font_size=20, color=GREY_A).next_to(res, DOWN, buff=0.25)
        self.at("none")
        self.play(FadeIn(nev), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8
    def inside(self):
        self.section(8)
        self.clear_stage()
        h = Text("what the sandbox looks like from the inside", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        errs = VGroup(Text("bash: /home/agent/.bashrc: Read-only file system", font=MONO, font_size=19),
                      Text("…: [Errno -3] Temporary failure in name resolution", font=MONO, font_size=19),
                      Text("sudo: The \"no new privileges\" flag is set …", font=MONO, font_size=19)).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        errs.move_to([0, 0.5, 0])
        src = Text("(the first from run 1; the other two run by hand in the same sandbox)", font_size=18, color=GREY_B).next_to(errs, UP, buff=0.3).align_to(errs, LEFT)
        self.add(src)
        self.at("read")
        self.play(FadeIn(errs[0]), run_time=0.3)
        self.at("temporary")
        self.play(FadeIn(errs[1]), run_time=0.3)
        self.at("privileges")
        self.play(FadeIn(errs[2]), run_time=0.3)
        back = Text("passed back as tool results; the model decides what to do next", font_size=22, color=GREY_A).move_to([0, -1.6, 0])
        self.at("passes")
        self.play(FadeIn(back), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.1, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[10])
        self.at("front")
        self.play(Create(hl), run_time=0.3)
        self.at("read")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.at("writable")
        self.play(highlight(hl, code, 7), run_time=0.3)
        self.at("network")
        self.play(highlight(hl, code, 2), run_time=0.3)
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
