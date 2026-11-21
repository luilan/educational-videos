"""How AI Agents Work: Inside the Harness, episode 12 — Sub-agents and Parallel Tool Calls.

Render from the repo root:  ./render.sh agents a12
Runs: code/a12_subagents_and_parallel/parallel_harness.py (2026-11-21, Qwen3 4B on Ollama; logs in run_2026-11-21.txt).
Codex: openai/codex @ 89461ea (core/src/tools/parallel.rs read/write lock; handlers/multi_agents_spec.rs;
core/src/config/mod.rs DEFAULT_AGENT_MAX_THREADS = 6, DEFAULT_AGENT_MAX_DEPTH = 1).
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from a12_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 12"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
CODE = """# parallel tool calls
with ThreadPoolExecutor() as pool:
    outs = list(pool.map(do, calls))        # all calls from one reply

# a sub-agent: the same loop, a new conversation
def do(call):
    if call["name"] == "spawn_agent":
        task = json.loads(call["arguments"])["message"]
        answer, sent, steps = agent(task, [SHELL], SEARCH, work)
        return answer                       # only this comes back
    return shell(json.loads(call["arguments"])["command"], work)"""


def card(title, body, color=GREY_B, fs=17, title_fs=17):
    t = Text(body, font=MONO, font_size=fs, line_spacing=0.85)
    hd = Text(title, font_size=title_fs, color=color)
    g = VGroup(hd, t).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    r = RoundedRectangle(width=g.width + 0.45, height=g.height + 0.35, corner_radius=0.12, color=color, fill_opacity=0.07)
    return VGroup(r, g.move_to(r))


def lane(y, label, color):
    return VGroup(Text(label, font=MONO, font_size=17).move_to([-5.2, y, 0], aligned_edge=RIGHT))


class SubagentsParallelVideo(VoicedScene):
    VIDEO = "a12"

    def construct(self):
        play_token_intro(self, TITLE, 12, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.calls()       # 2
        self.task()        # 3
        self.timing()      # 4
        self.codex_lock()  # 5
        self.subagents()   # 6
        self.context()     # 7
        self.same()        # 8
        self.codex_agents()  # 9
        self.code()        # 10
        self.outro()       # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1
    def hook(self):
        self.section(1)
        a = Text("several tool calls in one reply, run at the same time", font_size=26, color=TOOL_C)
        b = Text("sub-agents: a fresh agent for one piece of the job", font_size=26, color=MODEL_C)
        g = VGroup(a, b).arrange(DOWN, buff=0.8).move_to([0, 0.2, 0])
        self.at("several")
        self.play(FadeIn(a), run_time=0.4)
        self.at("subagents", "sub")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2
    def calls(self):
        self.section(2)
        self.clear_stage()
        r = card("one model reply (run 1, step 1)", '{"type": "function_call", "name": "shell", "arguments": "{\\"command\\": \\"python3 check.py lint\\"}"}\n'
                 '{"type": "function_call", "name": "shell", "arguments": "{\\"command\\": \\"python3 check.py types\\"}"}\n'
                 '{"type": "function_call", "name": "shell", "arguments": "{\\"command\\": \\"python3 check.py tests\\"}"}',
                 color=MODEL_C, fs=15).move_to([0, 0.5, 0])
        if r.width > 13:
            r.scale_to_fit_width(13)
        self.at("reply")
        self.play(FadeIn(r), run_time=0.5)
        opts = Text("one after another, or all at once?", font_size=26, color=YELLOW).move_to([0, -1.8, 0])
        self.at("after", "once")
        self.play(FadeIn(opts), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3
    def task(self):
        self.section(3)
        self.clear_stage()
        c = card("demo_checks/check.py", 'time.sleep(3)\nprint({"lint": "lint: 0 problems", "types": "types: 0 errors",\n       "tests": "tests: 14 passed"}[sys.argv[1]])', fs=18).move_to([0, 1.0, 0])
        self.at("three")
        self.play(FadeIn(c), run_time=0.5)
        n = Text("all 4 runs: the three calls in a single reply", font_size=24, color=GREEN_B).move_to([0, -1.5, 0])
        self.at("single")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4
    def timing(self):
        self.section(4)
        self.clear_stage()
        x0, sc = -3.6, 0.9
        def bar(y, start, color, label):
            b = Rectangle(width=3 * sc, height=0.45, stroke_width=0, fill_color=color, fill_opacity=0.9)
            b.move_to([x0 + (start + 1.5) * sc, y, 0])
            return VGroup(b, Text(label, font=MONO, font_size=15).move_to(b))
        hs = Text("serial: 9.0 s", font_size=24).move_to([-5.4, 1.6, 0], aligned_edge=LEFT)
        ser = VGroup(bar(0.9, 0, TOOL_C, "lint"), bar(0.9, 3, TOOL_C, "types"), bar(0.9, 6, TOOL_C, "tests"))
        hp = Text("parallel: 3.0 s", font_size=24).move_to([-5.4, -0.4, 0], aligned_edge=LEFT)
        par = VGroup(bar(-1.0, 0, GREEN_C, "lint"), bar(-1.6, 0, GREEN_C, "types"), bar(-2.2, 0, GREEN_C, "tests"))
        self.at("another")
        self.play(FadeIn(hs), FadeIn(ser, lag_ratio=0.5), run_time=1.0)
        self.at("same")
        self.play(FadeIn(hp), FadeIn(par), run_time=0.5)
        s = Text("same answer · 495 tokens sent in both", font_size=22, color=GREY_A).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(s), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5
    def codex_lock(self):
        self.section(5)
        self.clear_stage()
        h = Text("Codex · core/src/tools/parallel.rs", font_size=26).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        a = card("supports_parallel_tool_calls() → true", "lock.read()   ← many at once\n(exec_command, write_stdin, view_image, …)", color=GREEN_B, fs=19).move_to([-3.2, 0.0, 0])
        b = card("→ false", "lock.write()  ← one at a time", color=RED_B, fs=19).move_to([3.6, 0.0, 0])
        self.at("safe")
        self.play(FadeIn(a), run_time=0.4)
        self.at("write")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6
    def subagents(self):
        self.section(6)
        self.clear_stage()
        p = RoundedRectangle(width=5.2, height=3.6, corner_radius=0.15, color=MODEL_C, fill_opacity=0.06).move_to([-3.6, -0.2, 0])
        pl = Text("parent conversation", font_size=19, color=MODEL_C).next_to(p, UP, buff=0.1)
        c = RoundedRectangle(width=5.2, height=3.6, corner_radius=0.15, color=TOOL_C, fill_opacity=0.06).move_to([3.6, -0.2, 0])
        cl = Text("child conversation (fresh)", font_size=19, color=TOOL_C).next_to(c, UP, buff=0.1)
        self.at("spawn")
        self.play(Create(p), FadeIn(pl), run_time=0.4)
        call = Text('spawn_agent("Find the line …")', font=MONO, font_size=16).move_to(p.get_center() + UP * 1.0)
        self.at("writes")
        self.play(FadeIn(call), run_time=0.3)
        a1 = Arrow(p.get_right() + UP * 1.0, c.get_left() + UP * 1.0, buff=0.1, color=GREY_B)
        self.at("fresh")
        self.play(Create(c), FadeIn(cl), GrowArrow(a1), run_time=0.5)
        work = VGroup(*[Text(t, font=MONO, font_size=15) for t in ("$ grep -rn 'study-guide' .", "$ grep -rn 'figure' .", "$ sed -n '224p' …")]).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to(c.get_center() + UP * 0.2)
        self.play(FadeIn(work), run_time=0.4)
        a2 = Arrow(c.get_left() + DOWN * 1.0, p.get_right() + DOWN * 1.0, buff=0.1, color=GREEN_B)
        ans = Text("only the final answer", font_size=17, color=GREEN_B).move_to(p.get_center() + DOWN * 1.0)
        self.at("answer")
        self.play(GrowArrow(a2), FadeIn(ans), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7
    def context(self):
        self.section(7)
        self.clear_stage()
        h = Text("tokens sent, episode 4's search task", font_size=26).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        data = [("single agent", 1153, BLUE_D), ("sub-agent run 1: parent", 619, MODEL_C), ("     child", 2259, TOOL_C),
                ("sub-agent run 2: parent", 648, MODEL_C), ("     child", 954, TOOL_C)]
        rows = VGroup()
        for lab, v, c in data:
            b = Rectangle(width=v * 0.0022, height=0.4, stroke_width=0, fill_color=c, fill_opacity=0.9)
            rows.add(VGroup(Text(lab.strip(), font_size=19), b, Text(f"{v:,}", font=MONO, font_size=17)))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        for r, (lab, _, _) in zip(rows, data):
            r[0].move_to([-1.2, r[0].get_y(), 0], aligned_edge=RIGHT)
            if lab.startswith(" "):
                r[0].set_color(GREY_B)
            r[1].move_to([-0.9, r[0].get_y(), 0], aligned_edge=LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.15)
        rows.move_to([0.6, -0.2, 0])
        self.at("single")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("parent")
        self.play(FadeIn(rows[1]), FadeIn(rows[3]), run_time=0.4)
        self.at("childs", "child")
        self.play(FadeIn(rows[2]), FadeIn(rows[4]), run_time=0.4)
        n = Text("the child's tokens never enter the parent's conversation", font_size=21, color=GREY_A).to_edge(DOWN, buff=0.4)
        self.at("never")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8
    def same(self):
        self.section(8)
        self.clear_stage()
        a = Text("single agent, sub-agent run 1, sub-agent run 2:", font_size=24).move_to([0, 1.8, 0])
        b = Text("framework/study_guide.py, line 224", font=MONO, font_size=26, color=RED_B).next_to(a, DOWN, buff=0.35)
        self.at("three")
        self.play(FadeIn(a), FadeIn(b), run_time=0.4)
        k = Text("a sub-agent keeps the parent's context small, not the search smarter", font_size=24, color=YELLOW).move_to([0, -0.2, 0])
        self.at("keeps")
        self.play(FadeIn(k), run_time=0.4)
        brief = card("the parent's task for the child (run 1)", "Find the line in the project that specifies the file name of\na figure image for the study-guide PDF.",
                     fs=17).move_to([0, -1.9, 0])
        miss = Text("(no “exact format string”)", font_size=18, color=GREY_A).next_to(brief, RIGHT, buff=0.2)
        if miss.get_right()[0] > 7:
            miss.next_to(brief, DOWN, buff=0.1)
        self.at("left")
        self.play(FadeIn(brief), FadeIn(miss), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9
    def codex_agents(self):
        self.section(9)
        self.clear_stage()
        tools = VGroup(*[Text(t, font=MONO, font_size=24) for t in ("spawn_agent", "send_input", "wait_agent", "close_agent")]).arrange(RIGHT, buff=0.6)
        tools.move_to([0, 1.0, 0])
        self.at("spawn")
        self.play(FadeIn(tools, lag_ratio=0.3), run_time=0.8)
        lim = VGroup(Text("at most 6 agent threads (DEFAULT_AGENT_MAX_THREADS)", font_size=22),
                     Text("depth 1: a sub-agent cannot spawn its own (DEFAULT_AGENT_MAX_DEPTH)", font_size=22)).arrange(DOWN, buff=0.35).move_to([0, -1.0, 0])
        self.at("six")
        self.play(FadeIn(lim[0]), run_time=0.3)
        self.at("level")
        self.play(FadeIn(lim[1]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=19)
        code.move_to([0, 0.1, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("pool")
        self.play(Create(hl), run_time=0.3)
        self.at("same")
        self.play(highlight(hl, code, 8), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 11
    def outro(self):
        self.section(11)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
