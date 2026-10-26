"""How AI Agents Work: Inside the Harness, episode 2 — The Agent Loop.

Render from the repo root:  ./render.sh agents a02
Numbers: code/a02_the_agent_loop/agent_loop.py run on 2026-10-21 (Qwen3 4B, Ollama, CPU) and the Codex CLI 0.163.0
trace from episode 1 (per-call input tokens reported by the server; 26,171 of 30,292 input tokens cached).
Codex code: openai/codex @ 89461ea (codex-rs/core/src/stream_events_utils.rs, session/turn.rs), simplified.
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from a02_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 2"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
OURS = [197, 322, 388, 449, 492]                 # tokens sent per model call, tiny harness
CODEX = [3779, 4881, 6191, 7551, 7890]           # tokens sent per model call, Codex
STEPS = [("$ python3 test_stats.py", "exit 1 · AssertionError", RED_B), ("$ cat stats.py", "exit 0", GREY_A),
         ("$ echo 'def mean(xs): ...' > stats.py", "exit 0", GREY_A), ("$ python3 test_stats.py", "exit 0 · passes", GREEN_B),
         ("answers in words", "done", GREY_A)]
CODE = """for step in range(8):
    reply = model(history, tools)
    calls = tool_calls(reply)
    if not calls:
        return "answered"
    for call in calls:
        history.append(run(call, timeout=30))
    if same_command_3_times(history):
        return "stuck"
return "step limit\""""


def box(label, color, w=2.4, h=0.9, fs=22):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.15, color=color, fill_opacity=0.15)
    return VGroup(r, Text(label, font_size=fs, color=color).move_to(r))


def bars(values, color, x0, y0, width, height, title, fmt=lambda v: f"{v:,}"):
    """Small single-series bar chart: one bar per model call, value on top, call number below."""
    g = VGroup()
    top = max(values)
    bw = width / len(values)
    base = Line([x0, y0, 0], [x0 + width, y0, 0], color=GREY_D, stroke_width=1)
    g.add(base, Text(title, font_size=18, color=GREY_A).move_to([x0 + width / 2, y0 + height + 0.55, 0]))
    for i, v in enumerate(values):
        h = height * v / top
        r = RoundedRectangle(width=bw * 0.62, height=h, corner_radius=min(0.06, h / 2), stroke_width=0, fill_color=color,
                             fill_opacity=0.9).move_to([x0 + bw * (i + 0.5), y0 + h / 2, 0])
        g.add(r, Text(fmt(v), font=MONO, font_size=14).next_to(r, UP, buff=0.08),
              Text(str(i + 1), font_size=14, color=GREY_B).next_to([x0 + bw * (i + 0.5), y0, 0], DOWN, buff=0.1))
    g.add(Text("model call", font_size=14, color=GREY_B).move_to([x0 + width / 2, y0 - 0.55, 0]))
    return g


class AgentLoopVideo(VoicedScene):
    VIDEO = "a02"

    def construct(self):
        play_token_intro(self, TITLE, 2, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.step()        # 2
        self.run()         # 3
        self.grows()       # 4
        self.codex_stop()  # 5
        self.never()       # 6
        self.guards()      # 7
        self.code()        # 8
        self.outro()       # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        nodes = [box("model", MODEL_C), box("tool calls", TOOL_C), box("results", HARNESS_C)]
        for n, a in zip(nodes, (90, -30, 210)):
            n.move_to(2.5 * np.array([np.cos(a * DEGREES), np.sin(a * DEGREES), 0]) + 0.1 * DOWN)
        arrows = VGroup(*[CurvedArrow(nodes[i].get_center(), nodes[(i + 1) % 3].get_center(), angle=-TAU / 6, color=GREY_B)
                          for i in range(3)])
        for a in arrows:
            a.scale(0.42)
        self.at("loop")
        self.play(FadeIn(VGroup(*nodes)), Create(arrows), run_time=1.0)
        q = Text("what happens each time around? what makes it stop?", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.6)
        self.at("interesting")
        self.play(FadeIn(q), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. one step
    def step(self):
        self.section(2)
        self.clear_stage()
        conv = box("conversation", GREY_B, w=2.6).move_to([-4.6, 1.2, 0])
        model = box("model", MODEL_C).move_to([-0.9, 1.2, 0])
        dec = box("tool calls?", TOOL_C, w=2.4).move_to([2.9, 1.2, 0])
        run = box("run them", HARNESS_C).move_to([2.9, -1.0, 0])
        add = box("add results", HARNESS_C, w=2.6).move_to([-0.9, -1.0, 0])
        over = box("turn over", GREEN_B, w=2.2).move_to([5.6, 1.2, 0]).shift(DOWN * 0.0)
        a1 = Arrow(conv.get_right(), model.get_left(), buff=0.1, color=GREY_B)
        a2 = Arrow(model.get_right(), dec.get_left(), buff=0.1, color=GREY_B)
        yes = Arrow(dec.get_bottom(), run.get_top(), buff=0.1, color=TOOL_C)
        yl = Text("yes", font_size=18, color=TOOL_C).next_to(yes, RIGHT, buff=0.1)
        a3 = Arrow(run.get_left(), add.get_right(), buff=0.1, color=GREY_B)
        back = CurvedArrow(add.get_left() + 0.05 * UP, conv.get_bottom(), angle=-PI / 3, color=GREY_B)
        no = Arrow(dec.get_top() + 0.0 * UP, over.get_top(), buff=0.1, color=GREEN_B, path_arc=-PI / 2)
        nl = Text("no", font_size=18, color=GREEN_B).next_to(no, UP, buff=0.05)
        self.at("send")
        self.play(FadeIn(conv), GrowArrow(a1), FadeIn(model), run_time=0.6)
        self.at("contains")
        self.play(GrowArrow(a2), FadeIn(dec), run_time=0.5)
        self.at("run")
        self.play(GrowArrow(yes), FadeIn(yl), FadeIn(run), run_time=0.5)
        self.at("add")
        self.play(GrowArrow(a3), FadeIn(add), run_time=0.5)
        self.at("again")
        self.play(Create(back), run_time=0.6)
        self.at("doesnt")
        self.play(Create(no), FadeIn(nl), FadeIn(over), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 3. a real run
    def run(self):
        self.section(3)
        self.clear_stage()
        h = Text("agent_loop.py · Qwen3 4B · fix the failing test", font_size=24).to_edge(UP, buff=0.6)
        self.at("real")
        self.play(FadeIn(h), run_time=0.4)
        rows = VGroup()
        for i, (cmd, res, col) in enumerate(STEPS, 1):
            rows.add(VGroup(Text(f"call {i}", font_size=18, color=MODEL_C), Text(cmd, font=MONO, font_size=20),
                            Text(res, font=MONO, font_size=19, color=col)).arrange(RIGHT, buff=0.4))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([0, 0.2, 0])
        for row, cue in zip(rows, ("fails", "readstats", "rewrite", "passes", "answer")):
            self.at(cue)
            self.play(FadeIn(row, shift=0.1 * RIGHT), run_time=0.35)
        n = Text("read the file first · checked its own work ✓", font_size=24, color=GREEN_B).to_edge(DOWN, buff=0.7)
        self.at("first")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. it grows
    def grows(self):
        self.section(4)
        self.clear_stage()
        h = Text("every call resends the whole conversation", font_size=26).to_edge(UP, buff=0.5)
        self.at("whole")
        self.play(FadeIn(h), run_time=0.4)
        ours = bars(OURS, HARNESS_C, -6.0, -1.6, 5.0, 2.6, "our harness: tokens sent")
        codex = bars(CODEX, BLUE_C, 0.9, -1.6, 5.0, 2.6, "Codex: tokens sent")
        self.at("197")
        self.play(FadeIn(ours, lag_ratio=0.05), run_time=1.0)
        self.at("codecs")
        self.play(FadeIn(codex, lag_ratio=0.05), run_time=1.0)
        c = Text("Codex: 26,171 of 30,292 tokens (86%) served from the server's cache", font_size=20, color=YELLOW).to_edge(DOWN, buff=0.35)
        self.at("86")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. how Codex stops
    def codex_stop(self):
        self.section(5)
        self.clear_stage()
        h = Text("Codex CLI · how a turn ends", font_size=26).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        c1 = VGroup(Text("stream_events_utils.rs · when the reply contains a tool call", font_size=16, color=GREY_B),
                    Text("output.needs_follow_up = true;", font=MONO, font_size=22, color=TOOL_C)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        c2 = VGroup(Text("session/turn.rs · after each model call", font_size=16, color=GREY_B),
                    Text("if !needs_follow_up {\n    run_turn_stop_hooks(...);  // then the turn ends\n}", font=MONO, font_size=20,
                         line_spacing=0.9)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        VGroup(c1, c2).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to([0, 0.6, 0])
        self.at("flag")
        self.play(FadeIn(c1), run_time=0.5)
        self.at("false")
        self.play(FadeIn(c2), run_time=0.5)
        ends = Text("also ends a turn: user interrupt · error · token or time budget", font_size=20, color=GREY_A).move_to([0, -1.9, 0])
        self.at("interrupt")
        self.play(FadeIn(ends), run_time=0.4)
        lim = Text("no fixed step limit", font_size=26, color=YELLOW).move_to([0, -2.8, 0])
        self.at("limit")
        self.play(FadeIn(lim), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. a task that never ends
    def never(self):
        self.section(6)
        self.clear_stage()
        t = Text("task: “wait until ready.txt exists, then print it” (it never will)", font_size=22).to_edge(UP, buff=0.6)
        self.at("wait")
        self.play(FadeIn(t), run_time=0.4)
        cmd = VGroup(Text("call 1", font_size=18, color=MODEL_C),
                     Text("$ while [ ! -f ready.txt ]; do sleep 1; done; cat ready.txt", font=MONO, font_size=17)).arrange(RIGHT, buff=0.4)
        cmd.move_to([0, 1.2, 0])
        self.at("forever")
        self.play(FadeIn(cmd), run_time=0.4)
        k = Text("harness: killed after 30 s (timed out)", font=MONO, font_size=20, color=RED_B).move_to([0, 0.3, 0])
        self.at("killed")
        self.play(FadeIn(k), run_time=0.4)
        g = VGroup(Text("call 2", font_size=18, color=MODEL_C), Text("model gives up and explains: no tool call → loop ends", font_size=20)).arrange(RIGHT, buff=0.4).move_to([0, -0.6, 0])
        self.at("gave")
        self.play(FadeIn(g), run_time=0.4)
        n = Text("the safety net was the timeout, not the model", font_size=26, color=YELLOW).to_edge(DOWN, buff=0.8)
        self.at("safety")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. our safeguards
    def guards(self):
        self.section(7)
        self.clear_stage()
        h = Text("three guards in our tiny harness", font_size=26).to_edge(UP, buff=0.6)
        self.play(FadeIn(h), run_time=0.3)
        rows = [("timeout on every command", "30 s", "triggered", RED_B), ("limit on model calls", "8", "not reached", GREY_B),
                ("stuck detector", "same command 3× in a row", "not triggered", GREY_B)]
        g = VGroup()
        for name, val, st, col in rows:
            g.add(VGroup(Text(name, font_size=22), Text(val, font=MONO, font_size=20, color=GOLD), Text(st, font_size=20, color=col)))
        for r in g:
            r[0].move_to([-5.6, 0, 0], aligned_edge=LEFT); r[1].move_to([0.9, 0, 0]); r[2].move_to([5.0, 0, 0])
        for i, r in enumerate(g):
            r.shift((1.0 - i * 1.0) * UP)
        for r, cue in zip(g, ("timeout", "eight", "stuck")):
            self.at(cue)
            self.play(FadeIn(r[:2]), run_time=0.4)
        self.at("only")
        self.play(*[FadeIn(r[2]) for r in g], run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[6])
        self.at("guard")
        self.play(Create(hl), run_time=0.3)
        self.at("exit")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
