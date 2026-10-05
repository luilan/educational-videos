"""LLMs in Practice, episode 2 — Context Windows, and Why They Run Out.

Render from the repo root:  ./render.sh llms-in-practice p02
Every number on screen comes from code/p02_context_window/fit_the_window.py (Qwen2.5-0.5B-Instruct config and
tokenizer); the curve in section 9 is an illustrative shape.
"""
from manim import *

from common import MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from p02_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "LLMs IN PRACTICE  ·  EPISODE 2"
WINDOW, BOOK_TOKENS, KV_BYTES, KV_MIB = 32_768, 301_829, 12_288, 384
CHAT_TOKENS, BUDGET, KEPT_TOKENS = 133, 100, 82
WIN_COLOR = YELLOW
CODE = """def fit(messages, budget):
    system, turns = messages[:1], messages[1:]
    while count(system + turns) > budget:
        turns = turns[2:]          # drop the oldest exchange
    return system + turns

kept = fit(chat, budget=100)       # 133 tokens -> 82"""


def seg(width, color, label=None, height=0.7, font_size=22):
    box = Rectangle(width=width, height=height, stroke_width=1.5, stroke_color=BLACK, fill_color=color,
                    fill_opacity=0.85)
    if label is None:
        return VGroup(box)
    return VGroup(box, Text(label, font_size=font_size).move_to(box))


def msg_box(text, color, width=4.6):
    label = Text(text, font_size=20)
    box = RoundedRectangle(corner_radius=0.12, width=max(width, label.width + 0.4), height=0.55, stroke_color=color,
                           fill_color=color, fill_opacity=0.3)
    return VGroup(box, label.move_to(box))


class ContextWindowVideo(VoicedScene):
    VIDEO = "p02"

    def construct(self):
        play_token_intro(self, TITLE, 2, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.window()        # 2
        self.book()          # 3
        self.why_limit()     # 4
        self.memory()        # 5
        self.overflow()      # 6
        self.strategies()    # 7
        self.code()          # 8
        self.lessons()       # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        base = Line([-5, -2.6, 0], [5, -2.6, 0], color=GREY_C)
        heights = [0.8, 1.6, 2.3, 3.1, 3.8]
        bars = VGroup(*[Rectangle(width=1.0, height=h, stroke_width=0, fill_color=TOKEN_COLOR, fill_opacity=0.8)
                        for h in heights]).arrange(RIGHT, buff=0.6, aligned_edge=DOWN)
        bars.move_to([0, -2.6, 0], aligned_edge=DOWN)
        labels = VGroup(*[Text(f"turn {i}", font_size=20, color=GREY_B).next_to(b, DOWN, buff=0.15)
                          for i, b in enumerate(bars, 1)])
        self.at("resends")
        self.play(Create(base), run_time=0.4)
        self.at("growing")
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.2), FadeIn(labels), run_time=1.4)
        ceiling = DashedLine([-5, 1.6, 0], [5, 1.6, 0], color=WIN_COLOR, stroke_width=4)
        self.at("only")
        self.play(Create(ceiling), run_time=0.6)
        name = Text("context window", font_size=30, color=WIN_COLOR).next_to(ceiling, UP, buff=0.2)
        self.at("context")
        self.play(FadeIn(name, shift=0.2 * DOWN), bars[-1].animate.set_fill(RED), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 2. what the window is
    def window(self):
        self.section(2)
        self.clear_stage()
        frame = Rectangle(width=11, height=0.9, stroke_color=WIN_COLOR, stroke_width=4).move_to([0, 0.6, 0])
        prompt = seg(6.4, TOKEN_COLOR, "prompt", height=0.9).align_to(frame, LEFT).match_y(frame)
        reply = seg(2.6, GREEN_D, "reply", height=0.9).next_to(prompt, RIGHT, buff=0)
        cap = Text("maximum tokens in one request", font_size=28).next_to(frame, UP, buff=0.35)
        self.at("maximum")
        self.play(Create(frame), FadeIn(cap, shift=0.2 * DOWN), run_time=0.7)
        self.at("prompt")
        self.play(GrowFromEdge(prompt, LEFT), run_time=0.5)
        self.at("reply")
        self.play(GrowFromEdge(reply, LEFT), run_time=0.5)
        model = Text("Qwen2.5-0.5B-Instruct", font=MONO, font_size=24, color=GREY_A).next_to(frame, DOWN, buff=0.5)
        self.at("small")
        self.play(FadeIn(model, shift=0.2 * UP), run_time=0.5)
        size = Text(f"{WINDOW:,} tokens", font_size=56, color=WIN_COLOR).next_to(model, DOWN, buff=0.35)
        self.at("32")
        self.play(FadeIn(size, scale=1.3), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 3. how big is that
    def book(self):
        self.section(3)
        self.clear_stage()
        unit = 1.3
        win = Rectangle(width=unit, height=0.8, stroke_color=WIN_COLOR, stroke_width=4, fill_color=WIN_COLOR,
                        fill_opacity=0.2)
        n = BOOK_TOKENS / WINDOW
        strip = Rectangle(width=unit * n, height=0.8, stroke_width=0, fill_color=TEAL_D, fill_opacity=0.8)
        strip.scale_to_fit_width(12.6)
        u = strip.width / n
        win.stretch_to_fit_width(u)
        strip.move_to([0, -0.2, 0])
        win.align_to(strip, LEFT).match_y(strip).shift(1.1 * UP)
        win_label = Text("1 window", font_size=22, color=WIN_COLOR).next_to(win, UP, buff=0.15)
        ticks = VGroup(*[Line(UP * 0.4, DOWN * 0.4, color=BLACK, stroke_width=3).move_to(
            strip.get_left() + RIGHT * u * k) for k in range(1, 10)])
        self.at("lot")
        self.play(FadeIn(win), FadeIn(win_label), run_time=0.5)
        title = Text("Tiny Shakespeare", font_size=32).to_edge(UP, buff=0.8)
        self.at("shakespeare")
        self.play(FadeIn(title, shift=0.2 * DOWN), GrowFromEdge(strip, LEFT), run_time=1.2)
        count = Text(f"{BOOK_TOKENS:,} tokens", font_size=40, color=TEAL_B).next_to(strip, DOWN, buff=0.5)
        self.at("300")
        self.play(FadeIn(count, shift=0.2 * UP), run_time=0.5)
        nine = Text(f"= {n:.1f} windows", font_size=36, color=WIN_COLOR).next_to(count, DOWN, buff=0.3)
        self.at("nine")
        self.play(Create(ticks), FadeIn(nine, shift=0.2 * UP), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 4. why there is a limit
    def why_limit(self):
        self.section(4)
        self.clear_stage()
        head1 = Text("1 · position", font_size=30).move_to([-3.6, 2.7, 0])
        axis = NumberLine(x_range=[0, 4, 1], length=5.6, include_numbers=False, color=GREY_B).move_to([-3.6, 1.2, 0])
        trained = Line(axis.n2p(0), axis.n2p(2.8), color=GREEN, stroke_width=10)
        beyond = DashedLine(axis.n2p(2.8), axis.n2p(4), color=GREY_C, stroke_width=10)
        t_label = Text("trained lengths", font_size=20, color=GREEN).next_to(trained, UP, buff=0.2)
        q = Text("?", font_size=40, color=GREY_B).next_to(beyond, UP, buff=0.15)
        self.at("position")
        self.play(FadeIn(head1, shift=0.2 * DOWN), Create(axis), run_time=0.6)
        self.at("trained")
        self.play(Create(trained), FadeIn(t_label), run_time=0.6)
        self.at("beyond")
        self.play(Create(beyond), FadeIn(q, scale=1.4), run_time=0.6)

        head2 = Text("2 · attention", font_size=30).move_to([3.6, 2.7, 0])

        def tri(n, size):
            cell = size / n
            g = VGroup(*[Square(cell, stroke_width=1, stroke_color=BLACK, fill_color=BLUE_C, fill_opacity=0.85)
                         .move_to([j * cell, -i * cell, 0]) for i in range(n) for j in range(i + 1)])
            return g
        small = tri(4, 1.2)
        big = tri(8, 2.4)
        small.move_to([2.2, 0.6, 0], aligned_edge=UL).shift(0.6 * UP)
        big.move_to([4.0, 0.6, 0], aligned_edge=UL).shift(0.6 * UP)
        l_small = Text("n tokens", font_size=20, color=GREY_B).next_to(small, DOWN, buff=0.2)
        l_big = Text("2n tokens", font_size=20, color=GREY_B).next_to(big, DOWN, buff=0.2)
        self.at("attention")
        self.play(FadeIn(head2, shift=0.2 * DOWN), run_time=0.4)
        self.at("looks")
        self.play(LaggedStart(*[FadeIn(c) for c in small], lag_ratio=0.05), FadeIn(l_small), run_time=0.8)
        self.at("double")
        self.play(LaggedStart(*[FadeIn(c) for c in big], lag_ratio=0.01), FadeIn(l_big), run_time=1.0)
        four = Text("2× length → ~4× attention work", font_size=28, color=YELLOW).move_to([3.4, -2.4, 0])
        self.at("quadruples")
        self.play(FadeIn(four, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5. memory
    def memory(self):
        self.section(5)
        self.clear_stage()
        head = Text("3 · memory", font_size=32).to_edge(UP, buff=0.6)
        toks = VGroup(*[token(w, font_size=22) for w in ["The", "cat", "sat", "on", "the"]]).arrange(RIGHT, buff=0.5)
        toks.move_to([0, 1.2, 0])
        self.at("memory")
        self.play(FadeIn(head, shift=0.2 * DOWN), LaggedStart(*[FadeIn(t) for t in toks], lag_ratio=0.1), run_time=0.7)
        kv = VGroup()
        for t in toks:
            k = Rectangle(width=0.32, height=0.6, stroke_width=0, fill_color=GOLD, fill_opacity=0.85)
            v = Rectangle(width=0.32, height=0.6, stroke_width=0, fill_color=MAROON_C, fill_opacity=0.85)
            kv.add(VGroup(k, v).arrange(RIGHT, buff=0.06).next_to(t, DOWN, buff=0.3))
        legend = VGroup(Text("keys", font_size=22, color=GOLD), Text("values", font_size=22, color=MAROON_B))
        legend.arrange(RIGHT, buff=0.5).next_to(kv, DOWN, buff=0.35)
        self.at("keys")
        self.play(LaggedStart(*[GrowFromEdge(x, UP) for x in kv], lag_ratio=0.1), FadeIn(legend), run_time=0.8)
        cache = Text("the KV cache", font_size=28, color=GREY_A).next_to(legend, DOWN, buff=0.3)
        self.at("cache")
        self.play(FadeIn(cache), run_time=0.4)
        per = Text(f"{KV_BYTES:,} bytes ≈ 12 KB per token", font_size=32).move_to([0, -2.2, 0])
        self.at("12")
        self.play(FadeIn(per, shift=0.2 * UP), run_time=0.5)
        full = Text(f"× {WINDOW:,} tokens = {KV_MIB} MiB", font_size=36, color=YELLOW).next_to(per, DOWN, buff=0.3)
        self.at("384")
        self.play(FadeIn(full, shift=0.2 * UP), run_time=0.5)
        one = Text("for one conversation", font_size=22, color=GREY_B).next_to(full, DOWN, buff=0.2)
        self.at("single")
        self.play(FadeIn(one), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. overflow
    def overflow(self):
        self.section(6)
        self.clear_stage()
        frame = RoundedRectangle(corner_radius=0.2, width=5.6, height=4.6, stroke_color=WIN_COLOR, stroke_width=4)
        frame.move_to([-2.2, -0.3, 0])
        f_label = Text("context window", font_size=24, color=WIN_COLOR).next_to(frame, UP, buff=0.15)
        system = msg_box("system prompt", GREY_B)
        turns = VGroup(*[msg_box(f"turn {i}", BLUE_C if i % 2 else GREEN_C) for i in range(1, 7)])
        stack = VGroup(system, *turns).arrange(DOWN, buff=0.1).move_to(frame).align_to(frame, UP).shift(0.2 * DOWN)
        self.at("outgrows")
        self.play(Create(frame), FadeIn(f_label), FadeIn(system), LaggedStart(*[FadeIn(t) for t in turns],
                                                                              lag_ratio=0.1), run_time=0.9)
        new = msg_box("new message", RED).next_to(frame, DOWN, buff=0.25)
        self.at("go")
        self.play(FadeIn(new, shift=0.3 * UP), Indicate(frame, color=RED), run_time=0.6)
        pin = Text("kept", font_size=22, color=GREEN).next_to(frame, LEFT, buff=0.2).match_y(system)
        self.at("system")
        self.play(system[0].animate.set_stroke(GREEN, 4), FadeIn(pin), run_time=0.5)
        dropped = VGroup(turns[0], turns[1])
        self.at("drop")
        self.play(dropped.animate.move_to([3.6, 1.3, 0]).set_opacity(0.3), run_time=0.7)
        out = Text("dropped", font_size=24, color=RED).next_to(dropped, UP, buff=0.2)
        self.play(FadeIn(out), run_time=0.3)
        rest = VGroup(*turns[2:], new)
        target = VGroup(*[m.copy() for m in rest]).arrange(DOWN, buff=0.1).next_to(system, DOWN, buff=0.1)
        self.play(*[m.animate.move_to(t) for m, t in zip(rest, target)], run_time=0.6)
        forgets = Text("the model forgets\nhow it started", font_size=28, color=YELLOW).move_to([3.6, -1.2, 0])
        self.at("forgets")
        self.play(FadeIn(forgets, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 7. other strategies
    def strategies(self):
        self.section(7)
        self.clear_stage()
        old = VGroup(*[msg_box(f"turn {i}", BLUE_C if i % 2 else GREEN_C, width=2.4) for i in range(1, 5)])
        old.arrange(DOWN, buff=0.1).move_to([-4.6, 0.4, 0])
        self.at("summarize")
        self.play(FadeIn(old), run_time=0.5)
        note = msg_box("summary: eggs, 7 & 10 min", GOLD, width=3.6).move_to([0.3, 1.6, 0])
        self.at("note")
        self.play(TransformFromCopy(old, note), run_time=0.8)
        db = VGroup(Ellipse(width=1.6, height=0.5, color=TEAL_C), Rectangle(width=1.6, height=1.2, color=TEAL_C),
                    Ellipse(width=1.6, height=0.5, color=TEAL_C))
        db[1].next_to(db[0], DOWN, buff=-0.25)
        db[2].move_to(db[1].get_bottom())
        db.move_to([0.3, -1.3, 0])
        db_label = Text("stored turns", font_size=20, color=TEAL_B).next_to(db, DOWN, buff=0.15)
        self.at("store")
        self.play(FadeIn(db), FadeIn(db_label), run_time=0.5)
        piece = msg_box("turn 2", GREEN_C, width=2.0).move_to(db)
        self.at("bring")
        self.play(piece.animate.move_to([4.6, -1.3, 0]), run_time=0.7)
        ret = Text("retrieval · episode 5", font_size=30, color=YELLOW).move_to([3.6, 2.8, 0])
        self.at("retrieval")
        self.play(FadeIn(ret, shift=0.2 * DOWN), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.6, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.at("count")
        hl.match_y(code.line_numbers[2])
        self.play(Create(hl), run_time=0.4)
        self.at("drop")
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.at("system")
        self.play(highlight(hl, code, 1), run_time=0.4)
        result = Text(f"{CHAT_TOKENS} tokens  →  {KEPT_TOKENS} tokens  (budget {BUDGET})", font_size=30, color=YELLOW)
        result.next_to(code, DOWN, buff=0.4)
        self.at("133")
        self.play(highlight(hl, code, 6), FadeIn(result, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 9. lessons
    def lessons(self):
        self.section(9)
        self.clear_stage()
        costs = VGroup(Text("longer requests:", font_size=28), Text("slower · cost more", font_size=28, color=RED_B))
        costs.arrange(RIGHT, buff=0.3).move_to([0, 2.7, 0])
        self.at("free")
        self.play(FadeIn(costs[0]), run_time=0.4)
        self.at("slower")
        self.play(FadeIn(costs[1], shift=0.2 * LEFT), run_time=0.5)
        ax = Axes(x_range=[0, 1, 0.5], y_range=[0, 1, 0.5], x_length=6.5, y_length=2.8, tips=False,
                  axis_config={"color": GREY_B, "include_ticks": False}).move_to([0, 0, 0])
        curve = ax.plot(lambda x: 0.35 + 2.2 * (x - 0.5) ** 2, x_range=[0, 1], color=YELLOW)
        xl = VGroup(*[Text(s, font_size=20, color=GREY_B).next_to(ax.c2p(x, 0), DOWN, buff=0.2)
                      for s, x in [("start", 0.05), ("middle", 0.5), ("end", 0.95)]])
        yl = Text("how well it is used", font_size=20, color=GREY_B).rotate(PI / 2).next_to(ax, LEFT, buff=0.2)
        illus = Text("illustrative shape", font_size=18, color=GREY_B).next_to(ax, RIGHT, buff=0.3)
        self.at("models")
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(illus), run_time=0.6)
        self.at("start")
        self.play(Create(curve), run_time=1.2)
        dip = Dot(ax.c2p(0.5, 0.35), color=RED, radius=0.1)
        self.at("middle")
        self.play(FadeIn(dip, scale=2), run_time=0.4)
        tips = VGroup(Text("keep the context lean", font_size=28),
                      Text("put what matters where it's noticed", font_size=28)).arrange(DOWN, buff=0.25)
        tips.move_to([0, -2.6, 0]).set_color(GREEN_B)
        self.at("lean")
        self.play(FadeIn(tips[0], shift=0.2 * UP), run_time=0.4)
        self.at("matters")
        self.play(FadeIn(tips[1], shift=0.2 * UP), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. outro
    def outro(self):
        self.section(10)
        self.at("context")
        self.clear_stage(run_time=0.5)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
