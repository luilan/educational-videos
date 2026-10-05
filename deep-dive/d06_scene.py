"""How LLMs Work: Deep Dive, episode 6 — Long Context: Stretching RoPE.

Render from the repo root:  ./render.sh deep-dive d06
Every loss on screen comes from code/d06_long_context/long_context.py (an 810,049-parameter RoPE GPT trained on Tiny
Shakespeare with a 64-token context, evaluated on 256 tokens; curves are averages over 8 positions).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d06_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 6"
CURVES = {
    "plain": [1.86, 1.55, 1.59, 1.56, 1.54, 1.56, 1.54, 1.57, 1.57, 1.61, 1.67, 1.78, 2.07, 2.78, 2.81, 2.91, 2.98, 3.01,
              3.03, 3.0, 3.05, 3.07, 3.16, 3.28, 3.24, 3.38, 3.42, 3.45, 3.42, 3.39, 3.37, 3.43],
    "pi": [2.72, 3.37, 3.55, 3.57, 3.55, 3.52, 3.56, 3.63, 3.58, 3.61, 3.57, 3.6, 3.57, 3.61, 3.55, 3.55, 3.62, 3.6, 3.61,
           3.55, 3.61, 3.58, 3.62, 3.61, 3.6, 3.62, 3.59, 3.61, 3.56, 3.61, 3.57, 3.55],
    "ntk": [1.86, 1.57, 1.62, 1.58, 1.57, 1.61, 1.6, 1.62, 1.62, 1.62, 1.58, 1.63, 1.63, 1.64, 1.62, 1.65, 1.73, 1.87,
            2.15, 2.32, 2.31, 2.43, 2.52, 2.69, 2.68, 2.76, 2.73, 2.77, 2.74, 2.8, 2.8, 2.86],
    "pi_ft": [1.89, 1.59, 1.62, 1.58, 1.56, 1.58, 1.56, 1.6, 1.58, 1.57, 1.54, 1.58, 1.56, 1.57, 1.55, 1.57, 1.58, 1.62,
              1.57, 1.59, 1.56, 1.57, 1.56, 1.58, 1.58, 1.58, 1.58, 1.62, 1.58, 1.61, 1.58, 1.6],
}
COLORS = {"plain": RED_C, "pi": PURPLE_B, "ntk": GOLD, "pi_ft": GREEN_C}
LABELS = {"plain": "no change", "pi": "interpolation only", "ntk": "NTK base scaling", "pi_ft": "interp. + 200 steps"}
CODE = """class Rope:
    def __init__(self, base=10_000.0, scale=1.0):
        self.freqs = base ** (-torch.arange(0, d, 2) / d)
        self.scale = scale

    def __call__(self, x):
        angle = (positions * self.scale)[:, None] * self.freqs     # ... then rotate

pi = Rope(scale=64 / 256)                                   # position interpolation
ntk = Rope(base=10_000 * (256 / 64) ** (d / (d - 2)))       # NTK-aware scaling"""


class LongContextVideo(VoicedScene):
    VIDEO = "d06"

    def construct(self):
        play_token_intro(self, TITLE, 6, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.setup_exp()     # 2
        self.plain()         # 3
        self.pi()            # 4
        self.ntk()           # 5
        self.finetune()      # 6
        self.real()          # 7
        self.caveats()       # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def curve(self, key, upto=32):
        pts = [self.ax.c2p(4 + 8 * i, v) for i, v in enumerate(CURVES[key][:upto])]
        return VMobject(color=COLORS[key], stroke_width=5).set_points_smoothly(pts)

    def legend_item(self, key):
        item = VGroup(Line(ORIGIN, 0.5 * RIGHT, color=COLORS[key], stroke_width=5),
                      Text(LABELS[key], font_size=20, color=COLORS[key])).arrange(RIGHT, buff=0.15)
        k = list(CURVES).index(key)
        return item.move_to([3.2, 1.6 - 0.45 * k, 0], aligned_edge=LEFT)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        bar = Rectangle(width=6.0, height=0.6, stroke_width=0, fill_color=GREEN_C, fill_opacity=0.8).move_to([-1.5, 1.0, 0])
        bl = Text("trained length: 32,768 tokens (Qwen2.5)", font_size=24).next_to(bar, UP, buff=0.15)
        self.at("32")
        self.play(GrowFromEdge(bar, LEFT), FadeIn(bl), run_time=0.6)
        more = DashedLine(bar.get_right(), bar.get_right() + 4.5 * RIGHT, color=YELLOW, stroke_width=8)
        ml = Text("used on longer texts?", font_size=24, color=YELLOW).next_to(more, DOWN, buff=0.2)
        self.at("longer")
        self.play(Create(more), FadeIn(ml), run_time=0.6)
        st = Text("stretch RoPE's rotations", font_size=32, color=YELLOW).move_to([0, -1.0, 0])
        self.at("rope")
        self.play(FadeIn(st), run_time=0.4)
        lap = Text("tested at a scale you can run on a laptop", font_size=24, color=GREY_B).move_to([0, -2.0, 0])
        self.at("laptop")
        self.play(FadeIn(lap), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. setup
    def setup_exp(self):
        self.section(2)
        self.clear_stage()
        head = Text("a tiny RoPE GPT: 810,049 parameters, no position table", font_size=28).to_edge(UP, buff=0.35)
        self.at("tiny")
        self.play(FadeIn(head), run_time=0.4)
        self.ax = Axes(x_range=[0, 256, 64], y_range=[1.4, 4.4, 0.5], x_length=8.0, y_length=4.6, tips=False,
                       axis_config={"color": GREY_B}).move_to([-1.3, -0.1, 0])
        xl = VGroup(*[Text(str(v), font_size=18, color=GREY_B).next_to(self.ax.c2p(v, 1.4), DOWN, buff=0.1)
                      for v in (0, 64, 128, 192, 256)])
        yl = VGroup(*[Text(f"{v:.1f}", font_size=18, color=GREY_B).next_to(self.ax.c2p(0, v), LEFT, buff=0.1)
                      for v in (1.5, 2.0, 2.5, 3.0, 3.5)])
        xt = Text("position in the text", font_size=20, color=GREY_B).next_to(xl, DOWN, buff=0.1)
        yt = Text("loss", font_size=20, color=GREY_B).next_to(yl, LEFT, buff=0.15)
        trained = Rectangle(width=self.ax.c2p(64, 0)[0] - self.ax.c2p(0, 0)[0], height=4.6, stroke_width=0,
                            fill_color=GREEN, fill_opacity=0.12).move_to(self.ax.c2p(32, 2.9))
        tl = Text("trained:\n64 tokens", font_size=18, color=GREEN_B, line_spacing=0.8).move_to(self.ax.c2p(32, 4.1))
        self.at("64")
        self.play(Create(self.ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt), FadeIn(trained), FadeIn(tl),
                  run_time=0.8)
        long_l = Text("now read 256 tokens", font_size=22, color=YELLOW).move_to(self.ax.c2p(160, 4.15))
        self.at("256")
        self.play(FadeIn(long_l), run_time=0.4)
        self.at("position")
        self.play(Indicate(xt, color=YELLOW), run_time=0.5)
        self.chart = VGroup(self.ax, xl, yl, xt, yt, trained, tl, long_l)
        self.end_section()

    # ------------------------------------------------------------------ 3. no change
    def plain(self):
        self.section(3)
        self.at("change")
        c1 = self.curve("plain", 8)
        self.play(Create(c1), FadeIn(self.legend_item("plain")), run_time=0.8)
        self.at("climbs")
        c2 = self.curve("plain")
        self.play(Create(c2), FadeOut(c1), run_time=1.6)
        mark = DashedLine(self.ax.c2p(96, 1.4), self.ax.c2p(96, 3.9), color=RED_B)
        ml = Text("falls apart\npast ~96", font_size=18, color=RED_B, line_spacing=0.8).next_to(self.ax.c2p(112, 1.8), RIGHT, 0.1)
        self.at("96")
        self.play(Create(mark), FadeIn(ml), run_time=0.5)
        self.marks = VGroup(mark, ml)
        self.end_section()

    # ------------------------------------------------------------------ 4. position interpolation
    def pi(self):
        self.section(4)
        self.play(FadeOut(self.marks), run_time=0.3)
        idea = Text("idea 1: squeeze positions by 4 (position interpolation)", font_size=22, color=PURPLE_B)
        idea.move_to([0, -3.5, 0])
        self.at("interpolation")
        self.play(FadeIn(idea), run_time=0.4)
        self.at("bad")
        self.play(Create(self.curve("pi")), FadeIn(self.legend_item("pi")), run_time=1.2)
        why = Text("neighbours ¼ step apart:\nfast pairs can't\ntell them apart", font_size=18, color=PURPLE_B,
                   line_spacing=0.8).move_to([3.2, -0.6, 0], aligned_edge=LEFT)
        self.at("quarter")
        self.play(FadeIn(why), run_time=0.4)
        self.why, self.idea = why, idea
        self.end_section()

    # ------------------------------------------------------------------ 5. NTK scaling
    def ntk(self):
        self.section(5)
        idea = Text("idea 2: raise RoPE's base (NTK-aware): slow pairs stretch, fast pairs hardly change",
                    font_size=20, color=GOLD).move_to([0, -3.5, 0])
        self.at("ntk")
        self.play(FadeOut(self.why), Transform(self.idea, idea), run_time=0.5)
        self.at("training")
        self.play(Create(self.curve("ntk")), FadeIn(self.legend_item("ntk")), run_time=1.4)
        mark = DashedLine(self.ax.c2p(128, 1.4), self.ax.c2p(128, 3.9), color=GOLD)
        ml = Text("fine to 2×", font_size=18, color=GOLD).next_to(self.ax.c2p(128, 1.6), RIGHT, 0.1)
        self.at("127")
        self.play(Create(mark), FadeIn(ml), run_time=0.5)
        self.marks = VGroup(mark, ml)
        self.end_section()

    # ------------------------------------------------------------------ 6. fine-tune
    def finetune(self):
        self.section(6)
        idea = Text("idea 3: interpolate, then train 200 steps on 256-token texts", font_size=22, color=GREEN_C)
        idea.move_to([0, -3.5, 0])
        self.at("interpolate")
        self.play(FadeOut(self.marks), Transform(self.idea, idea), run_time=0.5)
        self.at("200")
        self.play(Create(self.curve("pi_ft")), FadeIn(self.legend_item("pi_ft")), run_time=1.4)
        flat = Text("flat: 1.62 · 1.56 · 1.58", font=MONO, font_size=20, color=GREEN_B).move_to(self.ax.c2p(212, 1.9))
        self.at("flat")
        self.play(FadeIn(flat), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. real models
    def real(self):
        self.section(7)
        self.clear_stage()
        head = Text("the same recipe, at scale", font_size=32).to_edge(UP, buff=0.6)
        self.at("real")
        self.play(FadeIn(head), run_time=0.4)
        a = Rectangle(width=2.5, height=0.6, stroke_width=0, fill_color=GREEN_C, fill_opacity=0.8)
        b = Rectangle(width=10.0, height=0.6, stroke_width=0, fill_color=GOLD, fill_opacity=0.5)
        VGroup(a, b).arrange(DOWN, buff=0.6, aligned_edge=LEFT).move_to([0, 0.4, 0])
        al = Text("trained: 32,768", font=MONO, font_size=22).next_to(a, RIGHT, buff=0.2)
        bl = Text("YaRN-stretched: 131,072 (Qwen2.5 documentation)", font=MONO, font_size=22).move_to(b)
        self.at("yarn")
        self.play(FadeIn(a), FadeIn(al), run_time=0.4)
        self.at("131")
        self.play(GrowFromEdge(b, LEFT), FadeIn(bl), run_time=0.7)
        fam = Text("other families: their own variants, usually with some long-text training", font_size=24,
                   color=GREY_A).move_to([0, -1.6, 0])
        self.at("families")
        self.play(FadeIn(fam), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. caveats
    def caveats(self):
        self.section(8)
        self.clear_stage()
        items = VGroup(Text("a longer window isn't free", font_size=32, color=YELLOW),
                       Text("✗ the KV cache grows with every token (LLMs in Practice, ep. 2)", font_size=26, color=RED_B),
                       Text("✗ the middle of a long text is used less well", font_size=26, color=RED_B),
                       Text("stretching the window ≠ using it well", font_size=28, color=GREY_A))
        items.arrange(DOWN, buff=0.45).move_to([0, 0.2, 0])
        for k, cue in enumerate(["free", "cache", "middle", "using"]):
            self.at(cue)
            self.play(FadeIn(items[k], shift=0.2 * UP), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[8])
        self.at("interpolation")
        self.play(Create(hl), run_time=0.3)
        self.at("base")
        self.play(highlight(hl, code, 9), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. outro
    def outro(self):
        self.section(10)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("attention")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
