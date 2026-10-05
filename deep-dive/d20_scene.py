"""How LLMs Work: Deep Dive, episode 20 — Learning-Rate Warmup and Schedules.

Render from the repo root:  ./render.sh deep-dive d20
Every number on screen comes from code/d20_lr_schedules/lr_schedules.py (a 4-layer tiny GPT trained 3,000 steps with
AdamW under five learning-rate schedules; peak 0.003, 200 warmup steps, decays ending at 10% of the peak).
"""
import math

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d20_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 20"
STEPS = [500, 1000, 1500, 2000, 2500, 3000]
CURVES = {
    "constant 0.001": ([2.000, 1.810, 1.709, 1.644, 1.630, 1.605], GREY_B),
    "constant 0.003": ([1.899, 1.731, 1.661, 1.620, 1.603, 1.598], BLUE_C),
    "warmup + cosine": ([1.915, 1.733, 1.640, 1.588, 1.564, 1.551], GREEN_C),
    "warmup + linear": ([1.912, 1.728, 1.643, 1.600, 1.579, 1.554], TEAL_C),
    "warmup-stable-decay": ([1.917, 1.740, 1.670, 1.623, 1.606, 1.542], GOLD),
}
TOTAL, WARM = 3000, 200


def warm(s):
    return min(1.0, (s + 1) / WARM)


LR = {
    "constant 0.001": lambda s: 1 / 3,
    "constant 0.003": lambda s: 1.0,
    "warmup + cosine": lambda s: warm(s) * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1, max(0, s - WARM) / (TOTAL - WARM))))),
    "warmup + linear": lambda s: warm(s) * (1 - 0.9 * max(0, s - WARM) / (TOTAL - WARM)),
    "warmup-stable-decay": lambda s: warm(s) * (1.0 if s < 2400 else 1 - 0.9 * (s - 2400) / 600),
}
CODE = """def wsd(step):                          # warmup, stable, decay
    if step < 200:
        return (step + 1) / 200             # warmup
    if step < 2400:
        return 1.0                          # stable at the peak
    return 1 - 0.9 * (step - 2400) / 600    # decay to 10%

sched = torch.optim.lr_scheduler.LambdaLR(opt, wsd)   # lr = peak × wsd(step)"""


class SchedulesVideo(VoicedScene):
    VIDEO = "d20"

    def construct(self):
        play_token_intro(self, TITLE, 20, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.setup_runs()    # 2
        self.warmup()        # 3
        self.constant()      # 4
        self.decay()         # 5
        self.wsd()           # 6
        self.practical()     # 7
        self.lesson()        # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def lr_axes(self, w=8.0, h=3.6, pos=(0, -0.4)):
        ax = Axes(x_range=[0, 3000, 1000], y_range=[0, 1.1, 0.5], x_length=w, y_length=h, tips=False,
                  axis_config={"color": GREY_B}).move_to([pos[0], pos[1], 0])
        xl = VGroup(*[Text(f"{v:,}", font_size=16, color=GREY_B).next_to(ax.c2p(v, 0), DOWN, buff=0.1)
                      for v in (0, 1000, 2000, 3000)])
        yl = Text("lr / peak", font_size=16, color=GREY_B).next_to(ax.c2p(0, 1.0), LEFT, buff=0.1)
        return ax, VGroup(xl, yl)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        ax, labels = self.lr_axes(pos=(0, -0.6))
        self.at("learning")
        self.play(Create(ax), FadeIn(labels), run_time=0.5)
        c = ax.plot(LR["warmup + cosine"], x_range=[0, 2999, 5], color=GREEN_C, stroke_width=5)
        r = Text("rises", font_size=24, color=YELLOW).move_to(ax.c2p(250, 1.08) + 0.3 * UP)
        f = Text("falls", font_size=24, color=YELLOW).move_to(ax.c2p(2600, 0.5) + 0.3 * UP)
        self.at("rises")
        self.play(Create(c), FadeIn(r), run_time=1.0)
        self.at("falls")
        self.play(FadeIn(f), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. setup
    def setup_runs(self):
        self.section(2)
        self.clear_stage()
        head = Text("the same tiny GPT, 3,000 steps, five learning-rate schedules", font_size=26).to_edge(UP, buff=0.4)
        self.at("five")
        self.play(FadeIn(head), run_time=0.4)
        ax, labels = self.lr_axes(w=7.0, h=3.6, pos=(-1.8, -0.5))
        self.play(Create(ax), FadeIn(labels), run_time=0.4)
        legend = VGroup()
        cues = ("constant", "three", "cosine", "linear", "stable")
        for k, (name, (vals, col)) in enumerate(CURVES.items()):
            curve = ax.plot(LR[name], x_range=[0, 2999, 5], color=col, stroke_width=4, use_smoothing=False)
            item = VGroup(Line(ORIGIN, 0.4 * RIGHT, color=col, stroke_width=5), Text(name, font_size=18, color=col))
            item.arrange(RIGHT, buff=0.15).move_to([4.6, 1.4 - 0.5 * k, 0], aligned_edge=LEFT).shift(1.4 * LEFT)
            self.at(cues[k])
            self.play(Create(curve), FadeIn(item), run_time=0.5)
            legend.add(item)
        self.end_section()

    # ------------------------------------------------------------------ 3. warmup
    def warmup(self):
        self.section(3)
        self.clear_stage()
        ax = Axes(x_range=[0, 400, 100], y_range=[0, 1.1, 0.5], x_length=7, y_length=3.4, tips=False,
                  axis_config={"color": GREY_B}).move_to([0, -0.4, 0])
        xl = VGroup(*[Text(str(v), font_size=16, color=GREY_B).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v in (0, 200, 400)])
        ramp = ax.plot(lambda s: warm(s), x_range=[0, 400, 2], color=YELLOW, stroke_width=5, use_smoothing=False)
        t = Text("warmup: 200 steps from near zero to the peak", font_size=26).to_edge(UP, buff=0.6)
        self.at("warm", "warmup")
        self.play(FadeIn(t), Create(ax), FadeIn(xl), Create(ramp), run_time=0.8)
        n = Text("episode 16: the first updates are the riskiest", font_size=24, color=GREY_A).move_to([0, -2.8, 0])
        self.at("riskiest")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ loss chart helpers
    def loss_axes(self):
        ax = Axes(x_range=[0, 3000, 500], y_range=[1.5, 2.05, 0.1], x_length=7.4, y_length=4.4, tips=False,
                  axis_config={"color": GREY_B}).move_to([-1.9, -0.4, 0])
        xl = VGroup(*[Text(f"{v:,}", font_size=15, color=GREY_B).next_to(ax.c2p(v, 1.5), DOWN, buff=0.1)
                      for v in (500, 1000, 1500, 2000, 2500, 3000)])
        yl = VGroup(*[Text(f"{v:.1f}", font_size=15, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.1)
                      for v in (1.5, 1.6, 1.7, 1.8, 1.9, 2.0)])
        cap = Text("validation loss vs step", font_size=18, color=GREY_B).next_to(xl, DOWN, buff=0.1)
        return ax, VGroup(xl, yl, cap)

    def loss_curve(self, name):
        vals, col = CURVES[name]
        pts = [self.ax.c2p(s, v) for s, v in zip(STEPS, vals)]
        if name == "warmup-stable-decay":
            pts = pts[:4] + [self.ax.c2p(2400, 1.606)] + pts[4:]
        return VMobject(color=col, stroke_width=5).set_points_as_corners(pts)

    def tag(self, name, y, text=None):
        _, col = CURVES[name]
        return Text(text or name, font_size=18, color=col).move_to([4.5, y, 0])

    # ------------------------------------------------------------------ 4. constant
    def constant(self):
        self.section(4)
        self.clear_stage()
        self.ax, labels = self.loss_axes()
        self.at("constant")
        self.play(Create(self.ax), FadeIn(labels), run_time=0.5)
        self.lines = VGroup()
        for name, y, txt in (("constant 0.001", 2.0, "constant 0.001: 2.00 → 1.605"),
                             ("constant 0.003", 1.5, "constant 0.003: 1.90 → 1.598")):
            c = self.loss_curve(name)
            self.play(Create(c), FadeIn(self.tag(name, y, txt)), run_time=0.7)
            self.lines.add(c)
        meet = Text("by the end, about 1.60 for both", font_size=20, color=YELLOW).move_to([4.5, 0.9, 0])
        self.at("both")
        self.play(FadeIn(meet), run_time=0.4)
        self.meet = meet
        self.end_section()

    # ------------------------------------------------------------------ 5. decay
    def decay(self):
        self.section(5)
        self.play(FadeOut(self.meet), run_time=0.3)
        self.at("decaying")
        for name, y, txt in (("warmup + cosine", 0.6, "cosine: 1.551"), ("warmup + linear", 0.2, "linear: 1.554")):
            c = self.loss_curve(name)
            self.play(Create(c), FadeIn(self.tag(name, y, txt)), run_time=0.7)
        pic = Text("smaller steps settle lower", font_size=20, color=GREY_A).move_to([4.5, -0.4, 0])
        self.at("picture")
        self.play(FadeIn(pic), run_time=0.4)
        self.pic = pic
        self.end_section()

    # ------------------------------------------------------------------ 6. WSD
    def wsd(self):
        self.section(6)
        self.play(FadeOut(self.pic), run_time=0.3)
        name = "warmup-stable-decay"
        c = self.loss_curve(name)
        self.at("stable")
        self.play(Create(c), FadeIn(self.tag(name, -0.8, "WSD: 1.606 at step 2,400")), run_time=1.0)
        band = Rectangle(width=self.ax.c2p(3000, 0)[0] - self.ax.c2p(2400, 0)[0], height=4.4, stroke_width=0,
                         fill_color=GOLD, fill_opacity=0.12).move_to(self.ax.c2p(2700, 1.775))
        bl = Text("decay", font_size=16, color=GOLD).move_to(self.ax.c2p(2700, 2.02))
        self.at("falls")
        self.play(FadeIn(band), FadeIn(bl), run_time=0.4)
        d = Text("→ 1.542: the best of the five", font_size=20, color=GOLD).move_to([4.5, -1.3, 0])
        self.at("54")
        self.play(FadeIn(d), Indicate(c, color=YELLOW), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 7. practical
    def practical(self):
        self.section(7)
        self.clear_stage()
        ax, labels = self.lr_axes(pos=(0, -0.2))
        c = ax.plot(LR["warmup-stable-decay"], x_range=[0, 2999, 5], color=GOLD, stroke_width=5, use_smoothing=False)
        self.at("practical")
        self.play(Create(ax), FadeIn(labels), Create(c), run_time=0.8)
        a = Text("train at the peak as long as you like…", font_size=24).move_to([-1.0, 2.4, 0])
        b = Text("…decay only when you want a finished model", font_size=24, color=GOLD).move_to([1.0, -2.8, 0])
        self.play(FadeIn(a), run_time=0.4)
        self.at("finished")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. lesson
    def lesson(self):
        self.section(8)
        self.clear_stage()
        rows = sorted(((v[0][-1], n, v[1]) for n, v in CURVES.items()))
        out = VGroup()
        for k, (v, name, col) in enumerate(rows):
            y = 1.6 - 0.75 * k
            lab = Text(name, font_size=22, color=col).move_to([-1.0, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=(v - 1.5) * 40, height=0.45, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-0.7, y, 0], aligned_edge=LEFT)
            out.add(VGroup(lab, b, Text(f"{v:.3f}", font=MONO, font_size=22).next_to(b, RIGHT, buff=0.15)))
        cap = Text("final validation loss (bars start at 1.5)", font_size=18, color=GREY_B).next_to(out, UP, buff=0.3)
        self.at("schedule")
        self.play(FadeIn(cap), LaggedStart(*[FadeIn(r) for r in out], lag_ratio=0.15), run_time=1.0)
        g = Text("same peak, same steps: about 0.06 of loss, for free", font_size=24, color=YELLOW).move_to([0, -2.6, 0])
        self.at("hundredths")
        self.play(FadeIn(g), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=21)
        code.move_to([0, 0.2, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("function")
        self.play(Create(hl), run_time=0.3)
        self.at("multiplier")
        self.play(highlight(hl, code, 7), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10. outro
    def outro(self):
        self.section(10)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
