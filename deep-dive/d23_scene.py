"""How LLMs Work: Deep Dive, episode 23 — Scaling Laws.

Render from the repo root:  ./render.sh deep-dive d23
Every number on screen comes from code/d23_scaling_laws/scaling_laws.py (six tiny GPTs of 7,697 to 1,198,273
non-embedding parameters on 4,096,000 tokens; one 453,857-parameter model as its training tokens double).
"""
import math

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d23_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 23"
SIZES = [(7697, 2.280), (27617, 2.070), (88097, 1.879), (154305, 1.774), (453857, 1.684), (1198273, 1.612)]
DATA = [(256000, 2.406), (512000, 2.165), (1024000, 1.987), (2048000, 1.798), (4096000, 1.684), (8192000, 1.586)]
FIT_N = (4.24, -0.071)
FIT_D = (10.68, -0.121)
CODE = """lx = [math.log(n) for n in params]
ly = [math.log(l) for l in losses]
slope = cov(lx, ly) / var(lx)               # least-squares line, in log-log space
a = math.exp(mean(ly) - slope * mean(lx))
# loss ≈ a * N ** slope   →   4.24 * N ** -0.071"""


class ScalingLawsVideo(VoicedScene):
    VIDEO = "d23"

    def construct(self):
        play_token_intro(self, TITLE, 23, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.sizes()         # 2
        self.loglog()        # 3
        self.off_line()      # 4
        self.data()          # 5
        self.real()          # 6
        self.predict()       # 7
        self.limits()        # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def log_axes(self, xr, yr, xticks, yticks, xfmt, pos, w=7.6, h=4.4, xlabel=""):
        ax = Axes(x_range=[*xr, 1], y_range=[*yr, 0.05], x_length=w, y_length=h, tips=False,
                  axis_config={"color": GREY_B}).move_to(pos)
        xl = VGroup(*[Text(xfmt(v), font_size=15, color=GREY_B).next_to(ax.c2p(math.log10(v), yr[0]), DOWN, buff=0.1)
                      for v in xticks])
        yl = VGroup(*[Text(f"{v:.1f}", font_size=15, color=GREY_B).next_to(ax.c2p(xr[0], math.log10(v)), LEFT, buff=0.1)
                      for v in yticks])
        cap = Text(xlabel, font_size=18, color=GREY_B).next_to(xl, DOWN, buff=0.1)
        return ax, VGroup(xl, yl, cap)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        boxes = VGroup(*[Square(s, color=MODEL_COLOR, fill_opacity=0.3) for s in (0.4, 0.8, 1.4, 2.2)])
        boxes.arrange(RIGHT, buff=0.5, aligned_edge=DOWN).move_to([0, 0.5, 0])
        self.at("bigger")
        self.play(LaggedStart(*[FadeIn(b, shift=0.2 * UP) for b in boxes], lag_ratio=0.2), run_time=0.8)
        t = Text("the loss falls predictably as models and data grow", font_size=28, color=YELLOW).move_to([0, -1.6, 0])
        self.at("predictable")
        self.play(FadeIn(t), run_time=0.4)
        l = Text("let's find the law at laptop scale", font_size=24, color=GREY_A).move_to([0, -2.4, 0])
        self.at("laptop")
        self.play(FadeIn(l), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. six sizes
    def sizes(self):
        self.section(2)
        self.clear_stage()
        head = Text("six tiny GPTs, the same 4,096,000 training tokens", font_size=26).to_edge(UP, buff=0.5)
        self.at("six")
        self.play(FadeIn(head), run_time=0.4)
        hdr = VGroup(Text("parameters (excluding embeddings)", font_size=20, color=GREY_B).move_to([-2.0, 2.0, 0]),
                     Text("val loss", font_size=20, color=GREY_B).move_to([2.6, 2.0, 0]))
        self.play(FadeIn(hdr), run_time=0.3)
        rows = VGroup(*[VGroup(Text(f"{n:,}", font=MONO, font_size=24).move_to([-2.0, 1.3 - 0.6 * k, 0]),
                               Text(f"{l:.3f}", font=MONO, font_size=24, color=YELLOW).move_to([2.6, 1.3 - 0.6 * k, 0]))
                        for k, (n, l) in enumerate(SIZES)])
        for k, cue in enumerate(("28", "07", "88", "77", "68", "61")):
            self.at(cue)
            self.play(FadeIn(rows[k]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. log-log
    def loglog(self):
        self.section(3)
        self.clear_stage()
        self.ax, labels = self.log_axes((3.5, 6.5), (math.log10(1.5), math.log10(2.4)), (10 ** 4, 10 ** 5, 10 ** 6),
                                        (1.6, 1.8, 2.0, 2.2), lambda v: f"{v:,}", [-1.6, -0.3, 0],
                                        xlabel="parameters (log scale); loss (log scale)")
        self.at("logarithmic")
        self.play(Create(self.ax), FadeIn(labels), run_time=0.6)
        self.dots = VGroup(*[Dot(self.ax.c2p(math.log10(n), math.log10(l)), color=YELLOW, radius=0.08) for n, l in SIZES])
        self.play(LaggedStart(*[FadeIn(d, scale=1.5) for d in self.dots], lag_ratio=0.1), run_time=0.8)
        a, s = FIT_N
        line = self.ax.plot(lambda lx: math.log10(a) + s * lx, x_range=[3.7, 6.3], color=BLUE_C, stroke_width=4)
        f = Text("loss ≈ 4.24 × N^−0.071", font=MONO, font_size=24, color=BLUE_C).move_to([4.4, 1.6, 0])
        self.at("power")
        self.play(Create(line), FadeIn(f), run_time=0.7)
        x10 = Text("× 10 parameters\n→ loss × 0.85", font_size=24, color=YELLOW, line_spacing=0.85).move_to([4.4, 0.4, 0])
        self.at("85")
        self.play(FadeIn(x10), run_time=0.4)
        self.line, self.labels = line, labels
        self.end_section()

    # ------------------------------------------------------------------ 4. off the line
    def off_line(self):
        self.section(4)
        last = self.dots[-1]
        ring = Circle(radius=0.22, color=RED_C).move_to(last)
        t = Text("1.612 measured,\n1.577 predicted", font=MONO, font_size=20, color=RED_B, line_spacing=0.85)
        t.move_to([4.4, -0.9, 0])
        self.at("biggest")
        self.play(Create(ring), FadeIn(t), run_time=0.5)
        r = Text("only 3.4 tokens\nper parameter:\nshort of data", font_size=20, color=GREY_A, line_spacing=0.85)
        r.move_to([4.7, -2.1, 0])
        self.at("three")
        self.play(FadeIn(r), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. data
    def data(self):
        self.section(5)
        self.clear_stage()
        head = Text("one model (453,857 parameters), training tokens doubling", font_size=24).to_edge(UP, buff=0.4)
        self.at("data")
        self.play(FadeIn(head), run_time=0.4)
        ax, labels = self.log_axes((5.2, 7.1), (math.log10(1.5), math.log10(2.5)), (10 ** 6, 10 ** 7), (1.6, 1.8, 2.0, 2.2,
                                                                                                         2.4),
                                   lambda v: f"{v:,}", [-1.6, -0.5, 0], xlabel="training tokens (log scale)")
        self.play(Create(ax), FadeIn(labels), run_time=0.5)
        dots = VGroup(*[Dot(ax.c2p(math.log10(n), math.log10(l)), color=GREEN_C, radius=0.08) for n, l in DATA])
        vals = Text("2.406 → 2.165 → 1.987 → 1.798 → 1.684 → 1.586", font=MONO, font_size=18, color=GREEN_B)
        vals.move_to([0, 2.5, 0])
        self.at("41")
        self.play(LaggedStart(*[FadeIn(d, scale=1.5) for d in dots], lag_ratio=0.3), FadeIn(vals), run_time=2.0)
        a, s = FIT_D
        line = ax.plot(lambda lx: math.log10(a) + s * lx, x_range=[5.3, 7.0], color=BLUE_C, stroke_width=4)
        f = Text("loss ≈ 10.68 × tokens^−0.121\n× 2 tokens → loss × 0.92", font=MONO, font_size=20, color=BLUE_C,
                 line_spacing=0.85).move_to([4.3, 0.8, 0])
        self.at("92")
        self.play(Create(line), FadeIn(f), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 6. real scaling laws
    def real(self):
        self.section(6)
        self.clear_stage()
        a = Text("real language models: the same shape,\nover many orders of magnitude of size, data and compute",
                 font_size=26, line_spacing=0.9).move_to([0, 1.2, 0])
        self.at("research")
        self.play(FadeIn(a), run_time=0.5)
        b = Text("Chinchilla (2022): for a fixed compute budget,\nabout 20 training tokens per parameter", font_size=28,
                 color=YELLOW, line_spacing=0.9).move_to([0, -0.6, 0])
        self.at("chinchilla")
        self.play(FadeIn(b), run_time=0.5)
        c = Text("our biggest model: 4,096,000 / 1,198,273 ≈ 3.4", font=MONO, font_size=22, color=GREY_A).move_to([0, -2.1, 0])
        self.at("20")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. prediction
    def predict(self):
        self.section(7)
        self.clear_stage()
        ax, labels = self.log_axes((3.5, 9.5), (math.log10(0.8), math.log10(2.4)), (10 ** 4, 10 ** 6, 10 ** 8),
                                   (1.0, 1.4, 2.0), lambda v: f"10^{int(math.log10(v))}", [-0.5, -0.4, 0], w=9.0,
                                   xlabel="parameters (log scale)")
        self.at("plan")
        self.play(Create(ax), FadeIn(labels), run_time=0.5)
        dots = VGroup(*[Dot(ax.c2p(math.log10(n), math.log10(l)), color=YELLOW, radius=0.07) for n, l in SIZES])
        a, s = FIT_N
        solid = ax.plot(lambda lx: math.log10(a) + s * lx, x_range=[3.7, 6.2], color=BLUE_C, stroke_width=4)
        dashed = DashedVMobject(ax.plot(lambda lx: math.log10(a) + s * lx, x_range=[6.2, 9.3], color=BLUE_C,
                                        stroke_width=4), num_dashes=20)
        self.play(FadeIn(dots), Create(solid), run_time=0.5)
        t = Text("fit on small, cheap runs → predict a 1,000× larger model", font_size=24, color=YELLOW)
        t.to_edge(UP, buff=0.6)
        self.at("thousand")
        self.play(Create(dashed), FadeIn(t), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 8. limits
    def limits(self):
        self.section(8)
        self.clear_stage()
        items = VGroup(Text("the laws describe the loss, not every ability", font_size=26),
                       Text("they bend when something runs short (our biggest model)", font_size=26),
                       Text("for the loss itself, they've held remarkably well", font_size=26, color=YELLOW))
        items.arrange(DOWN, buff=0.45).move_to([0, 0.3, 0])
        for k, cue in enumerate(("ability", "bend", "remarkably")):
            self.at(cue)
            self.play(FadeIn(items[k]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("straight")
        self.play(Create(hl), run_time=0.3)
        self.at("logarithms")
        self.play(highlight(hl, code, 0), run_time=0.3)
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
