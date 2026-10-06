"""How LLMs Work: Deep Dive, episode 33 — Quantization, Deeper.

Render from the repo root:  ./render.sh deep-dive d33
Every number on screen comes from code/d33_quantization/quantization.py (Qwen2.5-0.5B, loss on 4 × 512 tokens of Tiny
Shakespeare; weights or linear-layer inputs rounded). assets/d33/row.json is row 0 of layer 0's down_proj.
"""
import json
from pathlib import Path

import numpy as np
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from d33_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 33"
ROW = np.array(json.loads((Path(__file__).parent / "assets/d33/row.json").read_text()))
NF4 = [-1.00, -0.70, -0.53, -0.39, -0.28, -0.18, -0.09, 0.00, 0.08, 0.16, 0.25, 0.34, 0.44, 0.56, 0.72, 1.00]
BASE = 3.393
CODE = """x = w.reshape(rows, -1, 64)         # blocks of 64
scale = x.abs().amax(-1, True) / 7  # one per block
q = (x / scale).round()             # integers -7..7
w_hat = (q * scale).reshape(w.shape)"""


def loss_bars(rows, y0=1.6, dy=0.75, unit=1.1, x0=-1.6, base=True):
    """rows: (label, loss, color). Bars start at loss 3.0 so differences are visible; a dashed line marks full precision."""
    g = VGroup()
    for k, (name, v, col) in enumerate(rows):
        y = y0 - dy * k
        lab = Text(name, font_size=22).move_to([x0 - 0.3, y, 0], aligned_edge=RIGHT)
        w = min(v - 3.0, 4.6) * unit
        b = Rectangle(width=max(w, 0.02), height=0.45, stroke_width=0, fill_color=col, fill_opacity=0.85)
        b.move_to([x0, y, 0], aligned_edge=LEFT)
        num = Text(f"{v:.3f}" if v < 10 else f"{v:.1f}", font=MONO, font_size=20).next_to(b, RIGHT, buff=0.15)
        g.add(VGroup(lab, b, num))
    line = None
    if base:
        x = x0 + (BASE - 3.0) * unit
        line = DashedLine([x, y0 + 0.5, 0], [x, y0 - dy * (len(rows) - 1) - 0.5, 0], color=GREY_B, stroke_width=2)
    return g, line


class QuantizationVideo(VoicedScene):
    VIDEO = "d33"

    def construct(self):
        play_token_intro(self, TITLE, 33, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.outliers()      # 2
        self.groups()        # 3
        self.cost()          # 4
        self.nf4()           # 5
        self.activations()   # 6
        self.fixes()         # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        big = VGroup(*[Square(0.32, stroke_width=1, color=MODEL_COLOR, fill_opacity=0.5) for _ in range(32)])
        big.arrange_in_grid(rows=4, cols=8, buff=0.04).move_to([-3.2, 0.4, 0])
        bl = Text("32 bits per weight", font_size=22).next_to(big, DOWN)
        small = VGroup(*[Square(0.32, stroke_width=1, color=GREEN_C, fill_opacity=0.5) for _ in range(4)])
        small.arrange(RIGHT, buff=0.04).move_to([3.2, 0.4, 0])
        sl = Text("4 bits: 8x smaller", font_size=22, color=GREEN_B).next_to(small, DOWN)
        self.at("rounding")
        self.play(FadeIn(big), FadeIn(bl), run_time=0.5)
        self.at("smaller")
        self.play(TransformFromCopy(big, small), FadeIn(sl), run_time=0.8)
        w = Text("naively: much worse", font_size=26, color=RED_B).move_to([0, -2.0, 0])
        self.at("naively")
        self.play(FadeIn(w), run_time=0.4)
        f = Text("real tools: almost free", font_size=26, color=YELLOW).move_to([0, -2.7, 0])
        self.at("real")
        self.play(FadeIn(f), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. outliers
    def outliers(self):
        self.section(2)
        self.clear_stage()
        head = Text("one row of Qwen's layer 0 (4,864 weights)", font_size=26).to_edge(UP, buff=0.5)
        self.at("outliers")
        self.play(FadeIn(head), run_time=0.4)
        lim = 0.18
        ax = NumberLine(x_range=[-lim, lim, 0.06], length=11).move_to([0, -1.4, 0])
        nums = VGroup(*[Text(f"{v:+.2f}" if v else "0", font_size=18).next_to(ax.n2p(v), DOWN, buff=0.2)
                        for v in (-0.18, -0.12, -0.06, 0, 0.06, 0.12, 0.18)])
        counts, edges = np.histogram(ROW, bins=90, range=(-lim, lim))
        hmax = counts.max()
        bars = VGroup()
        for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
            if c == 0:
                continue
            h = max(c / hmax * 2.8, 0.03)
            r = Rectangle(width=ax.n2p(e1)[0] - ax.n2p(e0)[0], height=h, stroke_width=0, fill_color=BLUE_C, fill_opacity=0.85)
            r.move_to(ax.n2p((e0 + e1) / 2) + UP * h / 2, aligned_edge=DOWN)
            bars.add(r)
        self.at("row")
        self.play(Create(ax), FadeIn(nums), FadeIn(bars, lag_ratio=0.01), run_time=0.8)
        typ = Text("typical |w| ≈ 0.0125", font_size=22, color=BLUE_B).move_to([-3.6, 2.0, 0])
        self.at("typical")
        self.play(FadeIn(typ), run_time=0.3)
        mx = float(np.abs(ROW).max())
        xm = float(ROW[np.abs(ROW).argmax()])
        arr = Arrow(ax.n2p(xm) + UP * 1.4, ax.n2p(xm) + UP * 0.1, buff=0, color=RED_B)
        ml = Text(f"largest {mx:.3f}: 14x", font_size=22, color=RED_B).next_to(arr, UP, buff=0.1).shift(RIGHT * 0.4)
        ml.add_background_rectangle(opacity=0.85).set_z_index(3)
        self.at("largest")
        self.play(GrowArrow(arr), FadeIn(ml), run_time=0.5)
        step = mx / 7
        ticks = VGroup(*[Line(ax.n2p(k * step) + DOWN * 0.15, ax.n2p(k * step) + UP * 3.0, color=YELLOW, stroke_width=1.5,
                              stroke_opacity=0.7) for k in range(-7, 8)])
        tl = Text("4-bit levels, one scale per row", font_size=22, color=YELLOW).move_to([3.6, 2.0, 0])
        self.at("coarse")
        self.play(Create(ticks, lag_ratio=0.05), FadeIn(tl), run_time=0.7)
        z = Text("49% of the weights round to 0", font_size=24, color=RED_B).move_to([0, -2.6, 0])
        self.at("half")
        self.play(FadeIn(z), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. groups
    def groups(self):
        self.section(3)
        self.clear_stage()
        row = VGroup(*[Rectangle(width=1.3, height=0.45, stroke_width=1.5, color=GREY_B) for _ in range(8)]).arrange(RIGHT, buff=0.06)
        row.move_to([0, 2.6, 0])
        sc = VGroup(*[Text("scale", font_size=16, color=YELLOW).next_to(r, DOWN, buff=0.08) for r in row])
        self.at("groups")
        self.play(FadeIn(row), run_time=0.4)
        self.at("64")
        self.play(FadeIn(sc, lag_ratio=0.1), run_time=0.6)
        bad = Dot(row[2].get_center(), color=RED_B, radius=0.1)
        self.at("outlier")
        self.play(FadeIn(bad), row[2].animate.set_fill(RED_E, opacity=0.5), run_time=0.4)
        rows = [("full precision", 3.393, GREY_B), ("4 bits, per row", 4.186, RED_C), ("groups of 128", 3.756, GOLD),
                ("groups of 64", 3.622, GREEN_C), ("groups of 32", 3.570, GREEN_B)]
        g, line = loss_bars(rows, y0=1.0, dy=0.7, unit=4.0, x0=-1.6)
        cap = Text("loss (lower is better)", font_size=20, color=GREY_B).move_to([3.5, 1.7, 0])
        self.at("full")
        self.play(FadeIn(g[0]), FadeIn(cap), run_time=0.3)
        for k, cue in [(1, "row"), (2, "128"), (3, "64"), (4, "32")]:
            self.at(cue)
            self.play(FadeIn(g[k]), run_time=0.3)
        self.play(Create(line), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4. cost and 3 bits
    def cost(self):
        self.section(4)
        self.clear_stage()
        eq = Text("4 bits + 16-bit scale / 32 weights = 4.5 bits per weight", font_size=26).to_edge(UP, buff=0.7)
        self.at("scales")
        self.play(FadeIn(eq), run_time=0.4)
        tbl = VGroup(*[Text(t, font=MONO, font_size=22) for t in
                       ["per row      4.00 bits", "groups 128   4.12 bits", "groups 64    4.25 bits", "groups 32    4.50 bits"]])
        tbl.arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([-3.2, 0.4, 0])
        self.at("half")
        self.play(FadeIn(tbl, lag_ratio=0.1), run_time=0.5)
        h3 = Text("3 bits", font_size=26, color=YELLOW).move_to([3.0, 1.6, 0])
        self.at("three")
        self.play(FadeIn(h3), run_time=0.3)
        b1 = VGroup(Text("per row", font_size=22), Text("loss 12.5", font=MONO, font_size=26, color=RED_B)).arrange(DOWN)
        b1.move_to([1.8, 0.2, 0])
        self.at("destroys")
        self.play(FadeIn(b1), run_time=0.4)
        b2 = VGroup(Text("groups of 32", font_size=22), Text("loss 4.52", font=MONO, font_size=26, color=GREEN_B)).arrange(DOWN)
        b2.move_to([4.4, 0.2, 0])
        self.at("back")
        self.play(FadeIn(b2), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. NF4
    def nf4(self):
        self.section(5)
        self.clear_stage()
        ax = NumberLine(x_range=[-1, 1, 0.5], length=8.5).shift(RIGHT * 0.6).move_to([0, -0.6, 0])
        nums = VGroup(*[Text(t, font_size=18).next_to(ax.n2p(v), UP, buff=0.15)
                        for v, t in ((-1, "-max"), (0, "0"), (1, "+max"))])
        pts = [ax.n2p(x) + UP * 2.2 * np.exp(-(x / 0.33) ** 2 / 2) for x in np.linspace(-1, 1, 120)]
        bell = VMobject(color=BLUE_B, stroke_width=3).set_points_smoothly(pts)
        self.at("bell")
        self.play(Create(ax), FadeIn(nums), Create(bell), run_time=0.8)
        even = VGroup(*[Line(ax.n2p(k / 7) + DOWN * 0.5, ax.n2p(k / 7) + DOWN * 0.95, color=GOLD, stroke_width=3)
                        for k in range(-7, 8)])
        el = Text("evenly spaced", font_size=20, color=GOLD).next_to(even, LEFT, buff=0.3)
        self.at("evenly")
        self.play(Create(even, lag_ratio=0.05), FadeIn(el), run_time=0.6)
        nf = VGroup(*[Line(ax.n2p(v) + DOWN * 1.2, ax.n2p(v) + DOWN * 1.65, color=GREEN_B, stroke_width=3) for v in NF4])
        nl = Text("NF4", font_size=20, color=GREEN_B).next_to(nf, LEFT, buff=0.3)
        self.at("nf4")
        self.play(Create(nf, lag_ratio=0.05), FadeIn(nl), run_time=0.6)
        r = Text("loss increase, 4 bits, groups of 64:  even +0.229   NF4 +0.140", font_size=24, color=YELLOW)
        r.move_to([0, -3.0, 0])
        self.at("drops")
        self.play(FadeIn(r), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. activations
    def activations(self):
        self.section(6)
        self.clear_stage()
        head = Text("inputs to a linear layer: largest value per token", font_size=26).to_edge(UP, buff=0.5)
        self.at("inputs")
        self.play(FadeIn(head), run_time=0.4)
        rng = np.random.default_rng(4)
        vals = [1808] + list(1.86 * rng.uniform(0.6, 1.5, 23))
        bars = VGroup()
        for k, v in enumerate(vals):
            h = 4.6 if k == 0 else v * 0.12
            col = RED_C if k == 0 else TOKEN_COLOR
            bars.add(Rectangle(width=0.3, height=h, stroke_width=0, fill_color=col, fill_opacity=0.85))
        bars.arrange(RIGHT, buff=0.08, aligned_edge=DOWN).move_to([0, -2.3, 0], aligned_edge=DOWN)
        self.at("token")
        self.play(FadeIn(bars[1:], lag_ratio=0.02), run_time=0.5)
        self.at("first")
        self.play(GrowFromEdge(bars[0], DOWN), run_time=0.6)
        l0 = Text("first token: 1,808", font_size=22, color=RED_B).next_to(bars[0], RIGHT, buff=0.2).shift(UP * 1.8)
        l1 = Text("typical token: 1.86 (969x smaller)", font_size=22, color=TOKEN_COLOR).move_to([2.2, -0.6, 0])
        self.at("layer")
        self.play(FadeIn(l0), run_time=0.3)
        self.at("typical")
        self.play(FadeIn(l1), run_time=0.3)
        s = Text("the attention sink (episode 12)", font_size=22, color=GREY_A).move_to([2.2, 1.0, 0])
        self.at("sink")
        self.play(FadeIn(s), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. fixes
    def fixes(self):
        self.section(7)
        self.clear_stage()
        head = Text("8-bit inputs to every linear layer", font_size=26).to_edge(UP, buff=0.5)
        z = Text("one scale per tensor: 98% of the other tokens become all zeros", font_size=22, color=RED_B)
        z.next_to(head, DOWN, buff=0.3)
        self.at("tensor")
        self.play(FadeIn(head), FadeIn(z), run_time=0.4)
        rows = [("full precision", 3.393, GREY_B), ("per tensor", 4.709, RED_C), ("per tensor, first token kept", 3.475, GOLD),
                ("per token", 3.415, GREEN_B)]
        g, line = loss_bars(rows, y0=0.8, dy=0.85, unit=3.2, x0=-0.4)
        self.play(FadeIn(g[0]), run_time=0.3)
        for k, cue in [(1, "jumps"), (2, "keep"), (3, "every")]:
            self.at(cue)
            self.play(FadeIn(g[k]), run_time=0.3)
        self.play(Create(line), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=28)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("reshape")
        self.play(Create(hl), run_time=0.3)
        self.at("largest")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("round")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
