"""Foundations F8 — Averages and Spread.

Render from the repo root:  ./render.sh foundations f08
"""
import numpy as np
from manim import *

from common import code_panel, finish, highlight, next_up_card, token, used_in_card
from f08_script import LABEL, NEXT, TAGLINE, TITLE, USED_IN
from intro import play_token_intro
from voiced_scene import VoicedScene

BOX_COLOR = BLUE_D
DOT_COLOR = BLUE
DIST_COLOR = TEAL
MEAN_COLOR = YELLOW

CODE = """x = np.array([2, 4, 6, 8])
x.mean(), x.std()                  # (5.0, 2.236)
(x - x.mean()) / x.std()           # [-1.34, -0.45, 0.45, 1.34]

q, k = np.random.randn(2, 64)
q @ k / np.sqrt(64)                # typically around ±1"""

# ------------------------------------------------------------------ verified numbers
DATA = np.array([2, 4, 6, 8])
MEAN = DATA.mean()
STD = DATA.std()
DIST = DATA - MEAN
SQ = DIST ** 2
NORM = (DATA - MEAN) / STD
assert DATA.sum() == 20 and MEAN == 5
assert list(DIST) == [-3, -1, 1, 3] and list(SQ) == [9, 1, 1, 9] and SQ.mean() == 5
assert np.isclose(STD, np.sqrt(5)) and f"{STD:.2f}" == "2.24" and f"{STD:.3f}" == "2.236"
assert [f"{v:.2f}" for v in NORM] == ["-1.34", "-0.45", "0.45", "1.34"]
assert np.isclose(NORM.mean(), 0) and np.isclose(NORM.std(), 1)

# sections 7–8: simulated dot products of random standard-normal vectors
_rng = np.random.default_rng(8)
N_PAIRS = 20000
DOTS = {}
for _n in (1, 4, 16, 64):
    DOTS[_n] = (_rng.standard_normal((N_PAIRS, _n)) * _rng.standard_normal((N_PAIRS, _n))).sum(1)
for _n in (1, 4, 16, 64):
    assert abs(DOTS[_n].std() / np.sqrt(_n) - 1) < 0.03, _n
assert abs((DOTS[64] / 8).std() - 1) < 0.01


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


# section 8 inset: one random query against six random keys (64 numbers each)
_r = np.random.default_rng(12)
_q = _r.standard_normal(64)
_keys = _r.standard_normal((6, 64))
SCORES = _keys @ _q
P_RAW = softmax(SCORES)
P_SCALED = softmax(SCORES / 8)
assert P_RAW.max() > 0.98 and P_SCALED.max() < 0.4


def fmt(v, nd=2):
    s = f"{v:.{nd}f}"
    return s.replace("-", "−")


# ------------------------------------------------------------------ helpers
def box(label, color=BOX_COLOR, w=None, h=0.9, fs=32):
    t = Text(label, font_size=fs)
    r = RoundedRectangle(corner_radius=0.12, width=w or max(t.width + 0.5, 0.9), height=h,
                         stroke_color=color, fill_color=color, fill_opacity=0.3)
    return VGroup(r, t.move_to(r))


def arr(a, b, color=GREY_B, sw=4):
    return Arrow(a, b, buff=0.1, stroke_width=sw, color=color, max_tip_length_to_length_ratio=0.3,
                 max_stroke_width_to_length_ratio=10)


def bracket_pair(left, right, top, bottom, color=GREY_B):
    def one(x, d):
        pts = [[x + d, top, 0], [x, top, 0], [x, bottom, 0], [x + d, bottom, 0]]
        return VMobject(stroke_color=color, stroke_width=3).set_points_as_corners(pts)
    return VGroup(one(left, 0.14), one(right, -0.14))


def column(items, center, fs=30, color=WHITE, width=None, gap=0.58):
    """A column vector: returns (brackets, entries). Entries are separate Text objects."""
    n = len(items)
    cx, cy = center[0], center[1]
    entries = VGroup()
    for i, t in enumerate(items):
        entries.add(Text(t, font_size=fs, color=color).move_to([cx, cy + ((n - 1) / 2 - i) * gap, 0]))
    w = width or max(e.width for e in entries)
    half_h = (n - 1) / 2 * gap + 0.32
    br = bracket_pair(cx - w / 2 - 0.22, cx + w / 2 + 0.22, cy + half_h, cy - half_h)
    return br, entries


def illustrative(text="illustrative"):
    return Text(text, font_size=20, color=GREY_B).to_corner(DR, buff=0.3)


def double_arrow(x0, x1, y, color=YELLOW):
    return DoubleArrow([x0, y, 0], [x1, y, 0], buff=0, color=color, stroke_width=3, tip_length=0.12,
                       max_tip_length_to_length_ratio=0.35, max_stroke_width_to_length_ratio=20)


def hist_bars(samples, edges, scale, base_y, max_h, color=BLUE, x0=0.0):
    counts, _ = np.histogram(samples, bins=edges)
    heights = counts / counts.max() * max_h
    bars = VGroup()
    for lo, hi, h in zip(edges[:-1], edges[1:], heights):
        if h < 0.012:
            continue
        r = Rectangle(width=(hi - lo) * scale, height=h, stroke_width=0, fill_color=color, fill_opacity=0.8)
        r.move_to([x0 + (lo + hi) / 2 * scale, base_y + h / 2, 0])
        bars.add(r)
    return bars


# number line of sections 2–4: values 0..10
AX_U = 0.9
AX_Y = -2.0


def ax_x(v):
    return (v - 5) * AX_U


# number line of section 5: value 0 at screen x = 0, rescalable
U5 = 0.8
Y5 = -0.6
TICK_LABELS = {v: Text(str(v).replace("-", "−"), font_size=22, color=GREY_B) for v in range(-8, 9)}


def axis5(s):
    line = Line([-6.6, Y5, 0], [6.6, Y5, 0], color=GREY_B, stroke_width=3)
    ticks, labels = VGroup(), VGroup()
    for v in range(-8, 9):
        x = v * U5 * s
        if abs(x) > 6.5:
            continue
        ticks.add(Line([x, Y5 - 0.08, 0], [x, Y5 + 0.08, 0], color=GREY_B, stroke_width=3))
        labels.add(TICK_LABELS[v].copy().move_to([x, Y5 - 0.38, 0]))
    return VGroup(line, ticks, labels)


class SpreadVideo(VoicedScene):
    VIDEO = "f08"

    def clear(self, run_time=0.4):
        anims = []
        for m in list(self.mobjects):
            if isinstance(m, ValueTracker):
                self.remove(m)
                continue
            m.clear_updaters()
            anims.append(FadeOut(m))
        if anims:
            self.play(*anims, run_time=run_time)

    def construct(self):
        play_token_intro(self, TITLE, 0, TAGLINE, label=LABEL)
        self.s1_drift()
        self.s2_mean()
        self.s3_spread()
        self.s4_worked()
        self.s5_normalize()
        self.s6_layernorm()
        self.s7_dot_spread()
        self.s8_scaling()
        self.s9_code()
        self.s10_outro()
        finish(self)

    # 1. Numbers drifting through layers ------------------------------------------------------
    def s1_drift(self):
        self.section(1)
        layers = VGroup(*[RoundedRectangle(corner_radius=0.1, width=0.6, height=3.0, stroke_color=BOX_COLOR,
                                           fill_color=BOX_COLOR, fill_opacity=0.25) for _ in range(6)])
        layers.arrange(RIGHT, buff=0.5).move_to([0, 0.2, 0])
        brace = Brace(layers, DOWN, color=GREY_B)
        b_lab = Text("dozens of layers", font_size=28, color=GREY_B).next_to(brace, DOWN, buff=0.15)
        br, entries = column(["0.8", "−1.2", "0.5", "1.1"], [-5.2, 0.2, 0], fs=30, width=1.4)
        note = illustrative()
        self.at("deep")
        self.play(FadeIn(layers), FadeIn(br), FadeIn(entries), FadeIn(brace), FadeIn(b_lab), FadeIn(note),
                  run_time=0.6)
        self.at("through")
        self.play(VGroup(br, entries).animate.move_to([4.9, 0.2, 0]), run_time=1.4)

        big = [Text(s, font_size=38, color=RED).move_to(entries[i]) for i, s in [(0, "4812"), (2, "−9377")]]
        tiny = [Text(s, font_size=22, color=GREY_B).move_to(entries[i]) for i, s in [(1, "0.00001"), (3, "−0.00002")]]
        self.at("big")
        self.play(FadeTransform(entries[0], big[0]), FadeTransform(entries[2], big[1]), run_time=0.5)
        self.at("small")
        self.play(FadeTransform(entries[1], tiny[0]), FadeTransform(entries[3], tiny[1]), run_time=0.5)

        cross = VGroup(Text("✗", font_size=64, color=RED), Text("training breaks", font_size=36, color=RED))
        cross.arrange(RIGHT, buff=0.3).move_to([0, 2.75, 0])
        self.at("breaks")
        self.play(FadeIn(cross, scale=1.2), run_time=0.4)

        self.at("two")
        self.play(FadeOut(cross), FadeOut(brace), FadeOut(b_lab), run_time=0.4)
        tame = [Text(t, font_size=30, color=GREEN).move_to(entries[i])
                for i, t in enumerate(["1.3", "−0.4", "−1.1", "0.7"])]
        self.at("check")
        self.play(*[FadeTransform(a, b) for a, b in zip([big[0], tiny[0], big[1], tiny[1]], tame)], run_time=0.5)
        self.at("the")
        self.clear(run_time=0.25)
        mean_l = VGroup(Text("mean", font_size=52, color=MEAN_COLOR),
                        Text("the center", font_size=28, color=GREY_B)).arrange(DOWN, buff=0.3).move_to([-3.6, 0, 0])
        std_l = VGroup(Text("standard deviation", font_size=52, color=DIST_COLOR),
                       Text("the spread", font_size=28, color=GREY_B)).arrange(DOWN, buff=0.3).move_to([2.6, 0, 0])
        self.at("mean")
        self.play(FadeIn(mean_l, shift=0.2 * UP), run_time=0.5)
        self.at("deviation")
        self.play(FadeIn(std_l, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # 2. The mean ------------------------------------------------------------------------------
    def s2_mean(self):
        self.section(2)
        self.clear(run_time=0.35)
        line = Line([ax_x(-0.4), AX_Y, 0], [ax_x(10.4), AX_Y, 0], color=GREY_B, stroke_width=3)
        ticks = VGroup(*[Line([ax_x(v), AX_Y - 0.08, 0], [ax_x(v), AX_Y + 0.08, 0], color=GREY_B, stroke_width=3)
                         for v in range(11)])
        labels = VGroup(*[Text(str(v), font_size=22, color=GREY_B).move_to([ax_x(v), AX_Y - 0.38, 0])
                          for v in range(11)])
        self.axis = VGroup(line, ticks, labels)
        self.dots = VGroup(*[Dot([ax_x(v), AX_Y, 0], radius=0.11, color=DOT_COLOR).set_z_index(3) for v in DATA])
        self.at("numbers")
        self.play(Create(self.axis), LaggedStart(*[FadeIn(d, scale=0.5) for d in self.dots], lag_ratio=0.2),
                  run_time=0.6)

        f1 = Text("(2 + 4 + 6 + 8) ÷ 4", font_size=40)
        f2 = Text("= 5", font_size=40, color=MEAN_COLOR)
        self.formula = VGroup(f1, f2).arrange(RIGHT, buff=0.3).move_to([0, 2.7, 0])
        self.at("divide")
        self.play(FadeIn(f1), run_time=0.4)
        self.at("two")
        self.play(LaggedStart(*[Indicate(d, scale_factor=1.7, color=YELLOW) for d in self.dots], lag_ratio=0.3),
                  run_time=1.3)

        self.marker = Line([ax_x(5), AX_Y - 0.2, 0], [ax_x(5), 0.95, 0], color=MEAN_COLOR, stroke_width=4)
        self.marker.set_z_index(2)
        self.mean_lab = Text("mean = 5", font_size=30, color=MEAN_COLOR).move_to([ax_x(5), 1.25, 0])
        self.at("five")
        self.play(FadeIn(f2), Create(self.marker), FadeIn(self.mean_lab), run_time=0.5)
        self.end_section()

    # 3. Standard deviation: the recipe ------------------------------------------------------
    def s3_spread(self):
        self.section(3)
        self.play(FadeOut(self.formula), run_time=0.3)
        arrows = VGroup()
        for v in DATA:
            h = 0.3 if abs(v - MEAN) == 1 else 0.6
            arrows.add(Arrow([ax_x(MEAN), AX_Y + h, 0], [ax_x(v), AX_Y + h, 0], buff=0, color=DIST_COLOR,
                             stroke_width=4, tip_length=0.18, max_tip_length_to_length_ratio=0.3,
                             max_stroke_width_to_length_ratio=10))
        self.at("spread")
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.15), run_time=0.8)

        typ = Text("typical distance from the mean", font_size=34).move_to([0, 2.7, 0])
        self.at("typically")
        self.play(FadeIn(typ), run_time=0.4)

        self.squares = VGroup()
        for v in DATA:
            d = abs(v - MEAN) * AX_U
            sq = Square(side_length=d, stroke_color=DIST_COLOR, stroke_width=3, fill_color=DIST_COLOR,
                        fill_opacity=0.15)
            sq.move_to([(ax_x(v) + ax_x(MEAN)) / 2, AX_Y + d / 2, 0])
            self.squares.add(sq)
        self.at("square")
        self.play(FadeOut(arrows), *[GrowFromEdge(s, DOWN) for s in self.squares], run_time=0.55)

        left = Text("std = √(", font_size=38)
        mid = Text("average of squares", font_size=38)
        right = Text(")", font_size=38)
        self.recipe = VGroup(left, mid, right).arrange(RIGHT, buff=0.15).move_to([0, 2.7, 0])
        self.at("average")
        self.play(FadeOut(typ), FadeIn(mid), run_time=0.4)
        self.at("root")
        self.play(FadeIn(left), FadeIn(right), run_time=0.4)
        self.end_section()

    # 4. Worked example ----------------------------------------------------------------------
    def s4_worked(self):
        self.section(4)
        self.at("2")
        self.play(LaggedStart(*[Indicate(d, scale_factor=1.7, color=YELLOW) for d in self.dots], lag_ratio=0.3),
                  run_time=1.2)

        y_d = AX_Y - 1.05
        dist = [Text(fmt(v, 0), font_size=32, color=DIST_COLOR).move_to([ax_x(x), y_d, 0])
                for v, x in zip(DIST, DATA)]
        head = Text("distance:", font_size=26, color=GREY_B)
        head.move_to([ax_x(DATA[0]) - 0.7, y_d, 0], aligned_edge=RIGHT)
        self.dist = VGroup(head, *dist)
        self.at("distances")
        self.play(FadeIn(head), run_time=0.3)
        for word, t in zip(["3", "1", "1", "3"], dist):
            self.at(word)
            self.play(FadeIn(t, shift=0.1 * UP), run_time=0.3)

        sq_labels = VGroup()
        for v, s, sq in zip(DATA, SQ, self.squares):
            if s == 9:
                sq_labels.add(Text("9", font_size=40).move_to(sq))
            else:
                sq_labels.add(Text("1", font_size=28).move_to(sq))
        self.sq_labels = sq_labels
        self.at("squares")
        self.play(LaggedStart(*[FadeIn(t, scale=0.7) for t in sq_labels], lag_ratio=0.2), run_time=0.5)

        avg = VGroup(Text("average of squares:", font_size=32, color=GREY_B),
                     Text("(9 + 1 + 1 + 9) ÷ 4 = 5", font_size=36)).arrange(RIGHT, buff=0.35)
        avg.move_to([0, 3.15, 0])
        self.at("5")
        self.play(FadeOut(self.recipe), FadeIn(avg), run_time=0.5)

        s1 = Text("std = √5", font_size=38, color=DIST_COLOR)
        s2 = Text("≈ 2.24", font_size=38, color=DIST_COLOR)
        VGroup(s1, s2).arrange(RIGHT, buff=0.3).move_to([0, 2.3, 0])
        self.at("root")
        self.play(FadeIn(s1), run_time=0.4)
        self.at("24")
        self.play(FadeIn(s2, shift=0.1 * LEFT), run_time=0.4)
        self.end_section()

    # 5. Normalizing ---------------------------------------------------------------------------
    def s5_normalize(self):
        self.section(5)
        axis = axis5(1.0)
        marker = Line([MEAN * U5, Y5 - 0.35, 0], [MEAN * U5, Y5 + 0.35, 0], color=MEAN_COLOR, stroke_width=5)
        keep = {self.axis, self.dots, self.marker, *self.dots}
        fades = [FadeOut(m) for m in self.mobjects if m not in keep and not isinstance(m, ValueTracker)]
        self.play(*fades, run_time=0.3)
        self.play(FadeOut(self.axis), FadeIn(axis), Transform(self.marker, marker),
                  *[d.animate.move_to([v * U5, Y5, 0]) for d, v in zip(self.dots, DATA)], run_time=0.6)
        marker = self.marker

        cap2 = VGroup(Text("(", font_size=40), Text("x − 5", font_size=40), Text(") ÷ 2.24", font_size=40))
        cap2.arrange(RIGHT, buff=0.08).move_to([0, 2.3, 0])
        cap1 = cap2[1]
        self.at("subtract")
        self.play(FadeIn(cap1), VGroup(self.dots, marker).animate.shift(MEAN * U5 * LEFT), run_time=0.9)

        s = ValueTracker(1.0)
        axis.add_updater(lambda m: m.become(axis5(s.get_value())))
        self.at("divide")
        self.play(FadeIn(cap2[0]), FadeIn(cap2[2]), s.animate.set_value(STD), run_time=1.2)
        axis.clear_updaters()
        self.remove(s)

        vals = [Text(fmt(v), font_size=28).next_to(d, UP, buff=0.3) for v, d in zip(NORM, self.dots)]
        for word, t in zip(["1", "0", "0", "1"], vals):
            self.at(word)
            self.play(FadeIn(t, shift=0.1 * DOWN), run_time=0.35)

        m_lab = Text("mean = 0", font_size=32, color=MEAN_COLOR).move_to([0, Y5 + 1.25, 0])
        self.at("zero")
        self.play(FadeIn(m_lab), Indicate(marker, color=MEAN_COLOR, scale_factor=1.2), run_time=0.5)
        span = Line([0, Y5, 0], [U5 * STD, Y5, 0])
        brace = Brace(span, DOWN, color=DIST_COLOR).shift(0.55 * DOWN)
        s_lab = Text("std = 1", font_size=32, color=DIST_COLOR).next_to(brace, DOWN, buff=0.15)
        self.at("one")
        self.play(GrowFromCenter(brace), FadeIn(s_lab), run_time=0.45)
        self.end_section()

    # 6. Layer norm ----------------------------------------------------------------------------
    def s6_layernorm(self):
        self.section(6)
        self.clear(run_time=0.3)
        y = -0.3
        in_br, in_e = column([str(v) for v in DATA], [-5.4, y, 0], fs=30, color=DOT_COLOR, width=0.5)
        tok = token("cat").next_to(in_br, UP, buff=0.3)
        ln = box("layer norm", h=1.0).move_to([-2.6, y, 0])
        out_br, out_e = column([fmt(v) for v in NORM], [0.6, y, 0], fs=30, color=DIST_COLOR)
        stats = Text("mean 0 · std 1", font_size=22, color=GREY_B).next_to(out_br, DOWN, buff=0.3)
        gb = box("× gain + bias", color=PURPLE_B, h=1.0).move_to([4.3, y, 0])
        learned = Text("(learned)", font_size=26, color=PURPLE_A).next_to(gb, DOWN, buff=0.25)
        a1 = arr(in_br.get_right() + 0.05 * RIGHT, ln.get_left())
        a2 = arr(ln.get_right(), out_br.get_left() + 0.05 * LEFT)
        a3 = arr(out_br.get_right() + 0.05 * RIGHT, gb.get_left())
        note = illustrative()
        self.at("layer")
        self.play(FadeIn(tok), FadeIn(in_br), FadeIn(in_e), GrowArrow(a1), FadeIn(ln), FadeIn(note), run_time=0.6)
        self.at("vector")
        self.play(GrowArrow(a2), FadeIn(out_br), FadeIn(out_e), FadeIn(stats), run_time=0.5)
        self.at("learned")
        self.play(GrowArrow(a3), FadeIn(gb), FadeIn(learned), run_time=0.5)
        self.end_section()

    # 7. Spread of a dot product ---------------------------------------------------------------
    def s7_dot_spread(self):
        self.section(7)
        self.clear(run_time=0.35)
        h1 = Text("attention score:  q · k", font_size=34).move_to([0, 3.2, 0])
        h2 = Text("q · k = q₁k₁ + q₂k₂ + … + qₙkₙ", font_size=34).move_to([0, 3.2, 0])
        self.at("attention")
        self.play(FadeIn(h1), run_time=0.4)
        self.at("add")
        self.play(FadeOut(h1), FadeIn(h2), run_time=0.4)

        scale = 1 / 3  # screen units per value unit
        edges = (np.arange(-24.5, 25.5) * 0.5)
        caption = illustrative("simulated: 20,000 random pairs")

        def row(n, base_y):
            bars = hist_bars(DOTS[n], edges, scale, base_y, 1.1)
            sd = DOTS[n].std()
            lab = Text(f"n = {n}", font_size=30).move_to([-5.6, base_y + 0.45, 0])
            base = Line([-4.1, base_y, 0], [4.1, base_y, 0], color=GREY_B, stroke_width=2)
            da = double_arrow(-sd * scale, sd * scale, base_y + 0.45)
            sl = Text(f"spread ≈ {round(sd)}", font_size=28, color=YELLOW).move_to([5.3, base_y + 0.45, 0])
            return VGroup(lab, base, bars), VGroup(da, sl)

        rows = [row(1, 1.2), row(4, -0.45), row(16, -2.1)]
        for word, (g, s), extra in zip(["random", "dot", "grows"], rows, [caption, None, None]):
            self.at(word)
            anims = [FadeIn(g[0]), Create(g[1]), FadeIn(g[2], shift=0.15 * UP)]
            if extra is not None:
                anims.append(FadeIn(extra))
            self.play(*anims, run_time=0.4)
            self.play(GrowFromCenter(s[0]), FadeIn(s[1]), run_time=0.3)

        rule = Text("spread ≈ √n", font_size=38, color=YELLOW).move_to([0, -3.2, 0])
        self.at("root")
        self.play(FadeIn(rule, shift=0.1 * UP), run_time=0.4)
        self.end_section()

    # 8. Scaling by √d -------------------------------------------------------------------------
    def s8_scaling(self):
        self.section(8)
        keep_caption = [m for m in self.mobjects if isinstance(m, Text) and m.text.startswith("simulated")]
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep_caption and not isinstance(m, ValueTracker)],
                  run_time=0.35)
        scale = 0.2
        base_y = 0.9
        edges = np.arange(-30.5, 31.5)
        bars = hist_bars(DOTS[64], edges, scale, base_y, 2.0)
        base = Line([-6.2, base_y, 0], [6.2, base_y, 0], color=GREY_B, stroke_width=2)
        ticks = VGroup(*[Line([v * scale, base_y - 0.07, 0], [v * scale, base_y + 0.07, 0], color=GREY_B,
                              stroke_width=2) for v in range(-24, 25, 8)])
        tick_l = VGroup(*[Text(str(v).replace("-", "−"), font_size=22, color=GREY_B)
                          .move_to([v * scale, base_y - 0.3, 0]) for v in range(-24, 25, 8)])
        n_lab = Text("n = 64", font_size=32).move_to([-5.3, 2.7, 0])
        self.at("64")
        self.play(FadeIn(n_lab), Create(base), FadeIn(ticks), FadeIn(tick_l), FadeIn(bars, shift=0.15 * UP),
                  run_time=0.6)

        sd = DOTS[64].std()
        da = double_arrow(-sd * scale, sd * scale, base_y + 1.0)
        sl = Text("spread ≈ 8", font_size=32, color=YELLOW).move_to([3.9, 2.45, 0])
        self.at("8")
        self.play(GrowFromCenter(da), FadeIn(sl), run_time=0.4)
        wider = Text("8× wider", font_size=26, color=GREY_B).next_to(sl, DOWN, buff=0.15)
        self.at("wider")
        self.play(FadeIn(wider), run_time=0.3)

        div = Text("÷ √64 = ÷ 8", font_size=36, color=YELLOW).move_to([-4.6, 1.9, 0])
        self.at("dividing")
        self.play(FadeIn(div), FadeOut(da), FadeOut(sl), FadeOut(wider),
                  bars.animate.stretch(1 / 8, 0, about_point=[0, base_y, 0]), run_time=1.2)

        s1 = Text("spread ≈ 1", font_size=32, color=YELLOW).move_to([2.6, 2.45, 0])
        pointer = Arrow(s1.get_left() + 0.1 * LEFT, [0.3, base_y + 1.5, 0], buff=0.05, color=YELLOW,
                        stroke_width=3, max_tip_length_to_length_ratio=0.2)
        self.at("1")
        self.play(FadeIn(s1), GrowArrow(pointer), run_time=0.4)

        def mini(probs, cx, color, title, t_color):
            bw, pitch, bh, by = 0.34, 0.46, 1.35, -3.0
            g = VGroup()
            for i, p in enumerate(probs):
                x = cx + (i - (len(probs) - 1) / 2) * pitch
                h = max(p * bh, 0.02)
                g.add(Rectangle(width=bw, height=h, stroke_width=0, fill_color=color, fill_opacity=0.85)
                      .move_to([x, by + h / 2, 0]))
            floor = Line([cx - 1.5, by, 0], [cx + 1.5, by, 0], color=GREY_B, stroke_width=2)
            head = Text(title, font_size=26, color=t_color).move_to([cx, -0.7, 0])
            return VGroup(floor, g, head), g

        left, lbars = mini(P_RAW, -3.0, RED, "softmax(q·k): too sharp", RED)
        right, _ = mini(P_SCALED, 3.0, GREEN, "softmax(q·k ÷ 8): well behaved", GREEN)
        top = int(np.argmax(P_RAW))
        p_lab = Text(fmt(P_RAW[top]), font_size=22).next_to(lbars[top], UP, buff=0.08)
        self.at("softmax")
        self.play(FadeIn(left), FadeIn(right), FadeIn(p_lab), run_time=0.6)
        self.at("behaved")
        self.play(Indicate(right[2], color=GREEN, scale_factor=1.1), run_time=0.6)
        self.end_section()

    # 9. Code ------------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, font_size=26)
        code.move_to(ORIGIN)
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[1])
        self.clear(run_time=0.3)
        self.play(FadeIn(code, shift=0.3 * UP), run_time=0.4)
        self.at("mean")
        self.play(Create(hl), run_time=0.4)
        self.at("normalizing")
        self.play(highlight(hl, code, 2), run_time=0.4)
        self.end_section()

    # 10. Outro ------------------------------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        card = used_in_card(USED_IN)
        self.at("see")
        self.clear(run_time=0.3)
        self.play(FadeIn(card), run_time=0.4)
        self.at("9")
        self.play(Indicate(card[1][1], color=YELLOW, scale_factor=1.08), run_time=0.8)
        self.at("6")
        self.play(Indicate(card[1][0], color=YELLOW, scale_factor=1.08), run_time=0.8)
        nxt = next_up_card(NEXT)
        self.at("rotations")
        self.play(FadeOut(card), run_time=0.3)
        self.play(FadeIn(nxt, shift=0.2 * UP), run_time=0.4)
        self.end_section()
