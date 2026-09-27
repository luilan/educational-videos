"""Foundations F4 — Straight Lines and Bends.

Render from the repo root:  ./render.sh foundations f04
"""
import numpy as np
from manim import *

from common import code_panel, finish, highlight, next_up_card, used_in_card
from f04_script import LABEL, NEXT, TAGLINE, TITLE, USED_IN
from intro import play_token_intro
from voiced_scene import VoicedScene

BOX_COLOR = BLUE_D
BEND_COLOR = YELLOW

CODE = """def relu(x):
    return np.maximum(0, x)

relu(np.array([-2, -0.5, 0, 1, 3]))   # [0, 0, 0, 1, 3]

h = relu(x @ W1)                      # linear, then bend
y = h @ W2                            # linear again"""

# ------------------------------------------------------------------ verified numbers
RELU_IN = np.array([-2, -0.5, 0, 1, 3])
assert np.array_equal(np.maximum(0, RELU_IN), [0, 0, 0, 1, 3])

# section 2: an illustrative 2×2 matrix and two input vectors
W_EX = np.array([[1.0, -0.5], [0.5, 1.0]])
VA = np.array([1.0, 0.5])
VB = np.array([0.5, 1.5])
assert np.allclose(W_EX @ (2 * VA), 2 * (W_EX @ VA))
assert np.allclose(W_EX @ (VA + VB), W_EX @ VA + W_EX @ VB)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


_xs = np.linspace(-3, 0, 30001)
GELU_MIN_X = float(_xs[np.argmin(gelu(_xs))])
assert abs(gelu(3) - 2.996) < 5e-4 and abs(gelu(-3) + 0.004) < 5e-4
assert abs(gelu(GELU_MIN_X) + 0.17) < 5e-3


# section 8: target curve and greedy piecewise-linear (ReLU-sum) approximations
def target(x):
    return np.sin(1.3 * x + 0.4) + 0.25 * x


def greedy_knots(n_pieces, lo=-3.0, hi=3.0):
    grid = np.linspace(lo, hi, 601)
    knots = [lo, hi]
    out = [list(knots)]
    while len(knots) - 1 < n_pieces:
        approx = np.interp(grid, knots, target(np.array(knots)))
        knots = sorted(knots + [round(float(grid[np.argmax(np.abs(target(grid) - approx))]), 2)])
        out.append(list(knots))
    return out


KNOT_SETS = greedy_knots(8)


# ------------------------------------------------------------------ helpers
def box(label, color=BOX_COLOR, w=None, h=0.9, fs=34, label_color=WHITE):
    t = Text(label, font_size=fs, color=label_color)
    r = RoundedRectangle(corner_radius=0.12, width=w or max(t.width + 0.5, 0.9), height=h,
                         stroke_color=color, fill_color=color, fill_opacity=0.3)
    return VGroup(r, t.move_to(r))


def arr(a, b, color=GREY_B, sw=4):
    return Arrow(a, b, buff=0.1, stroke_width=sw, color=color, max_tip_length_to_length_ratio=0.3,
                 max_stroke_width_to_length_ratio=10)


def chain(items, gap=0.8, center=ORIGIN):
    """Lay items out left→right; returns (items group, arrows between them)."""
    g = VGroup(*items).arrange(RIGHT, buff=gap).move_to(center)
    arrows = VGroup(*[arr(items[i].get_right(), items[i + 1].get_left()) for i in range(len(items) - 1)])
    return g, arrows


def stack(labels, center, bw=1.4, bh=0.46, gap=0.1, fs=24):
    """Vertical layer stack (bottom→top) with x in at the bottom and y out at the top.

    Returns (items, arrow_in, arrow_out, x_label, y_label)."""
    items = VGroup()
    for lab in labels:
        if lab == "⋮":
            items.add(Text("⋮", font_size=36, color=GREY_B))
        elif lab == "ReLU":
            items.add(box("ReLU", color=BEND_COLOR, w=bw, h=bh * 0.85, fs=22))
        else:
            items.add(box(lab, w=bw, h=bh, fs=fs))
    items.arrange(UP, buff=gap).move_to(center)
    a_in = arr(items.get_bottom() + 0.6 * DOWN, items.get_bottom())
    a_out = arr(items.get_top(), items.get_top() + 0.6 * UP)
    x_lab = Text("x", font_size=30, color=BLUE).next_to(a_in, DOWN, buff=0.08)
    y_lab = Text("y", font_size=30, color=TEAL).next_to(a_out, UP, buff=0.08)
    return items, a_in, a_out, x_lab, y_lab


def bracket_pair(left, right, top, bottom, color=GREY_B):
    def one(x, d):
        pts = [[x + d, top, 0], [x, top, 0], [x, bottom, 0], [x + d, bottom, 0]]
        return VMobject(stroke_color=color, stroke_width=3).set_points_as_corners(pts)
    return VGroup(one(left, 0.14), one(right, -0.14))


def hinge_icon(color=YELLOW, flip=False):
    pts = [[-0.22, -0.12, 0], [0.0, -0.12, 0], [0.2, 0.14, 0]]
    icon = VMobject(stroke_color=color, stroke_width=4).set_points_as_corners(pts)
    if flip:
        icon.flip(UP)
    return icon


def illustrative():
    return Text("illustrative", font_size=20, color=GREY_B).to_corner(DR, buff=0.3)


def polyline(axes, xs, ys, color=YELLOW, sw=5):
    return VMobject(stroke_color=color, stroke_width=sw).set_points_as_corners(
        [axes.c2p(x, y) for x, y in zip(xs, ys)])


class LinesAndBendsVideo(VoicedScene):
    VIDEO = "f04"

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
        self.s1_linear()
        self.s2_matrix()
        self.s3_collapse()
        self.s4_limits()
        self.s5_fix()
        self.s6_relu()
        self.s7_gelu()
        self.s8_approx()
        self.s9_code()
        self.s10_outro()
        finish(self)

    # 1. Linear functions ------------------------------------------------------------------
    def s1_linear(self):
        self.section(1)
        fbox = box("f", w=1.3, h=1.0, fs=44)
        inp = Text("input", font_size=32, color=BLUE)
        out = Text("output", font_size=32, color=TEAL)
        diag, arrows = chain([inp, fbox, out], gap=1.1, center=0.3 * UP)
        self.at("function")
        self.play(FadeIn(fbox), FadeIn(inp), FadeIn(out), *[GrowArrow(a) for a in arrows], run_time=0.8)

        axes = Axes(x_range=[-2.5, 2.5, 1], y_range=[-5, 5, 1], x_length=5, y_length=6.2, tips=True,
                    axis_config={"color": GREY_B, "stroke_width": 2}).move_to([2.9, -0.4, 0])
        x_name = Text("input", font_size=22, color=BLUE).next_to(axes.x_axis.get_end(), RIGHT, buff=0.15)
        y_name = Text("output", font_size=22, color=TEAL).next_to(axes.y_axis.get_end(), RIGHT, buff=0.15)
        line = axes.plot(lambda x: 2 * x, x_range=[-2.45, 2.45], color=BLUE, stroke_width=5)
        formula = Text("f(x) = 2x", font_size=40).move_to([-3.9, 0.6, 0])
        self.at("linear")
        self.play(VGroup(diag, arrows).animate.scale(0.72).move_to([-3.9, 2.3, 0]),
                  Create(axes), FadeIn(x_name), FadeIn(y_name), FadeIn(formula), Create(line), run_time=0.7)

        def mark(x, y):
            dot = Dot(axes.c2p(x, y), color=YELLOW, radius=0.08)
            guides = VGroup(DashedLine(axes.c2p(x, 0), axes.c2p(x, y), color=GREY_B, stroke_width=2),
                            DashedLine(axes.c2p(0, y), axes.c2p(x, y), color=GREY_B, stroke_width=2))
            tx = Text(str(x), font_size=24, color=BLUE).next_to(axes.c2p(x, 0), DOWN, buff=0.15)
            ty = Text(str(y), font_size=24, color=TEAL).next_to(axes.c2p(0, y), LEFT, buff=0.15)
            return VGroup(guides, dot, tx, ty)

        m1, m2 = mark(1, 2), mark(2, 4)
        f1 = Text("f(1) = 2", font_size=32).move_to([-3.9, -0.4, 0])
        f2 = Text("f(2) = 4", font_size=32).move_to([-3.9, -1.2, 0])
        rule = Text("input ×2  →  output ×2", font_size=28, color=YELLOW).move_to([-3.9, -2.3, 0])
        self.at("double")
        self.play(FadeIn(m1), FadeIn(f1), run_time=0.5)
        self.at("output")
        self.play(FadeIn(m2), FadeIn(f2), run_time=0.5)
        self.at("too")
        self.play(FadeIn(rule, shift=0.15 * UP), run_time=0.4)

        o_dot = Dot(axes.c2p(0, 0), color=YELLOW, radius=0.09)
        ring = Circle(radius=0.25, color=YELLOW, stroke_width=3).move_to(o_dot)
        o_lab = Text("(0, 0)", font_size=24, color=YELLOW).next_to(o_dot, UL, buff=0.2)
        self.at("origin")
        self.play(line.animate.set_color(YELLOW).set_stroke(width=7), FadeIn(o_dot), Create(ring), FadeIn(o_lab),
                  run_time=0.7)
        self.end_section()

    # 2. Matrices are linear -------------------------------------------------------------------
    def s2_matrix(self):
        self.section(2)
        self.clear()
        wbox = box("W", w=1.3, h=1.0, fs=44)
        xin = Text("x", font_size=36, color=BLUE)
        xout = Text("Wx", font_size=36, color=TEAL)
        diag, arrows = chain([xin, wbox, xout], gap=1.0, center=0.4 * UP)
        self.at("matrix")
        self.play(FadeIn(wbox), FadeIn(xin), FadeIn(xout), *[GrowArrow(a) for a in arrows], run_time=0.6)

        def plane(cx):
            return NumberPlane(x_range=[-1, 3, 1], y_range=[-1, 3, 1], x_length=4.4, y_length=4.4,
                               background_line_style={"stroke_color": BLUE_E, "stroke_width": 1.5,
                                                      "stroke_opacity": 0.7},
                               axis_config={"stroke_color": GREY_B, "stroke_width": 2}).move_to([cx, -0.6, 0])

        p_in, p_out = plane(-3.9), plane(3.9)
        t_in = Text("input", font_size=28, color=GREY_B).next_to(p_in, UP, buff=0.25)
        t_out = Text("output", font_size=28, color=GREY_B).next_to(p_out, UP, buff=0.25)
        mid = VGroup(arrows[0].copy(), wbox.copy(), arrows[1].copy())
        mid.scale(0.8).move_to([0, -0.6, 0])
        s = ValueTracker(1.0)

        def vec(p, v, color, scale=None):
            k = s.get_value() if scale is None else scale
            return Arrow(p.c2p(0, 0), p.c2p(*(k * v)), buff=0, stroke_width=6, color=color,
                         max_tip_length_to_length_ratio=0.25, max_stroke_width_to_length_ratio=10)

        a_live = always_redraw(lambda: vec(p_in, VA, BLUE))
        wa_live = always_redraw(lambda: vec(p_out, W_EX @ VA, BLUE))
        a_lab = always_redraw(lambda: Text("a", font_size=28, color=BLUE)
                              .next_to(p_in.c2p(*(s.get_value() * VA)), RIGHT, buff=0.12))
        wa_lab = always_redraw(lambda: Text("Wa", font_size=28, color=BLUE)
                               .next_to(p_out.c2p(*(s.get_value() * (W_EX @ VA))), RIGHT, buff=0.12))
        self.at("scale")
        self.play(FadeOut(xin), FadeOut(xout),
                  ReplacementTransform(VGroup(arrows[0], wbox, arrows[1]), mid),
                  FadeIn(p_in), FadeIn(p_out), FadeIn(t_in), FadeIn(t_out), run_time=0.6)
        self.play(FadeIn(a_live), FadeIn(wa_live), FadeIn(a_lab), FadeIn(wa_lab), run_time=0.3)
        self.at("vector")
        self.play(s.animate.set_value(2.0), run_time=1.4)
        for m in (a_live, wa_live, a_lab, wa_lab):
            m.clear_updaters()
        a2_lab = Text("2a", font_size=28, color=BLUE).move_to(a_lab, aligned_edge=LEFT)
        wa2_lab = Text("2·Wa", font_size=28, color=BLUE).move_to(wa_lab, aligned_edge=LEFT)
        cap1 = Text("W(2a) = 2·Wa", font_size=34, color=YELLOW).move_to([0, -3.35, 0])
        self.at("same")
        self.play(FadeOut(a_lab), FadeOut(wa_lab), FadeIn(a2_lab), FadeIn(wa2_lab), FadeIn(cap1), run_time=0.4)

        # addition
        WA, WB = W_EX @ VA, W_EX @ VB
        va, vb = vec(p_in, VA, BLUE, 1), vec(p_in, VB, TEAL, 1)
        wva, wvb = vec(p_out, WA, BLUE, 1), vec(p_out, WB, TEAL, 1)
        la = Text("a", font_size=26, color=BLUE).next_to(p_in.c2p(*VA), DR, buff=0.06)
        lb = Text("b", font_size=26, color=TEAL).next_to(p_in.c2p(*VB), LEFT, buff=0.1)
        lwa = Text("Wa", font_size=26, color=BLUE).next_to(p_out.c2p(*WA), RIGHT, buff=0.1)
        lwb = Text("Wb", font_size=26, color=TEAL).next_to(p_out.c2p(*WB), LEFT, buff=0.1)
        self.at("add")
        self.play(FadeOut(a_live), FadeOut(wa_live), FadeOut(a2_lab), FadeOut(wa2_lab), FadeOut(cap1),
                  GrowArrow(va), GrowArrow(vb), GrowArrow(wva), GrowArrow(wvb),
                  FadeIn(la), FadeIn(lb), FadeIn(lwa), FadeIn(lwb), run_time=0.5)

        def summed(p, u, v, label):
            guides = VGroup(DashedLine(p.c2p(*u), p.c2p(*(u + v)), color=GREY_B, stroke_width=2),
                            DashedLine(p.c2p(*v), p.c2p(*(u + v)), color=GREY_B, stroke_width=2))
            total = vec(p, u + v, YELLOW, 1)
            lab = Text(label, font_size=26, color=YELLOW).next_to(p.c2p(*(u + v)), RIGHT, buff=0.12)
            return guides, total, lab

        g1, s1, l1 = summed(p_in, VA, VB, "a + b")
        g2, s2, l2 = summed(p_out, WA, WB, "Wa + Wb")
        self.play(Create(g1), GrowArrow(s1), FadeIn(l1), run_time=0.5)
        self.at("outputs")
        self.play(Create(g2), GrowArrow(s2), FadeIn(l2), run_time=0.5)
        cap2 = Text("W(a + b) = Wa + Wb", font_size=34, color=YELLOW).move_to([0, -3.35, 0])
        self.play(FadeIn(cap2, shift=0.15 * UP), run_time=0.4)
        self.end_section()

    # 3. Linear layers collapse --------------------------------------------------------------
    def s3_collapse(self):
        self.section(3)
        self.clear()
        xin = Text("x", font_size=36, color=BLUE)
        w1, w2 = box("W₁", w=1.3, h=1.0, fs=38), box("W₂", w=1.3, h=1.0, fs=38)
        yout = Text("y", font_size=36, color=TEAL)
        items, arrows = chain([xin, w1, w2, yout], gap=0.9, center=0.9 * UP)
        self.at("catch")
        self.play(FadeIn(items), *[GrowArrow(a) for a in arrows], run_time=0.8)

        merged = box("W = W₂W₁", color=TEAL, h=1.0, fs=36)
        items2, arrows2 = chain([xin.copy(), merged, yout.copy()], gap=0.9, center=0.9 * UP)
        self.at("single")
        self.play(FadeTransform(VGroup(w1, arrows[1], w2), merged),
                  xin.animate.move_to(items2[0]), yout.animate.move_to(items2[2]),
                  Transform(arrows[0], arrows2[0]), Transform(arrows[2], arrows2[1]), run_time=0.8)
        cap = Text("W₂(W₁x) = (W₂W₁)x", font_size=36, color=YELLOW).move_to([0, -0.9, 0])
        sub = Text("one matrix: their product", font_size=26, color=GREY_B).next_to(cap, DOWN, buff=0.3)
        self.at("product")
        self.play(FadeIn(cap), FadeIn(sub), run_time=0.5)

        labels = ["W₁", "W₂", "W₃", "⋮", "W₉₈", "W₉₉", "W₁₀₀"]
        st, a_in, a_out, x_lab, y_lab = stack(labels, center=[0, -0.15, 0], bw=1.6, bh=0.52, fs=28)
        brace = Brace(st, direction=RIGHT, color=GREY_B)
        b_lab = Text("100 linear layers", font_size=28).next_to(brace, RIGHT, buff=0.2)
        self.at("stack")
        self.play(FadeOut(VGroup(xin, yout, merged, arrows[0], arrows[2], cap, sub)),
                  LaggedStart(FadeIn(x_lab), GrowArrow(a_in), *[FadeIn(b, shift=0.1 * UP) for b in st],
                              GrowArrow(a_out), FadeIn(y_lab), lag_ratio=0.12),
                  run_time=1.0)
        self.at("layers")
        self.play(GrowFromCenter(brace), FadeIn(b_lab), run_time=0.4)

        single = box("W", color=TEAL, w=1.4, h=0.9, fs=36).move_to([0, -0.15, 0])
        n_in = arr(single.get_bottom() + 0.7 * DOWN, single.get_bottom())
        n_out = arr(single.get_top(), single.get_top() + 0.7 * UP)
        self.at("collapse")
        self.play(FadeTransform(st, single), FadeOut(brace), FadeOut(b_lab),
                  Transform(a_in, n_in), Transform(a_out, n_out),
                  x_lab.animate.next_to(n_in, DOWN, buff=0.08), y_lab.animate.next_to(n_out, UP, buff=0.08),
                  run_time=0.7)
        still = Text("still one matrix", font_size=34, color=YELLOW).next_to(single, RIGHT, buff=0.6)
        self.at("one")
        self.play(FadeIn(still, shift=0.15 * LEFT), run_time=0.4)
        self.end_section()

    # 4. What a linear map can't do ------------------------------------------------------------
    def s4_limits(self):
        self.section(4)
        self.clear()
        plane = NumberPlane(x_range=[-3, 3, 0.5], y_range=[-2, 2, 0.5], x_length=4.8, y_length=3.2,
                            background_line_style={"stroke_color": BLUE_D, "stroke_width": 2, "stroke_opacity": 0.8},
                            faded_line_ratio=1,
                            axis_config={"stroke_color": WHITE, "stroke_width": 2.5}).move_to([-2.6, -0.2, 0])
        cell = Polygon(*[plane.c2p(x, y) for x, y in [(0, 0), (1, 0), (1, 1), (0, 1)]], stroke_color=TEAL,
                       stroke_width=3, fill_color=TEAL, fill_opacity=0.45)
        grid = VGroup(plane, cell)
        title = Text("one linear map", font_size=30, color=GREY_B).move_to([-2.6, 3.25, 0])
        self.at("linear")
        self.play(Create(grid), FadeIn(title), run_time=0.8)

        def item(mark, color, text):
            return VGroup(Text(mark, font_size=36, color=color), Text(text, font_size=32)).arrange(RIGHT, buff=0.25)

        i1, i2, i3 = item("✓", GREEN, "stretch"), item("✓", GREEN, "rotate"), item("✗", RED, "can't bend")
        VGroup(i1, i2, i3).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to([3.9, 0, 0])
        c = plane.get_center()
        self.at("stretch")
        self.play(ApplyMatrix([[1.35, 0], [0, 0.8]], grid, about_point=c), FadeIn(i1), run_time=0.6)
        a = 30 * DEGREES
        self.at("rotate")
        self.play(ApplyMatrix([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], grid, about_point=c),
                  FadeIn(i2), run_time=0.7)
        self.at("bend")
        self.play(FadeIn(i3, shift=0.1 * UP), run_time=0.35)
        self.play(Indicate(i3[0], color=RED, scale_factor=1.3), run_time=0.4)

        # inside vs outside a circle
        self.at("try")
        self.clear(run_time=0.5)
        rng = np.random.default_rng(4)
        center = np.array([-1.2, -0.2, 0])
        unit = 1.0

        def ring_points(n, r0, r1):
            ang = rng.uniform(0, TAU, n)
            rad = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, n))
            return [center + unit * np.array([r * np.cos(t), r * np.sin(t), 0]) for r, t in zip(rad, ang)]

        inner = VGroup(*[Dot(p, radius=0.08, color=BLUE) for p in ring_points(22, 0.0, 1.15)])
        outer = VGroup(*[Dot(p, radius=0.08, color=ORANGE) for p in ring_points(38, 1.8, 2.8)])
        circ = DashedVMobject(Circle(radius=1.47, color=GREY_B, stroke_width=3), num_dashes=40).move_to(center)
        note = illustrative()
        self.at("inside")
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in inner], lag_ratio=0.05), FadeIn(note), run_time=0.5)
        self.at("circle")
        self.play(Create(circ), run_time=0.5)
        self.at("outside")
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in outer], lag_ratio=0.03), run_time=0.5)

        t = ValueTracker(0.0)

        def sweep_line():
            k = t.get_value()
            theta = (105 - 60 * k) * DEGREES
            d = -1.6 + 3.2 * k
            direction = np.array([np.cos(theta), np.sin(theta), 0])
            normal = np.array([np.sin(theta), -np.cos(theta), 0])
            mid = center + d * normal
            return Line(mid - 2.5 * direction, mid + 2.5 * direction, color=WHITE, stroke_width=5)

        cut = always_redraw(sweep_line)
        self.at("straight")
        self.add(cut)
        self.play(t.animate.set_value(1.0), run_time=1.1, rate_func=smooth)
        cut.clear_updaters()
        cross = Text("✗", font_size=80, color=RED).move_to([4.2, 0.6, 0])
        why = Text("no straight line\nseparates them", font_size=28, color=RED, line_spacing=0.8)
        why.next_to(cross, DOWN, buff=0.3)
        self.at("cant")
        self.play(FadeIn(cross, scale=1.4), FadeIn(why), run_time=0.4)
        self.end_section()

    # 5. The fix: a bend -------------------------------------------------------------------------
    def s5_fix(self):
        self.section(5)
        self.clear()
        xin = Text("x", font_size=36, color=BLUE)
        w1, w2 = box("W₁", w=1.2, h=0.9, fs=36), box("W₂", w=1.2, h=0.9, fs=36)
        bend = box("bend", color=BEND_COLOR, h=0.9, fs=34)
        yout = Text("y", font_size=36, color=TEAL)
        items, arrows = chain([xin, w1, bend, w2, yout], gap=0.8, center=1.3 * UP)
        self.at("bend")
        self.play(FadeIn(items), *[GrowArrow(a) for a in arrows], run_time=0.7)
        nl = Text("nonlinear function", font_size=28, color=BEND_COLOR).next_to(bend, UP, buff=0.35)
        self.at("nonlinear")
        self.play(FadeIn(nl, shift=0.1 * DOWN), run_time=0.4)

        cells = VGroup(*[Text(f"{v:g}".replace("-", "−"), font_size=36) for v in RELU_IN])
        for k, c in enumerate(cells):
            c.move_to([(k - 2) * 1.2 + bend.get_x(), -1.3, 0])
        br = bracket_pair(cells.get_left()[0] - 0.35, cells.get_right()[0] + 0.35, -0.95, -1.65)
        vec_g = VGroup(br, cells)
        note = illustrative()
        self.at("applied")
        self.play(FadeIn(vec_g), FadeIn(note), run_time=0.4)
        fan = VGroup(*[
            arr(bend.get_bottom() + 0.18 * (c.get_x() - bend.get_x()) * RIGHT, c.get_top() + 0.12 * UP,
                color=BEND_COLOR, sw=3)
            for c in cells])
        self.at("every")
        self.play(LaggedStart(*[GrowArrow(a) for a in fan], lag_ratio=0.25), run_time=0.9)
        self.end_section()

    # 6. ReLU ----------------------------------------------------------------------------------
    def s6_relu(self):
        self.section(6)
        self.clear()
        axes = Axes(x_range=[-3, 3, 1], y_range=[-3, 3, 1], x_length=4.8, y_length=4.8, tips=True,
                    axis_config={"color": GREY_B, "stroke_width": 2}).move_to([0, -0.4, 0])
        left = Line(axes.c2p(-2.8, 0), axes.c2p(0, 0), color=BLUE, stroke_width=6)
        right = Line(axes.c2p(0, 0), axes.c2p(2.8, 2.8), color=BLUE, stroke_width=6)
        title = Text("ReLU(x) = max(0, x)", font_size=36).next_to(axes, UP, buff=0.25)
        self.at("relu")
        self.play(Create(axes), FadeIn(title), run_time=0.4)
        self.play(Create(left), Create(right), run_time=0.5)

        p_dot = Dot(axes.c2p(2, 2), color=GREEN, radius=0.09)
        p_lab = Text("2 → 2", font_size=28, color=GREEN).next_to(p_dot, RIGHT, buff=0.2)
        self.at("positive")
        self.play(right.animate.set_color(GREEN), FadeIn(p_dot), FadeIn(p_lab), run_time=0.5)

        ghost = DashedLine(axes.c2p(-2.8, -2.8), axes.c2p(0, 0), color=GREY_B, stroke_width=3)
        n_dot = Dot(axes.c2p(-2, -2), color=RED, radius=0.09)
        self.at("negative")
        self.play(Create(ghost), FadeIn(n_dot), run_time=0.4)
        flat = DashedLine(axes.c2p(-2.8, 0), axes.c2p(0, 0), color=GREY_B, stroke_width=3)
        n_lab = Text("−2 → 0", font_size=28, color=RED).next_to(axes.c2p(-2, 0), UP, buff=0.3)
        self.at("zero")
        self.play(Transform(ghost, flat), n_dot.animate.move_to(axes.c2p(-2, 0)), left.animate.set_color(RED),
                  FadeIn(n_lab), run_time=0.6)
        self.play(FadeOut(ghost), run_time=0.2)

        relu_g = VGroup(axes, left, right, title, p_dot, p_lab, n_dot, n_lab)
        self.at("tiny")
        self.play(relu_g.animate.shift(3.3 * LEFT), run_time=0.6)

        labels = ["W₁", "ReLU", "W₂", "ReLU", "W₃", "⋮", "ReLU", "W₁₀₀"]
        st, a_in, a_out, x_lab, y_lab = stack(labels, center=[3.1, -0.15, 0], bw=1.5, bh=0.5, gap=0.09, fs=26)
        self.at("stacked")
        self.play(LaggedStart(FadeIn(x_lab), GrowArrow(a_in), *[FadeIn(b, shift=0.1 * UP) for b in st],
                              GrowArrow(a_out), FadeIn(y_lab), lag_ratio=0.1), run_time=0.9)
        ok = VGroup(Text("✓", font_size=56, color=GREEN), Text("no collapse", font_size=28, color=GREEN))
        ok.arrange(DOWN, buff=0.15).move_to([5.45, -0.2, 0])
        cy = st.get_y()
        self.at("collapse")
        self.play(*[b.animate(rate_func=there_and_back).shift(0.35 * (cy - b.get_y()) * UP) for b in st],
                  FadeIn(ok, scale=1.2), run_time=0.6)
        self.end_section()

    # 7. GELU ----------------------------------------------------------------------------------
    def s7_gelu(self):
        self.section(7)
        self.clear()
        axes = Axes(x_range=[-4, 4, 1], y_range=[-1, 4, 1], x_length=9.6, y_length=5.5, tips=True,
                    axis_config={"color": GREY_B, "stroke_width": 2}).move_to([0, -0.55, 0])
        ticks = VGroup(Text("−3", font_size=22, color=GREY_B).next_to(axes.c2p(-3, 0), DOWN, buff=0.15),
                       Text("3", font_size=22, color=GREY_B).next_to(axes.c2p(3, 0), DOWN, buff=0.15))
        relu_ghost = DashedVMobject(VMobject(stroke_color=GREY_B, stroke_width=3).set_points_as_corners(
            [axes.c2p(-4, 0), axes.c2p(0, 0), axes.c2p(3.95, 3.95)]), num_dashes=50).set_stroke(opacity=0.6)
        curve = axes.plot(gelu, x_range=[-4, 4], color=TEAL, stroke_width=6)
        title = Text("GELU: a smooth ReLU", font_size=36).move_to([0, 3.2, 0])
        legend = VGroup(
            VGroup(Line(ORIGIN, 0.5 * RIGHT, color=TEAL, stroke_width=6), Text("GELU", font_size=24)).arrange(RIGHT),
            VGroup(DashedLine(ORIGIN, 0.5 * RIGHT, color=GREY_B, stroke_width=3),
                   Text("ReLU", font_size=24, color=GREY_B)).arrange(RIGHT))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([-5.4, 2.4, 0])
        self.at("jell")
        self.play(Create(axes), FadeIn(ticks), FadeIn(relu_ghost), FadeIn(title), FadeIn(legend), run_time=0.4)
        self.play(Create(curve), run_time=0.45)

        o = axes.c2p(0, 0.15)
        ring = Ellipse(width=2.4, height=1.5, color=YELLOW, stroke_width=4).move_to(o)
        s_lab = Text("smooth: no corner", font_size=28, color=YELLOW).move_to([-3.3, -1.0, 0])
        self.at("smooth")
        self.play(Create(ring), FadeIn(s_lab), run_time=0.5)

        pos_hl = axes.plot(gelu, x_range=[1.6, 4], color=GREEN, stroke_width=9)
        self.at("positive")
        self.play(FadeOut(ring), FadeOut(s_lab), Create(pos_hl), run_time=0.4)
        p_dot = Dot(axes.c2p(3, gelu(3)), color=GREEN, radius=0.08)
        p_lab = VGroup(Text("large positive:", font_size=26, color=GREEN),
                       Text("passes through", font_size=26, color=GREEN),
                       Text("gelu(3) ≈ 2.996", font_size=26)).arrange(DOWN, buff=0.15).move_to([4.6, -1.25, 0])
        self.at("pass")
        self.play(FadeIn(p_dot), FadeIn(p_lab), run_time=0.4)

        neg_hl = axes.plot(gelu, x_range=[-4, -1.6], color=RED, stroke_width=9)
        self.at("negative")
        self.play(Create(neg_hl), run_time=0.4)
        n_dot = Dot(axes.c2p(-3, gelu(-3)), color=RED, radius=0.08)
        n_lab = VGroup(Text("large negative:", font_size=26, color=RED),
                       Text("fades to 0", font_size=26, color=RED),
                       Text("gelu(−3) ≈ −0.004", font_size=26)).arrange(DOWN, buff=0.15).move_to([-3.6, 0.2, 0])
        self.at("fade")
        self.play(FadeIn(n_dot), FadeIn(n_lab), run_time=0.4)

        dip = axes.c2p(GELU_MIN_X, gelu(GELU_MIN_X))
        d_dot = Dot(dip, color=YELLOW, radius=0.08)
        d_lab = Text("small dip: min ≈ −0.17", font_size=26, color=YELLOW).move_to([-3.4, -3.2, 0])
        d_arrow = Arrow(d_lab.get_right() + 0.1 * RIGHT + 0.1 * UP, dip + 0.08 * DL, buff=0.05, color=YELLOW,
                        stroke_width=3, max_tip_length_to_length_ratio=0.25)
        self.at("curve")
        self.play(FadeIn(d_dot), FadeIn(d_lab), GrowArrow(d_arrow), run_time=0.5)
        self.end_section()

    # 8. Many simple pieces --------------------------------------------------------------------
    def s8_approx(self):
        self.section(8)
        self.clear()
        axes = Axes(x_range=[-3.2, 3.2, 1], y_range=[-1.6, 1.6, 1], x_length=10, y_length=4.4, tips=False,
                    axis_config={"color": GREY_B, "stroke_width": 2}).move_to([0, 0.75, 0])
        self.at("layers")
        self.play(Create(axes), run_time=0.5)
        tgt = axes.plot(target, x_range=[-3, 3], color=BLUE, stroke_width=7).set_stroke(opacity=0.8)
        self.at("curved")
        self.play(Create(tgt), run_time=0.7)

        grid = np.linspace(-3, 3, 241)

        def approx(knots):
            return polyline(axes, grid, np.interp(grid, knots, target(np.array(knots))), color=YELLOW, sw=4)

        def knot_dots(knots):
            return VGroup(*[Dot(axes.c2p(k, target(k)), radius=0.06, color=YELLOW) for k in knots])

        row_lab = Text("ReLU pieces:", font_size=26, color=GREY_B)
        row_lab.move_to([-4.9, -2.25, 0])
        slots = [np.array([-3.35 + 0.9 * i, -2.25, 0]) for i in range(len(KNOT_SETS))]
        icons = [hinge_icon(flip=(i % 2 == 1)).move_to(p) for i, p in enumerate(slots)]
        pluses = [Text("+", font_size=26, color=GREY_B).move_to((slots[i] + slots[i + 1]) / 2)
                  for i in range(len(slots) - 1)]

        cur = approx(KNOT_SETS[0])
        dots = knot_dots(KNOT_SETS[0])
        self.at("many")
        self.play(Create(cur), FadeIn(dots), FadeIn(row_lab), FadeIn(icons[0]), run_time=0.3)
        for i in range(1, len(KNOT_SETS)):
            new_dots = knot_dots(KNOT_SETS[i])
            self.play(Transform(cur, approx(KNOT_SETS[i])), Transform(dots, new_dots),
                      FadeIn(pluses[i - 1]), FadeIn(icons[i], scale=0.6), run_time=0.3)
        cap = Text("approximate almost any function", font_size=34, color=YELLOW).move_to([0, -3.25, 0])
        self.at("function")
        self.play(FadeIn(cap, shift=0.15 * UP), run_time=0.4)
        self.end_section()

    # 9. Code ------------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, font_size=26)
        code.move_to(ORIGIN)
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[1])
        self.clear(run_time=0.35)
        self.play(FadeIn(code, shift=0.3 * UP), run_time=0.4)
        self.at("relu")
        self.play(Create(hl), run_time=0.4)
        self.at("line")
        self.play(Indicate(hl, color=YELLOW, scale_factor=1.03), run_time=0.5)
        self.at("multiplication")
        self.play(highlight(hl, code, 5), run_time=0.4)
        self.at("bend")
        self.play(Indicate(hl, color=YELLOW, scale_factor=1.03), run_time=0.5)
        self.at("next")
        self.play(highlight(hl, code, 6), run_time=0.4)
        self.end_section()

    # 10. Outro ------------------------------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        self.clear()
        card = used_in_card(USED_IN)
        self.at("episode")
        self.play(FadeIn(card), run_time=0.34)
        self.at("8")
        self.play(Indicate(card[1][0], color=YELLOW, scale_factor=1.08), run_time=0.8)
        self.at("9")
        self.play(Indicate(card[1][1], color=YELLOW, scale_factor=1.08), run_time=0.8)
        nxt = next_up_card(NEXT)
        self.at("logarithms")
        self.play(FadeOut(card), run_time=0.3)
        self.play(FadeIn(nxt, shift=0.2 * UP), run_time=0.4)
        self.end_section()
