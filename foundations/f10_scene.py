"""Foundations F10 — Slopes and Gradients.

Render from the repo root:  ./render.sh foundations f10
"""
import numpy as np
from manim import *

from common import code_panel, finish, highlight, next_up_card, used_in_card
from f10_script import LABEL, NEXT, TAGLINE, TITLE, USED_IN
from intro import play_token_intro
from voiced_scene import VoicedScene

CURVE = BLUE_C      # curves
TANGENT = GREEN_C   # tangent lines / slopes
BALL = YELLOW       # the ball rolling downhill
LAYER = PURPLE_B    # function / layer boxes

# ---------------------------------------------------------------------------- verified numbers
DX_TINY = 0.001
DY_TINY = 3.001 ** 2 - 9                      # 0.006001
assert 3 ** 2 == 9
assert abs(DY_TINY - 0.006001) < 1e-12
assert round(DY_TINY / DX_TINY, 3) == 6.001    # ≈ 6 = 2 · 3

LR = 0.1
PATH = [3.0]
for _ in range(5):
    PATH.append(PATH[-1] - LR * 2 * PATH[-1])  # x ← x − lr · 2x = 0.8 x
PATH_TXT = ["3", "2.4", "1.92", "1.536", "1.229", "0.983"]
assert [f"{round(v, 3):g}" for v in PATH] == PATH_TXT
assert abs(3 - 0.1 * 6 - 2.4) < 1e-12

# chain rule example (illustrative slopes)
DYDX, DZDY = 3, 2
assert DYDX * DZDY == 6

# 2-D bowl for section 6 (illustrative): L = 0.5·w1² + 2·w2², gradient (w1, 4·w2)
W_START = np.array([2.4, 1.0])
LR2 = 0.2
W_PATH = [W_START]
for _ in range(5):
    w = W_PATH[-1]
    W_PATH.append(w - LR2 * np.array([w[0], 4 * w[1]]))


def bowl(w):
    return 0.5 * w[0] ** 2 + 2 * w[1] ** 2


CODE = """slope = lambda x: 2 * x          # derivative of x ** 2

x, lr = 3.0, 0.1
for step in range(5):
    x = x - lr * slope(x)        # step against the slope
    print(round(x, 3))           # 2.4, 1.92, 1.536, 1.229, 0.983"""


# ---------------------------------------------------------------------------- helpers
def label(text, color=WHITE, font_size=28, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def caption(text, font_size=20):
    return Text(text, font_size=font_size, color=GREY_B).to_corner(DR, buff=0.3)


def make_axes(x_range, y_range, x_length, y_length, center):
    ax = Axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, tips=True,
              axis_config={"color": GREY_B, "stroke_width": 2, "tick_size": 0.06,
                           "tip_width": 0.18, "tip_height": 0.18})
    return ax.move_to(center)


def x_ticks(ax, values, font_size=22):
    return VGroup(*[label(str(v), GREY_B, font_size).next_to(ax.c2p(v, ax.y_range[0]), DOWN, buff=0.15)
                    for v in values])


def y_ticks(ax, values, font_size=22):
    return VGroup(*[label(str(v), GREY_B, font_size).next_to(ax.c2p(ax.x_range[0], v), LEFT, buff=0.15)
                    for v in values])


def tangent(ax, f, df, x0, half, color=TANGENT, stroke_width=5):
    return Line(ax.c2p(x0 - half, f(x0) - half * df(x0)), ax.c2p(x0 + half, f(x0) + half * df(x0)),
                color=color, stroke_width=stroke_width)


def fbox(text, color=LAYER, w=1.3, h=1.0, font_size=40):
    box = RoundedRectangle(corner_radius=0.15, width=w, height=h, stroke_color=color,
                           fill_color=color, fill_opacity=0.2)
    return VGroup(box, label(text, WHITE, font_size).move_to(box))


def node(text, color=BLUE_C):
    c = Circle(radius=0.42, stroke_color=color, fill_color=color, fill_opacity=0.2)
    return VGroup(c, label(text, WHITE, 36).move_to(c))


class GradientsVideo(VoicedScene):
    VIDEO = "f10"

    # ------------------------------------------------------------------ stage helpers
    def clear_anims(self, keep=()):
        anims = []
        for m in list(self.mobjects):
            if m in keep:
                continue
            if isinstance(m, ValueTracker):
                self.remove(m)
                continue
            m.clear_updaters()
            anims.append(FadeOut(m))
        return anims

    def clear(self, run_time=0.4, keep=()):
        anims = self.clear_anims(keep)
        if anims:
            self.play(*anims, run_time=run_time)

    def construct(self):
        play_token_intro(self, TITLE, 0, TAGLINE, label=LABEL)
        self.s1_loss()
        self.s2_nudge()
        self.s3_derivative()
        self.s4_downhill()
        self.s5_descent()
        self.s6_gradient()
        self.s7_chain()
        self.s8_backprop()
        self.s9_code()
        self.s10_outro()
        self.end_section()
        finish(self)

    # 1. A loss curve and a ball on it -------------------------------------------------------------
    def s1_loss(self):
        self.section(1)
        head = VGroup(label("training:", GREY_B, 32),
                      label("adjust the weights to reduce the loss", WHITE, 32)).arrange(RIGHT, buff=0.25)
        head.move_to([0, 3.1, 0])
        self.play(FadeIn(head, shift=0.2 * DOWN), run_time=0.6)

        def f(w):
            return 0.28 * (w - 3.2) ** 2 + 0.6

        def df(w):
            return 0.56 * (w - 3.2)

        ax = make_axes([0, 8, 1], [0, 6, 1], 8.6, 4.8, [0, -0.7, 0])
        curve = ax.plot(f, x_range=[0, 7.5], color=CURVE, stroke_width=5)
        xl = label("weight", GREY_B, 24).next_to(ax.x_axis.get_end(), DOWN, buff=0.2)
        yl = label("loss", GREY_B, 24).next_to(ax.y_axis.get_end(), RIGHT, buff=0.2)
        w0 = 6.0
        ball = Dot(ax.c2p(w0, f(w0)), radius=0.15, color=BALL).set_z_index(3)
        self.at("loss")
        self.play(Create(ax), Create(curve), FadeIn(xl), FadeIn(yl), FadeIn(ball, scale=0.5), run_time=0.9)

        p = ball.get_center()
        ask = VGroup(Arrow(p + 0.2 * LEFT, p + 1.3 * LEFT, buff=0, color=GREY_B, stroke_width=4),
                     Arrow(p + 0.2 * RIGHT, p + 1.3 * RIGHT, buff=0, color=GREY_B, stroke_width=4),
                     label("which way?", GREY_B, 26).move_to(p + np.array([-0.95, 0.5, 0])))
        self.at("which")
        self.play(FadeIn(ask), run_time=0.5)

        tan = tangent(ax, f, df, w0, 1.3)
        slope_lab = label("slope", TANGENT, 30).next_to(tan.get_start(), RIGHT, buff=0.25)
        self.at("slope")
        self.play(FadeOut(ask), Create(tan), FadeIn(slope_lab), run_time=0.6)
        self.end_section()

    # 2. Slope = how much the output changes for a nudge -------------------------------------------
    def s2_nudge(self):
        self.section(2)
        self.clear()

        def f(x):
            return 4 * (1 - np.exp(-0.9 * x))

        ax = make_axes([0, 6, 1], [0, 4.5, 1], 7.2, 4.8, [-2.6, -0.6, 0])
        curve = ax.plot(f, x_range=[0, 6], color=CURVE, stroke_width=5)
        xl = label("input", GREY_B, 24).next_to(ax.x_axis.get_end(), DOWN, buff=0.2)
        yl = label("output", GREY_B, 24).next_to(ax.y_axis.get_end(), RIGHT, buff=0.2)
        head = label("the slope of a curve", WHITE, 36).move_to([0, 3.1, 0])
        self.play(Create(ax), Create(curve), FadeIn(xl), FadeIn(yl), FadeIn(head, shift=0.2 * DOWN), run_time=0.7)

        DX = 0.8
        xt = ValueTracker(1.6)

        def tri():
            x0 = xt.get_value()
            p0, p1, p2 = ax.c2p(x0, f(x0)), ax.c2p(x0 + DX, f(x0)), ax.c2p(x0 + DX, f(x0 + DX))
            run = Line(p0, p1, color=WHITE, stroke_width=4)
            rise = Line(p1, p2, color=YELLOW, stroke_width=6)
            lx = label("Δx", WHITE, 24).next_to(run, DOWN, buff=0.12)
            ly = label("Δy", YELLOW, 24).next_to(rise, RIGHT, buff=0.12)
            ly.set_y(min((p1[1] + p2[1]) / 2, p1[1] - 0.22))   # below the curve when the rise is tiny
            return VGroup(run, rise, lx, ly, Dot(p0, radius=0.06), Dot(p2, radius=0.06, color=YELLOW))

        triangle = always_redraw(tri)
        px = 4.3
        rule = label("slope = Δy / Δx", WHITE, 34).move_to([px, 1.7, 0])
        l1 = label("Δx: nudge the input", GREY_B, 24)
        l2 = label("Δy: change in the output", YELLOW, 24)
        legend = VGroup(l1, l2).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(rule, DOWN, buff=0.35)
        self.at("nudge")
        self.play(FadeIn(triangle), FadeIn(rule), FadeIn(legend), run_time=0.8)

        steep = VGroup(label("steep", WHITE, 30), label("→ big Δy", YELLOW, 30)).arrange(RIGHT, buff=0.2)
        steep.move_to([px, -0.9, 0])
        self.at("steep")
        self.play(xt.animate.set_value(0.15), FadeIn(steep, shift=0.2 * UP), run_time=0.8)

        flat = VGroup(label("flat", WHITE, 30), label("→ tiny Δy", YELLOW, 30)).arrange(RIGHT, buff=0.2)
        flat.move_to([px, -1.7, 0]).align_to(steep, LEFT)
        self.at("flat")
        self.play(xt.animate.set_value(4.6), FadeIn(flat, shift=0.2 * UP), run_time=0.9)
        self.end_section()

    # 3. The derivative of x² -------------------------------------------------------------------
    def s3_derivative(self):
        self.section(3)
        self.clear()

        def f(x):
            return x ** 2

        def df(x):
            return 2 * x

        ax = make_axes([0, 4, 1], [0, 16, 4], 4.8, 5.4, [-3.3, -0.4, 0])
        curve = ax.plot(f, x_range=[0, 4], color=CURVE, stroke_width=5)
        ticks = VGroup(x_ticks(ax, [1, 2, 3, 4]), y_ticks(ax, [4, 8, 12, 16]))
        head = label("y = x²", WHITE, 36).move_to([ax.c2p(2, 0)[0], 3.1, 0])
        self.at("squared")
        self.play(Create(ax), Create(curve), FadeIn(ticks), FadeIn(head, shift=0.2 * DOWN), run_time=0.8)

        P = ax.c2p(3, 9)
        dot = Dot(P, radius=0.09, color=BALL).set_z_index(3)
        guides = VGroup(DashedLine(ax.c2p(3, 0), P, color=GREY_B, stroke_width=2),
                        DashedLine(ax.c2p(0, 9), P, color=GREY_B, stroke_width=2))
        pt_lab = label("(3, 9)", WHITE, 26).move_to(ax.c2p(1.9, 10.6))
        self.at("3")
        self.play(FadeIn(dot, scale=0.5), Create(guides), FadeIn(pt_lab), run_time=0.6)

        # a ×1000 zoom around (3, 9): same aspect as the main plot, so the curve looks like its own tangent
        sx = ax.x_axis.get_unit_size()
        sy = ax.y_axis.get_unit_size()
        frame = Rectangle(width=4.8, height=3.2, color=GREY_B, stroke_width=2).move_to([3.5, 1.25, 0])
        o = frame.get_center() + np.array([-0.5 * sx, -3.0 * sy, 0])   # where (3, 9) sits in the inset

        def z(u, v):  # inset point for x = 3 + u·0.001, y = 9 + v·0.001
            return o + np.array([u * sx, v * sy, 0])

        zcurve = ParametricFunction(lambda u: z(u, 6 * u + 0.001 * u * u), t_range=[-0.2, 1.2],
                                    color=CURVE, stroke_width=5)
        zdot = Dot(z(0, 0), radius=0.09, color=BALL).set_z_index(3)
        zoom_tag = label("zoomed in ×1000", GREY_B, 20)
        zoom_tag.move_to(frame.get_corner(UL) + np.array([0.2, -0.3, 0]), aligned_edge=LEFT)
        spot = Square(side_length=0.36, color=GREY_B, stroke_width=2).move_to(P)
        links = VGroup(Line(spot.get_corner(UR), frame.get_corner(UL), color=GREY_D, stroke_width=2),
                       Line(spot.get_corner(DR), frame.get_corner(DL), color=GREY_D, stroke_width=2))
        self.at("nudge")
        self.play(Create(spot), Create(links), Create(frame), Create(zcurve), FadeIn(zdot), FadeIn(zoom_tag),
                  run_time=0.8)

        run = Line(z(0, 0), z(1, 0), color=WHITE, stroke_width=4)
        rise = Line(z(1, 0), z(1, 6.001), color=YELLOW, stroke_width=6)
        zlabs = VGroup(label("Δx", WHITE, 24).next_to(run, DOWN, buff=0.12),
                       label("Δy", YELLOW, 24).next_to(rise, RIGHT, buff=0.12))
        cx, vx = 0.8, 3.95
        rows = VGroup(
            VGroup(label("x: 3 → 3.001", WHITE, 24), label("Δx = 0.001", WHITE, 24)),
            VGroup(label("y: 9 → 9.006001", WHITE, 24), label("Δy = 0.006001", YELLOW, 24)),
        )
        for k, r in enumerate(rows):
            r[0].move_to([cx, -0.85 - 0.6 * k, 0], aligned_edge=LEFT)
            r[1].move_to([vx, -0.85 - 0.6 * k, 0], aligned_edge=LEFT)
        self.at("tiny")
        self.play(Create(run), Create(rise), FadeIn(zlabs), FadeIn(rows), run_time=0.8)

        ratio = label("Δy / Δx = 6.001 ≈ 6", YELLOW, 32).move_to([frame.get_x(), -2.35, 0])
        self.at("six")
        self.play(FadeIn(ratio, shift=0.2 * UP), run_time=0.5)

        tan = tangent(ax, f, df, 3, 1.05)
        slope_lab = label("slope = 6", TANGENT, 30).move_to(ax.c2p(1.9, 14.2))
        self.at("6")
        self.play(FadeOut(links), FadeOut(spot), Create(tan), FadeIn(slope_lab), run_time=0.7)

        inset = VGroup(frame, zcurve, zdot, zoom_tag, run, rise, zlabs, rows, ratio)
        deriv = VGroup(label("derivative of x²", WHITE, 36), label("= 2x", YELLOW, 36)).arrange(RIGHT, buff=0.25)
        deriv.move_to([3.4, 1.0, 0])
        check = label("at x = 3:   2 × 3 = 6", GREY_A, 30).next_to(deriv, DOWN, buff=0.6)
        self.at("derivative")
        self.at("derivative")
        self.play(FadeOut(inset), FadeIn(deriv, shift=0.2 * UP), run_time=0.6)
        self.at("2x")
        self.play(FadeIn(check, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # 4. The slope tells you which way is downhill ------------------------------------------------
    def s4_downhill(self):
        self.section(4)
        self.clear()
        ax = make_axes([-4, 4, 1], [0, 16, 4], 7.0, 4.4, [-1.8, -0.9, 0])
        self.ax = ax
        curve = ax.plot(lambda x: x ** 2, x_range=[-4, 4], color=CURVE, stroke_width=5)
        ticks = x_ticks(ax, [-3, -2, -1, 0, 1, 2, 3])
        name = label("y = x²", GREY_B, 26).next_to(ax.c2p(-3.6, 12.96), LEFT, buff=0.3)
        self.xb = ValueTracker(3.0)
        ball = Dot(ax.c2p(3, 9), radius=0.15, color=BALL).set_z_index(3)
        ball.add_updater(lambda m: m.move_to(ax.c2p(self.xb.get_value(), self.xb.get_value() ** 2)))
        self.ball = ball
        self.play(Create(ax.x_axis), Create(curve), FadeIn(ticks), FadeIn(name), FadeIn(ball), run_time=0.7)

        tan = tangent(ax, lambda x: x ** 2, lambda x: 2 * x, 3, 0.75)
        self.at("downhill")
        self.play(Create(tan), run_time=0.6)
        self.at("3")
        self.play(Indicate(ticks[6], scale_factor=1.5), Flash(ball.get_center(), color=BALL, flash_radius=0.3),
                  run_time=0.6)

        pos = label("slope > 0", TANGENT, 30).move_to([2.9, -0.35, 0])
        self.at("positive")
        self.play(FadeIn(pos, shift=0.2 * LEFT), run_time=0.5)

        p = ball.get_center()
        arrow = Arrow(p + 0.2 * LEFT, p + 1.9 * LEFT, buff=0, color=YELLOW, stroke_width=6)
        step = label("step against the slope", YELLOW, 26)
        step.next_to(arrow, UP, buff=0.2).align_to(arrow, RIGHT).shift(0.2 * LEFT)
        self.at("left")
        self.play(GrowArrow(arrow), FadeIn(step, shift=0.2 * LEFT), run_time=0.6)
        self.s4_extra = VGroup(tan, pos, arrow, step)
        self.end_section()

    # 5. Gradient descent in one dimension --------------------------------------------------------
    def s5_descent(self):
        self.section(5)
        ax, xb, ball = self.ax, self.xb, self.ball
        head = label("gradient descent", WHITE, 36).move_to([0, 3.3, 0])
        self.play(FadeOut(self.s4_extra), FadeIn(head, shift=0.2 * DOWN), run_time=0.5)

        rule = VGroup(label("x", WHITE, 32), label("←", WHITE, 32), label("x", WHITE, 32), label("−", WHITE, 32),
                      label("learning rate", YELLOW, 32), label("×", WHITE, 32), label("slope", TANGENT, 32))
        rule.arrange(RIGHT, buff=0.2).move_to([0, 2.5, 0])
        self.at("new")
        self.play(FadeIn(rule, shift=0.2 * DOWN), run_time=0.6)

        lr = VGroup(label("learning rate", YELLOW, 28), label("= 0.1", WHITE, 28)).arrange(RIGHT, buff=0.2)
        lr.move_to([-3.3, 1.75, 0])
        self.at("1")
        self.play(FadeIn(lr, shift=0.2 * UP), run_time=0.5)

        vx, vy0, vdy = 5.3, 0.9, 0.45
        vals = VGroup(*[label(("x = " if k == 0 else "") + t, WHITE if k == 0 else GREY_A, 28)
                        for k, t in enumerate(PATH_TXT)])
        for k, v in enumerate(vals):
            v.move_to([vx, vy0 - k * vdy, 0])
            if k > 0:
                v.align_to(vals[0][2], LEFT)
        first = label("3 − 0.1 × 6 = 2.4", WHITE, 28).move_to([6.5, 1.6, 0], aligned_edge=RIGHT)
        trail = VGroup()

        def step_to(k):
            d = Dot(ax.c2p(PATH[k - 1], PATH[k - 1] ** 2), radius=0.07, color=BALL, fill_opacity=0.5)
            trail.add(d)
            self.add(d)
            return [xb.animate.set_value(PATH[k]), FadeIn(vals[k], shift=0.15 * LEFT)]

        self.at("4")
        self.add(trail)
        self.play(FadeIn(vals[0]), FadeIn(first, shift=0.2 * UP), *step_to(1), run_time=0.7)

        self.at("repeat")
        for k in range(2, 6):
            self.play(*step_to(k), run_time=0.4)

        mn = Dot(ax.c2p(0, 0), radius=0.11, color=GREEN)
        mn_lab = label("minimum", GREEN, 26).next_to(ax.c2p(0, 0), UP, buff=0.8)
        self.at("zero")
        self.play(FadeIn(mn, scale=0.5), FadeIn(mn_lab, shift=0.2 * UP), run_time=0.5)
        ball.clear_updaters()
        self.end_section()

    # 6. Many weights: the gradient vector ---------------------------------------------------------
    def s6_gradient(self):
        self.section(6)
        self.clear()
        C, s = np.array([-2.6, -0.9, 0.0]), 0.85

        def P(w):
            return C + s * np.array([w[0], w[1], 0.0])

        head = label("a real model: millions of weights", WHITE, 34).move_to([0, 3.2, 0])
        self.at("millions")
        self.play(FadeIn(head, shift=0.2 * DOWN), run_time=0.5)

        levels = [0.25, 1.0, 2.2, 3.5, bowl(W_START), 6.8]
        cols = color_gradient([BLUE_E, BLUE_C, TEAL_C], len(levels))
        rings = VGroup(*[Ellipse(width=2 * s * np.sqrt(2 * c), height=2 * s * np.sqrt(c / 2), color=col,
                                 stroke_width=3).move_to(C) for c, col in zip(levels, cols)])
        w1 = Arrow(C + 3.5 * LEFT, C + 3.6 * RIGHT, buff=0, color=GREY_B, stroke_width=2,
                   max_tip_length_to_length_ratio=0.03)
        w2 = Arrow(C + 2.0 * DOWN, C + 2.3 * UP, buff=0, color=GREY_B, stroke_width=2,
                   max_tip_length_to_length_ratio=0.05)
        w1l = label("w₁", GREY_B, 26).next_to(w1.get_end(), DOWN, buff=0.15)
        w2l = label("w₂", GREY_B, 26).next_to(w2.get_end(), RIGHT, buff=0.15)
        pt = Dot(P(W_START), radius=0.1, color=WHITE).set_z_index(3)
        note = caption("illustrative: the loss over just two of the weights")
        self.at("weights")
        self.play(FadeIn(w1), FadeIn(w2), FadeIn(w1l), FadeIn(w2l),
                  LaggedStart(*[Create(r) for r in rings], lag_ratio=0.12), FadeIn(pt), FadeIn(note),
                  run_time=1.0)

        g = np.array([W_START[0], 4 * W_START[1]])
        u = np.array([*(g / np.linalg.norm(g)), 0.0])
        grad = Arrow(pt.get_center(), pt.get_center() + 1.5 * u, buff=0, color=TEAL, stroke_width=6)
        grad_lab = label("∇L", TEAL, 30).next_to(grad.get_end(), LEFT, buff=0.3)
        px = 4.0
        vec = VGroup(label("∇L", TEAL, 32), label("= (∂L/∂w₁, ∂L/∂w₂)", WHITE, 32)).arrange(RIGHT, buff=0.2)
        vec.move_to([px, 0.35, 0])
        per = label("one slope for each weight", GREY_B, 26).next_to(vec, DOWN, buff=0.3)
        self.at("gradient")
        self.play(GrowArrow(grad), FadeIn(grad_lab), FadeIn(vec, shift=0.2 * UP), run_time=0.7)
        self.at("slope")
        self.play(FadeIn(per, shift=0.2 * UP), run_time=0.5)
        self.at("vector")
        self.play(Indicate(vec, scale_factor=1.08), run_time=0.6)

        up = label("points uphill", TEAL, 26).next_to(grad.get_end(), RIGHT, buff=0.25)
        self.at("uphill")
        self.play(FadeIn(up, shift=0.2 * RIGHT), run_time=0.5)

        pts = [P(w) for w in W_PATH]
        stp = Arrow(pts[0], pts[1], buff=0, color=YELLOW, stroke_width=6, max_tip_length_to_length_ratio=0.35)
        opp = label("step the opposite way", YELLOW, 28).move_to([px, -1.5, 0])
        rest = VGroup(*[Line(pts[k], pts[k + 1], color=YELLOW, stroke_width=3) for k in range(1, len(pts) - 1)])
        dots = VGroup(*[Dot(q, radius=0.06, color=YELLOW) for q in pts[1:]])
        self.at("opposite")
        self.play(GrowArrow(stp), FadeIn(opp, shift=0.2 * UP), run_time=0.4)
        self.play(LaggedStart(*[AnimationGroup(Create(ln), FadeIn(d)) for ln, d in zip(rest, dots[1:])],
                              lag_ratio=0.7), FadeIn(dots[0]), run_time=0.8)
        self.end_section()

    # 7. The chain rule ----------------------------------------------------------------------------
    def s7_chain(self):
        self.section(7)
        self.clear(run_time=0.3)
        y0 = 0.6
        xs = [-5.0, -2.5, 0.0, 2.5, 5.0]
        parts = [node("x", BLUE_C), fbox("f"), node("y", BLUE_C), fbox("g"), node("z", BLUE_C)]
        for m, x in zip(parts, xs):
            m.move_to([x, y0, 0])
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.12, color=GREY_B, stroke_width=4)
                          for a, b in zip(parts[:-1], parts[1:])])
        self.play(LaggedStart(*[FadeIn(m) for m in parts], lag_ratio=0.2), FadeIn(arrows), run_time=0.7)

        head = VGroup(label("layers are functions inside functions:", WHITE, 30),
                      label("z = g(f(x))", YELLOW, 30)).arrange(RIGHT, buff=0.3).move_to([0, 3.1, 0])
        self.at("inside")
        self.play(FadeIn(head, shift=0.2 * DOWN), run_time=0.5)

        rule = label("the chain rule:", WHITE, 32)
        mult = label("slopes multiply", YELLOW, 32)
        VGroup(rule, mult).arrange(RIGHT, buff=0.3).move_to([0, 2.2, 0])
        self.at("chain")
        self.play(FadeIn(rule, shift=0.2 * DOWN), run_time=0.4)
        self.at("multiply")
        self.play(FadeIn(mult, shift=0.2 * DOWN), run_time=0.4)

        s1 = label("dy/dx = 3", WHITE, 30).next_to(parts[1], DOWN, buff=0.35)
        note = caption("illustrative slopes")
        self.at("three")
        self.play(FadeIn(s1, shift=0.2 * UP), Indicate(parts[1][1], scale_factor=1.3), FadeIn(note), run_time=0.6)
        s2 = label("dz/dy = 2", WHITE, 30).next_to(parts[3], DOWN, buff=0.35)
        self.at("twice")
        self.play(FadeIn(s2, shift=0.2 * UP), Indicate(parts[3][1], scale_factor=1.3), run_time=0.6)

        brace = Brace(VGroup(*parts, s1, s2), DOWN, color=GREY_B)
        total = VGroup(label("dz/dx", WHITE, 36), label("= 3 × 2 = 6", YELLOW, 36)).arrange(RIGHT, buff=0.25)
        total.next_to(brace, DOWN, buff=0.3)
        self.at("six")
        self.play(GrowFromCenter(brace), FadeIn(total, shift=0.2 * UP), run_time=0.7)
        self.end_section()

    # 8. Backpropagation ---------------------------------------------------------------------------
    def s8_backprop(self):
        self.section(8)
        self.clear()
        head = label("backpropagation", WHITE, 38).move_to([0, 3.1, 0])
        self.play(FadeIn(head, shift=0.2 * DOWN), run_time=0.5)

        names = ["input", "layer", "layer", "layer", "loss"]
        colors = [BLUE_C, LAYER, LAYER, LAYER, GOLD_D]
        boxes = VGroup(*[fbox(n, c, w=1.6, h=0.9, font_size=28) for n, c in zip(names, colors)])
        boxes.arrange(RIGHT, buff=0.75).move_to([0, 0.9, 0])
        fwd = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.08, color=GREY_B, stroke_width=4)
                       for a, b in zip(boxes[:-1], boxes[1:])])
        self.at("layer")
        self.play(LaggedStart(*[FadeIn(b, shift=0.2 * RIGHT) for b in boxes], lag_ratio=0.15), FadeIn(fwd),
                  run_time=0.9)

        by = boxes.get_bottom()[1] - 0.45
        back = Arrow([boxes[-1].get_right()[0], by, 0], [boxes[0].get_left()[0], by, 0], buff=0, color=RED,
                     stroke_width=7, max_tip_length_to_length_ratio=0.03)
        tags = VGroup(*[label("× slope", RED, 24).move_to([boxes[k].get_x(), by - 0.45, 0]) for k in (3, 2, 1)])
        self.at("back")
        self.play(GrowArrow(back, rate_func=linear),
                  LaggedStart(*[FadeIn(t, shift=0.15 * LEFT) for t in tags], lag_ratio=0.6), run_time=2.4)

        out = VGroup(label("one backward pass", WHITE, 32), label("→", WHITE, 32),
                     label("the whole gradient", YELLOW, 32)).arrange(RIGHT, buff=0.25).move_to([0, -2.4, 0])
        self.at("gradient")
        self.play(FadeIn(out, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # 9. The code ---------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, font_size=26)
        code.move_to([0, 0.2, 0])
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[0])
        self.play(*self.clear_anims(), FadeIn(code, shift=0.3 * UP), run_time=0.5)
        self.at("slope")
        self.play(Create(hl), run_time=0.3)
        self.at("step")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.end_section()

    # 10. Outro -----------------------------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        card = used_in_card(USED_IN).move_to(ORIGIN)
        rows = card[1]
        self.at("see")
        self.play(*self.clear_anims(), FadeIn(card, shift=0.2 * UP), run_time=0.7)
        self.at("11")
        self.play(Indicate(rows[0], scale_factor=1.08), run_time=0.5)
        self.at("landscape", "gradients")
        self.play(Indicate(rows[1], scale_factor=1.08), run_time=0.5)
        nxt = next_up_card(NEXT)
        self.at("series")
        self.play(FadeOut(card), FadeIn(nxt, shift=0.2 * UP), run_time=0.7)
