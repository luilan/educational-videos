"""Foundations F5 — Exponentials and Logarithms.

Render from the repo root:  ./render.sh foundations f05
"""
import numpy as np
from manim import *

from common import code_panel, finish, next_up_card, token, used_in_card
from f05_script import LABEL, NEXT, TAGLINE, TITLE, USED_IN
from intro import play_token_intro
from voiced_scene import VoicedScene

EXP_COLOR, LOG_COLOR = BLUE, TEAL

# ------------------------------------------------------------------ every number shown on screen
assert 2 ** 10 == 1024 and f"{np.e:.3f}" == "2.718"
assert f"{np.exp(-10):.7f}" == "0.0000454" and np.exp(-10) < 1e-4
assert f"{np.exp(1):.1f}" == "2.7" and f"{np.exp(3):.1f}" == "20.1" and f"{20.1 / 2.7:.1f}" == "7.4"
assert np.log(1) == 0 and f"{np.log(0.9):.1f}" == "-0.1" and f"{np.log(0.01):.1f}" == "-4.6"
assert np.isclose(np.log(6.0 * 7.0), np.log(6.0) + np.log(7.0))

CODE = """np.exp([1, 3])        # array([ 2.718, 20.086])
np.exp(-10)           # 4.54e-05
np.log(0.9)           # -0.105
np.log(0.01)          # -4.605"""


# ------------------------------------------------------------------ helpers
def T(text, font_size=30, color=WHITE, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def make_axes(x_range, y_range, x_length, y_length, center):
    ax = Axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, tips=False,
              axis_config={"color": GREY_B, "stroke_width": 2, "tick_size": 0.06})
    return ax.move_to(center)


def exp_curve(ax, x_min, y_max, color=EXP_COLOR):
    return ax.plot(np.exp, x_range=[x_min, np.log(y_max), 0.02], color=color, stroke_width=5)


def log_curve(ax, y_min, x_max, color=LOG_COLOR):
    """ln x drawn as the exact mirror of e^t: the points (e^t, t). Smooth all the way down to x ≈ 0."""
    return ax.plot_parametric_curve(lambda t: np.array([np.exp(t), t]), t_range=[y_min, np.log(x_max), 0.02],
                                    color=color, stroke_width=5)


class ExpLogVideo(VoicedScene):
    VIDEO = "f05"

    # ------------------------------------------------------------------ stage helpers
    def on_stage(self, keep=()):
        """Top-level mobjects on stage that are not part of another on-stage mobject."""
        top = [m for m in self.mobjects if not isinstance(m, ValueTracker)]
        inner = set()
        for m in top:
            for d in m.get_family()[1:]:
                inner.add(id(d))
        keep_ids = set()
        for k in keep:
            for d in k.get_family():
                keep_ids.add(id(d))
        return [m for m in top if id(m) not in inner and id(m) not in keep_ids]

    def clear_anims(self, keep=()):
        return [FadeOut(m) for m in self.on_stage(keep)]

    def construct(self):
        play_token_intro(self, TITLE, 0, TAGLINE, label=LABEL)
        self.s1_mirror()
        self.s2_doubling()
        self.s3_e()
        self.s4_positive()
        self.s5_differences()
        self.s6_undo()
        self.s7_probabilities()
        self.s8_log_scale()
        self.s9_code()
        self.s10_outro()
        finish(self)

    # 1. Two functions, mirror images ------------------------------------------------------------
    def s1_mirror(self):
        self.section(1)
        ax = make_axes([-3, 4.5, 1], [-3, 4.5, 1], 6, 6, [-2.4, -0.3, 0])
        self.play(Create(ax), run_time=0.8)

        top = 4.5
        e_c = exp_curve(ax, -3, top)
        e_lab = T("eˣ", 34, EXP_COLOR).next_to(ax.c2p(np.log(top), top), RIGHT, buff=0.15)
        e_name = T("exponential", 38, EXP_COLOR).move_to([4.0, 1.1, 0])
        self.at("exponential")
        self.play(Create(e_c), FadeIn(e_lab), FadeIn(e_name, shift=0.2 * UP), run_time=1.0)

        diag = DashedLine(ax.c2p(-3, -3), ax.c2p(top, top), color=GREY_B, stroke_width=2, dash_length=0.12)
        diag_lab = T("y = x", 26, GREY_B).next_to(ax.c2p(top, top), RIGHT, buff=0.15)
        l_c = log_curve(ax, -3, top)
        l_lab = T("ln x", 34, LOG_COLOR).next_to(ax.c2p(top, np.log(top)), RIGHT, buff=0.15)
        l_name = T("logarithm", 38, LOG_COLOR).move_to([4.0, 0.0, 0])

        # the mirror: a copy of e^x flips over the line y = x and lands exactly on ln x
        src = e_c.copy()
        flip = e_c.copy()
        axis = normalize(np.array([1.0, 1.0, 0.0]))
        origin = ax.c2p(0, 0)

        def flip_step(m, a):
            m.become(src.copy().rotate(PI * a, axis=axis, about_point=origin))
            m.set_stroke(color=interpolate_color(ManimColor(EXP_COLOR), ManimColor(LOG_COLOR), a))

        self.at("logarithm")
        self.play(Create(diag), FadeIn(diag_lab), run_time=0.3)
        self.add(flip)
        self.play(UpdateFromAlphaFunc(flip, flip_step), FadeIn(l_name, shift=0.2 * UP), run_time=0.9)
        self.remove(flip)
        self.add(l_c)
        self.play(FadeIn(l_lab), run_time=0.2)

        mirror = VGroup(T("mirror images", 28, GREY_B), T("across y = x", 28, GREY_B)).arrange(DOWN, buff=0.15)
        mirror.move_to([4.0, -1.35, 0])
        self.at("build")
        self.play(FadeIn(mirror), run_time=0.5)
        self.end_section()

    # 2. Doubling -----------------------------------------------------------------------------------
    def s2_doubling(self):
        self.section(2)
        self.play(*self.clear_anims(), run_time=0.5)

        head1 = T("×2 every step", 38, GREEN)
        head2 = T("(not +2)", 30, GREY_B)
        VGroup(head1, head2).arrange(RIGHT, buff=0.4).move_to([0, 3.1, 0])
        self.at("multiplying")
        self.play(FadeIn(head1, shift=0.2 * DOWN), run_time=0.5)
        self.at("adding")
        self.play(FadeIn(head2), run_time=0.4)

        base_y, s = -2.7, 0.3
        xs = [-5.6, -4.2, -2.8, -1.4]
        values = [2, 4, 8, 16]
        bars, vals = VGroup(), VGroup()
        for x, v in zip(xs, values):
            bar = Rectangle(width=0.7, height=v * s, stroke_color=BLUE, stroke_width=2,
                            fill_color=BLUE, fill_opacity=0.5).move_to([x, base_y + v * s / 2, 0])
            bars.add(bar)
            vals.add(T(str(v), 30).next_to(bar, UP, buff=0.12))
        baseline = Line([-6.2, base_y, 0], [-0.8, base_y, 0], color=GREY_B, stroke_width=2)
        for k, (cue, rt) in enumerate(zip(["two", "four", "eight", "16"], [0.18, 0.18, 0.2, 0.4])):
            self.at(cue)
            anims = [GrowFromEdge(bars[k], DOWN), FadeIn(vals[k])]
            if k == 0:
                anims.append(Create(baseline))
            self.play(*anims, run_time=rt)

        times = VGroup(*[
            T("×2", 24, YELLOW).move_to([(xs[k] + xs[k + 1]) / 2, base_y + (values[k] + values[k + 1]) * s / 2, 0])
            for k in range(3)])
        self.at("doubling")
        self.play(LaggedStart(*[FadeIn(t, scale=0.8) for t in times], lag_ratio=0.3), run_time=0.7)

        after = T("after 10 doublings", 30, GREY_B)
        power = T("2¹⁰ = 1,024", 66, YELLOW)
        past = T("past a thousand", 34, GREEN)
        VGroup(after, power, past).arrange(DOWN, buff=0.45).move_to([3.4, 0.0, 0])
        self.at("10")
        self.play(FadeIn(after), FadeIn(power, shift=0.2 * UP), run_time=0.5)
        self.at("000")
        self.play(FadeIn(past, shift=0.2 * UP), run_time=0.4)
        self.end_section()

    # 3. The number e and the graph of e^x ------------------------------------------------------------
    def s3_e(self):
        self.section(3)
        self.play(*self.clear_anims(), run_time=0.5)

        big = T("e ≈ 2.718", 110, YELLOW).move_to([0, 0.5, 0])
        euler = T("Euler's number", 30, GREY_B).next_to(big, DOWN, buff=0.45)
        self.at("e")
        self.play(FadeIn(big[0], scale=0.8), FadeIn(euler), run_time=0.5)
        self.at("2")
        self.play(FadeIn(big[1:4]), run_time=0.25)
        self.at("718")
        self.play(FadeIn(big[4:]), run_time=0.25)

        self.at("function")
        self.play(big.animate.scale(0.4).move_to([-4.9, 3.2, 0]), FadeOut(euler), run_time=0.6)

        ax = make_axes([-3, 2.5, 1], [-1, 8, 1], 5.5, 5, [-2.7, -0.7, 0])
        curve = exp_curve(ax, -3, 8)
        lab = T("eˣ", 38, EXP_COLOR).next_to(ax.c2p(np.log(8), 8), RIGHT, buff=0.15)
        self.at("x")
        self.play(Create(ax), Create(curve), FadeIn(lab), run_time=0.7)

        p1 = VGroup(T("1", 32, YELLOW), T("always positive", 32)).arrange(RIGHT, buff=0.3)
        p2 = VGroup(T("2", 32, YELLOW), T("exaggerates differences", 32)).arrange(RIGHT, buff=0.3)
        VGroup(p1, p2).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to([3.75, -0.2, 0])
        self.at("properties")
        self.play(LaggedStart(FadeIn(p1, shift=0.2 * LEFT), FadeIn(p2, shift=0.2 * LEFT), lag_ratio=0.4),
                  run_time=0.7)
        self.end_section()
        self.graph3 = (ax, curve, lab, big, p1, p2)

    # 4. Always positive ------------------------------------------------------------------------------
    def s4_positive(self):
        self.section(4)
        ax3, curve3, lab3, big, p1, p2 = self.graph3
        ax = make_axes([-11, 3, 1], [-1, 8, 1], 11, 5, [0, -0.6, 0])
        curve = exp_curve(ax, -11, 8)
        lab = T("eˣ", 38, EXP_COLOR).next_to(ax.c2p(np.log(8), 8), RIGHT, buff=0.15)
        ticks = VGroup(*[T(s, 22, GREY_B).next_to(ax.c2p(x, 0), DOWN, buff=0.15)
                         for x, s in [(-10, "−10"), (-5, "−5")]],
                       T("0", 22, GREY_B).next_to(ax.c2p(0, 0), DL, buff=0.1))
        head = VGroup(T("1", 36, YELLOW), T("always positive", 36)).arrange(RIGHT, buff=0.3).move_to([0, 3.2, 0])
        self.play(FadeOut(big), FadeOut(p1), FadeOut(p2), FadeIn(head, shift=0.2 * DOWN),
                  ReplacementTransform(ax3, ax), ReplacementTransform(curve3, curve), ReplacementTransform(lab3, lab),
                  run_time=0.8)
        self.play(FadeIn(ticks), run_time=0.15)

        area = ax.get_area(curve, x_range=[-11, np.log(8)], color=GREEN, opacity=0.35)
        self.at("positive")
        self.play(FadeIn(area), head[1].animate.set_color(GREEN), run_time=0.8)

        dot = Dot(ax.c2p(-10, np.exp(-10)), radius=0.09, color=GREEN)
        note = T("e⁻¹⁰ ≈ 0.0000454", 38, GREEN)
        note.move_to([0, 0.9, 0]).align_to([-5.0, 0, 0], LEFT)
        pointer = Arrow([dot.get_x(), note.get_bottom()[1] - 0.1, 0], dot.get_center() + 0.12 * UP, buff=0,
                        color=GREEN, stroke_width=4, max_tip_length_to_length_ratio=0.1)
        self.at("10")
        self.play(FadeIn(dot, scale=2), GrowArrow(pointer), FadeIn(note), run_time=0.6)

        tiny = T("less than 0.0001, but still positive", 26, GREY_B)
        tiny.next_to(note, DOWN, buff=0.3).align_to([-4.4, 0, 0], LEFT)
        self.at("thousandth")
        self.play(FadeIn(tiny), run_time=0.4)

        asym = DashedLine(ax.c2p(-11, 0), ax.c2p(3, 0), color=YELLOW, stroke_width=3, dash_length=0.15)
        never1 = T("never reaches 0", 28, YELLOW)
        never2 = T("never goes negative", 28, YELLOW)
        VGroup(never1, never2).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([-0.6, -1.55, 0])
        self.at("zero")
        self.play(Create(asym), FadeIn(never1), run_time=0.6)
        self.at("negative")
        self.play(FadeIn(never2), run_time=0.4)
        self.end_section()

    # 5. Exaggerates differences ------------------------------------------------------------------
    def s5_differences(self):
        self.section(5)
        head = VGroup(T("2", 36, YELLOW), T("exaggerates differences", 36)).arrange(RIGHT, buff=0.3)
        head.move_to([0, 3.2, 0])
        self.play(*self.clear_anims(), run_time=0.5)
        self.play(FadeIn(head), run_time=0.4)

        nl = NumberLine(x_range=[0, 4, 1], length=4.8, color=GREY_B, stroke_width=2, tick_size=0.08)
        nl.move_to([-3.8, -0.2, 0])
        nums = VGroup(*[T(str(k), 24, GREY_B).next_to(nl.n2p(k), DOWN, buff=0.2) for k in range(5)])
        title = T("inputs x", 28, GREY_B).move_to([-3.8, 1.9, 0])
        self.at("inputs")
        self.play(Create(nl), FadeIn(nums), FadeIn(title), run_time=0.3)

        d1 = Dot(nl.n2p(1), radius=0.11, color=BLUE)
        d3 = Dot(nl.n2p(3), radius=0.11, color=BLUE)
        self.at("1")
        self.play(FadeIn(d1, scale=2), run_time=0.3)
        self.at("3")
        self.play(FadeIn(d3, scale=2), run_time=0.3)
        brace = Brace(Line(nl.n2p(1), nl.n2p(3)), UP, buff=0.25, color=GREY_B)
        diff = T("differ by 2", 30).next_to(brace, UP, buff=0.15)
        self.at("2")
        self.play(GrowFromCenter(brace), FadeIn(diff), run_time=0.5)

        base_y, s = -2.7, 4.4 / np.exp(3)
        bars, labs, unders = VGroup(), VGroup(), VGroup()
        for x, k, text in [(1.8, 1, "e¹ ≈ 2.7"), (4.4, 3, "e³ ≈ 20.1")]:
            h = np.exp(k) * s
            bar = Rectangle(width=1.1, height=h, stroke_color=EXP_COLOR, stroke_width=2,
                            fill_color=EXP_COLOR, fill_opacity=0.5).move_to([x, base_y + h / 2, 0])
            bars.add(bar)
            labs.add(T(text, 34).next_to(bar, UP, buff=0.18))
            unders.add(T(f"x = {k}", 24, GREY_B).next_to(bar, DOWN, buff=0.18))
        baseline = Line([0.8, base_y, 0], [5.4, base_y, 0], color=GREY_B, stroke_width=2)
        self.at("7")
        self.play(Create(baseline), GrowFromEdge(bars[0], DOWN), FadeIn(labs[0]), FadeIn(unders[0]), run_time=0.5)
        self.at("20")
        self.play(GrowFromEdge(bars[1], DOWN), FadeIn(labs[1]), FadeIn(unders[1]), run_time=0.7)

        ratio = T("20.1 ÷ 2.7 ≈ 7.4×", 42, YELLOW).move_to([-3.8, -2.2, 0])
        self.at("7")
        self.play(FadeIn(ratio, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # 6. The logarithm undoes the exponential --------------------------------------------------------
    def s6_undo(self):
        self.section(6)
        self.play(*self.clear_anims(), run_time=0.5)

        b_x = token("x", color=GREY_B, font_size=36).move_to([-5.7, 1.1, 0])
        b_e = token("eˣ", color=EXP_COLOR, font_size=36).move_to([-3.6, 1.1, 0])
        b_back = token("x", color=GREEN, font_size=36).move_to([-1.5, 1.1, 0])

        def link(a, b, name, color):
            arr = Arrow(a.get_right(), b.get_left(), buff=0.12, color=color, stroke_width=4,
                        max_tip_length_to_length_ratio=0.2)
            return arr, T(name, 26, color).next_to(arr, UP, buff=0.1)

        a1, n1 = link(b_x, b_e, "exp", EXP_COLOR)
        a2, n2 = link(b_e, b_back, "ln", LOG_COLOR)
        self.at("undoes")
        self.play(FadeIn(b_x), GrowArrow(a1), FadeIn(n1), FadeIn(b_e), run_time=0.5)
        self.play(GrowArrow(a2), FadeIn(n2), FadeIn(b_back), run_time=0.5)

        ident = T("ln(eˣ) = x", 50, YELLOW).move_to([-3.6, -0.5, 0])
        self.at("just")
        self.at("x")
        self.play(FadeIn(ident, shift=0.2 * UP), run_time=0.5)

        ax = make_axes([0, 5, 1], [-4, 2, 1], 5, 5.4, [3.6, -0.6, 0])
        curve = log_curve(ax, -4, 5)
        lab = T("ln x", 34, LOG_COLOR).move_to(ax.c2p(4.5, 2.0))
        self.at("log")
        self.play(Create(ax), Create(curve), FadeIn(lab), run_time=0.8)

        one = Dot(ax.c2p(1, 0), radius=0.09, color=WHITE)
        one_lab = T("ln 1 = 0", 30).next_to(one, DR, buff=0.12)
        self.at("zero")
        self.play(FadeIn(one, scale=2), FadeIn(one_lab), run_time=0.5)

        t = ValueTracker(0.0)
        mover = always_redraw(lambda: Dot(ax.c2p(np.exp(t.get_value()), t.get_value()), radius=0.09, color=YELLOW))
        self.at("shrink")
        self.add(mover)
        self.play(t.animate.set_value(-3.8), run_time=2.3, rate_func=rate_functions.ease_in_quad)

        x_axis_x = ax.c2p(0, 0)[0]
        dive = Arrow([x_axis_x - 0.4, ax.c2p(0, -1)[1], 0], [x_axis_x - 0.4, ax.c2p(0, -4)[1], 0], buff=0,
                     color=RED, stroke_width=5, max_tip_length_to_length_ratio=0.12)
        inf = T("ln x → −∞", 30, RED).move_to(ax.c2p(0, -3.4)).align_to(ax.c2p(0.4, 0), LEFT)
        self.at("infinity")
        self.play(GrowArrow(dive), FadeIn(inf), run_time=0.5)
        self.end_section()

    # 7. Logs of probabilities ----------------------------------------------------------------------
    def s7_probabilities(self):
        self.section(7)
        self.play(*self.clear_anims(), run_time=0.5)

        ax = make_axes([0, 1, 0.1], [-5, 0.5, 1], 8, 5.5, [-0.3, -0.6, 0])
        xt = VGroup(*[T(s, 22, GREY_B).next_to(ax.c2p(x, 0), UP, buff=0.15) for x, s in [(0.5, "0.5"), (1, "1")]])
        yt = VGroup(*[T(s, 22, GREY_B).next_to(ax.c2p(0, y), LEFT, buff=0.15)
                      for y, s in [(0, "0"), (-1, "−1"), (-2, "−2"), (-3, "−3"), (-4, "−4"), (-5, "−5")]])
        p_lab = T("p", 30, GREY_B).next_to(ax.c2p(1, 0), RIGHT, buff=0.4)
        lnp = T("ln p", 30, LOG_COLOR).next_to(ax.c2p(0, 0.5), UP, buff=0.2)
        curve = log_curve(ax, -5, 1)
        head = T("the log of a probability", 34).move_to([0, 3.3, 0])
        self.play(Create(ax), FadeIn(xt), FadeIn(yt), FadeIn(p_lab), run_time=0.6)
        self.at("probabilities")
        self.play(Create(curve), FadeIn(lnp), FadeIn(head), run_time=0.9)

        d9 = Dot(ax.c2p(0.9, np.log(0.9)), radius=0.1, color=GREEN)
        l9 = T("ln 0.9 ≈ −0.1", 32, GREEN).next_to(d9, DOWN, buff=0.55)
        self.at("9")
        self.play(FadeIn(d9, scale=2), FadeIn(l9, shift=0.2 * UP), run_time=0.5)

        d01 = Dot(ax.c2p(0.01, np.log(0.01)), radius=0.1, color=RED)
        l01 = T("ln 0.01 ≈ −4.6", 32, RED).next_to(d01, RIGHT, buff=0.35)
        self.at("01")
        self.play(FadeIn(d01, scale=2), FadeIn(l01, shift=0.2 * LEFT), run_time=0.5)

        cap = T("less likely → more negative", 32, YELLOW).move_to([2.2, -1.9, 0])
        self.at("negative")
        self.play(FadeIn(cap, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # 8. Multiplication into addition; log-scale axis ---------------------------------------------------
    def s8_log_scale(self):
        self.section(8)
        self.play(*self.clear_anims(), run_time=0.5)

        rule = T("log(a × b) = log a + log b", 48)
        rule.move_to([0, 2.4, 0])
        self.at("multiplication")
        self.play(FadeIn(rule[:8]), run_time=0.4)
        self.at("addition")
        self.play(FadeIn(rule[8:], shift=0.2 * LEFT), run_time=0.5)

        y0, x0, step = -0.7, -5.85, 1.3
        names = ["1", "10", "100", "1K", "10K", "100K", "1M", "10M", "100M", "1B"]
        line = Line([-6.3, y0, 0], [6.3, y0, 0], color=GREY_B, stroke_width=2)
        ticks = VGroup(*[Line([x0 + k * step, y0 - 0.12, 0], [x0 + k * step, y0 + 0.12, 0], color=GREY_B,
                              stroke_width=2) for k in range(10)])
        labels = VGroup(*[T(n, 24).next_to(ticks[k], DOWN, buff=0.15) for k, n in enumerate(names)])
        cap = T("log scale", 30, GREY_B).move_to([0, 1.2, 0])
        self.at("scale")
        self.play(Create(line), FadeIn(cap), LaggedStart(*[FadeIn(VGroup(t, l)) for t, l in zip(ticks, labels)],
                                                         lag_ratio=0.1), run_time=0.9)

        tens = VGroup(*[T("×10", 20, YELLOW).move_to([x0 + (k + 0.5) * step, y0 + 0.35, 0]) for k in range(9)])
        self.at("ten")
        self.play(LaggedStart(*[FadeIn(t) for t in tens], lag_ratio=0.1), run_time=0.7)

        def mark(k, text):
            ring = SurroundingRectangle(labels[k], color=YELLOW, buff=0.1, corner_radius=0.08)
            word = T(text, 28, YELLOW).next_to(ring, DOWN, buff=0.25)
            return ring, word

        r1, w1 = mark(3, "a thousand")
        r2, w2 = mark(9, "a billion")
        w2.align_to([6.6, 0, 0], RIGHT)
        self.at("thousand")
        self.play(Create(r1), FadeIn(w1), run_time=0.4)
        self.at("billion")
        self.play(Create(r2), FadeIn(w2), run_time=0.4)
        self.end_section()

    # 9. NumPy --------------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, 30)
        code.move_to([0, 0.1, 0])
        row_h = abs(code.line_numbers[0].get_y() - code.line_numbers[1].get_y())
        hl.stretch_to_fit_height(2 * row_h)
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT)
        hl.set_y((code.line_numbers[0].get_y() + code.line_numbers[1].get_y()) / 2)
        self.play(*self.clear_anims(), FadeIn(code, shift=0.3 * UP), run_time=0.6)
        self.at("exp")
        self.play(Create(hl), run_time=0.3)
        self.at("log")
        self.play(hl.animate.set_y((code.line_numbers[2].get_y() + code.line_numbers[3].get_y()) / 2), run_time=0.3)
        self.end_section()

    # 10. Outro --------------------------------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        card = used_in_card(USED_IN).move_to(ORIGIN)
        rows = card[1]
        self.at("see")
        self.play(*self.clear_anims(), FadeIn(card, shift=0.2 * UP), run_time=0.6)
        for k, cue in enumerate(["6", "10", "11", "12"]):
            self.at(cue)
            self.play(Indicate(rows[k], scale_factor=1.08), run_time=0.4)
        nxt = next_up_card(NEXT)
        self.at("sampling")
        self.play(FadeOut(card), FadeIn(nxt, shift=0.2 * UP), run_time=0.6)
        self.end_section()
