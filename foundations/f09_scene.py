"""Foundations F9 — Waves and Rotations.

Render from the repo root:  ./render.sh foundations f09
"""
import numpy as np
from manim import *

from common import code_panel, finish, next_up_card, used_in_card
from f09_script import LABEL, NEXT, TAGLINE, TITLE, USED_IN
from intro import play_token_intro
from voiced_scene import VoicedScene

SIN_COLOR = YELLOW
COS_COLOR = TEAL
MONO = "DejaVu Sans Mono"

# ---------------------------------------------------------------------------- verified numbers
FULL_TURN = 2 * np.pi
assert f"{FULL_TURN:.3f}" == "6.283" and f"{FULL_TURN:.2f}" == "6.28"
assert f"{2 * FULL_TURN:.2f}" == "12.57"

THETA = np.pi / 6                                   # 30 degrees
ROT = np.array([[np.cos(THETA), -np.sin(THETA)],
                [np.sin(THETA), np.cos(THETA)]])
V_ROT = ROT @ np.array([1.0, 0.0])
assert f"{V_ROT[0]:.3f}" == "0.866" and f"{V_ROT[1]:.3f}" == "0.500"
assert f"{np.linalg.norm(V_ROT):.1f}" == "1.0"

STEP = np.radians(20)                               # one "step" in section 8 (illustrative)
assert np.isclose(5 * STEP - 2 * STEP, 3 * STEP)

# four stacked waves of section 6 (illustrative frequencies)
FREQS = [1, 1 / 2, 1 / 4, 1 / 8]
FREQ_NAMES = ["sin(p)", "sin(p/2)", "sin(p/4)", "sin(p/8)"]
FREQ_COLORS = [BLUE_C, TEAL_C, GREEN_C, PURPLE_B]
N_POS = 20


def fmt(v, nd=2):
    s = f"{v:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−")


assert [fmt(np.sin(w * 3)) for w in FREQS] == ["0.14", "1.00", "0.68", "0.37"]
assert [fmt(np.sin(w * 10)) for w in FREQS] == ["−0.54", "−0.96", "0.60", "0.95"]

CODE = """theta = np.pi / 6                        # 30 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
v = np.array([1.0, 0.0])
R @ v                                    # array([0.866, 0.5])
np.linalg.norm(R @ v)                    # 1.0"""

# circle + wave geometry shared by sections 2-4
C = np.array([-4.2, -0.3, 0.0])     # circle centre
R = 1.9                              # screen length of "1"
X0 = -1.4                            # wave axis: angle 0 sits here
K = 3.6 / FULL_TURN                  # screen units per radian along the wave axis


# ---------------------------------------------------------------------------- helpers
def label(text, color=GREY_B, font_size=26, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def caption(text, font_size=20):
    return Text(text, font_size=font_size, color=GREY_B).to_corner(DR, buff=0.3)


def unit(a):
    return np.array([np.cos(a), np.sin(a), 0.0])


def circle_pt(t):
    return C + R * unit(t)


def wave_pt(t, f=np.sin):
    return np.array([X0 + K * t, C[1] + R * f(t), 0.0])


def column(entries, colors, font_size=26, buff=0.18):
    """A column vector of Text entries between square brackets."""
    rows = VGroup(*[Text(e, font=MONO, font_size=font_size, color=c) for e, c in zip(entries, colors)])
    rows.arrange(DOWN, buff=buff)
    for r in rows:
        r.align_to(rows, RIGHT)
    h, w = rows.height + 0.3, rows.width + 0.4

    def bracket(side):
        x = side * w / 2
        d = -side * 0.12
        pts = [[x + d, h / 2, 0], [x, h / 2, 0], [x, -h / 2, 0], [x + d, -h / 2, 0]]
        return VMobject(stroke_color=GREY_B, stroke_width=2.5).set_points_as_corners(pts)

    brackets = VGroup(bracket(-1), bracket(1)).move_to(rows)
    return VGroup(brackets, rows)


def clock_icon(center, r=0.6):
    face = Circle(radius=r, stroke_color=WHITE, stroke_width=3).move_to(center)
    ticks = VGroup(*[Line(center + 0.78 * r * unit(a), center + 0.95 * r * unit(a), color=GREY_B, stroke_width=2)
                     for a in np.arange(12) * TAU / 12])
    hour = Line(center, center + 0.5 * r * unit(np.radians(30)), color=YELLOW, stroke_width=6)
    minute = Line(center, center + 0.82 * r * unit(np.radians(140)), color=TEAL_C, stroke_width=4)
    second = Line(center, center + 0.85 * r * unit(np.radians(-110)), color=RED_C, stroke_width=2)
    return VGroup(face, ticks, hour, minute, second, Dot(center, radius=0.06, color=WHITE))


def beyond(mob, origin, d, r):
    """Place mob just outside radius r from origin in direction d (no overlap with the arrow tip)."""
    ext = 0.5 * (abs(mob.width * d[0]) + abs(mob.height * d[1]))
    return mob.move_to(origin + (r + ext) * d)


class WavesVideo(VoicedScene):
    VIDEO = "f09"

    # ------------------------------------------------------------------ stage helpers
    def clear_anims(self, keep=()):
        anims = []
        for m in list(self.mobjects):
            if m in keep:
                continue
            if isinstance(m, ValueTracker):
                m.clear_updaters()
                self.remove(m)
                continue
            m.clear_updaters()
            anims.append(FadeOut(m))
        return anims

    def clear(self, run_time=0.4, keep=()):
        anims = self.clear_anims(keep)
        if anims:
            self.play(*anims, run_time=run_time)

    def drive(self, tracker, target, duration, rate=smooth):
        """Move a tracker to target in the background, so other cues can play meanwhile."""
        tracker.clear_updaters()
        start, state = tracker.get_value(), {"t": 0.0}

        def upd(m, dt):
            state["t"] += dt
            a = min(state["t"] / duration, 1.0)
            m.set_value(start + (target - start) * rate(a))

        tracker.add_updater(upd)
        if tracker not in self.mobjects:
            self.add(tracker)

    def construct(self):
        play_token_intro(self, TITLE, 0, TAGLINE, label=LABEL)
        self.s1_hook()
        self.s2_circle()
        self.s3_wave()
        self.s4_radians()
        self.s5_frequency()
        self.s6_several()
        self.s7_rotate()
        self.s8_relative()
        self.s9_code()
        self.s10_outro()
        self.end_section()
        finish(self)

    # 1. Waves and rotations in episode 4 ---------------------------------------------------------
    def s1_hook(self):
        self.section(1)
        ep = label("Episode 4 · Where Am I? Position", GREY_B, 26).move_to([0, 3.0, 0])
        self.at("4")
        self.play(FadeIn(ep, shift=0.2 * DOWN), run_time=0.5)

        waves = VGroup()
        for y, f, c in zip([1.2, 0.0, -1.2], [1, 2, 4], [BLUE_C, TEAL_C, GREEN_C]):
            waves.add(FunctionGraph(lambda x, y=y, f=f: y + 0.4 * np.sin(TAU * f * (x + 6.0) / 5.2),
                                    x_range=[-6.0, -0.8, 0.01], color=c, stroke_width=4))
        self.at("waves")
        self.play(LaggedStart(*[Create(w) for w in waves], lag_ratio=0.2), run_time=0.9)

        cen = np.array([3.4, 0.0, 0.0])
        ring = Circle(radius=1.4, stroke_color=GREY_B, stroke_width=3).move_to(cen)
        ang = ValueTracker(0.5)
        arrow = always_redraw(lambda: Arrow(cen, cen + 1.4 * unit(ang.get_value()), buff=0, color=BLUE_C,
                                            stroke_width=7, max_tip_length_to_length_ratio=0.2))
        hub = Dot(cen, radius=0.07, color=WHITE)
        self.at("rotations")
        self.play(Create(ring), FadeIn(arrow), FadeIn(hub), run_time=0.5)
        ang.add_updater(lambda m, dt: m.increment_value(1.5 * dt))
        self.add(ang)

        w_sin = Text("sine", font_size=48, color=SIN_COLOR).move_to([-1.4, -2.7, 0])
        w_cos = Text("cosine", font_size=48, color=COS_COLOR).move_to([1.6, -2.7, 0])
        self.at("sine")
        self.play(FadeIn(w_sin, shift=0.2 * UP), run_time=0.4)
        self.at("cosine")
        self.play(FadeIn(w_cos, shift=0.2 * UP), run_time=0.4)
        self.end_section()

    # 2. The unit circle: height = sine, horizontal = cosine --------------------------------------
    def s2_circle(self):
        self.section(2)
        theta = ValueTracker(0.0)
        self.theta = theta
        axes = VGroup(Line(C + (R + 0.3) * LEFT, C + (R + 0.3) * RIGHT),
                      Line(C + (R + 0.3) * DOWN, C + (R + 0.3) * UP)).set_stroke(GREY_D, 2)
        ring = Circle(radius=R, stroke_color=GREY_B, stroke_width=3).move_to(C)
        self.play(*self.clear_anims(), FadeIn(axes), Create(ring), run_time=0.5)
        self.add(theta)

        def P():
            return circle_pt(theta.get_value())

        radius = always_redraw(lambda: Line(C, P(), color=BLUE_C, stroke_width=4))
        dot = always_redraw(lambda: Dot(P(), radius=0.1, color=WHITE))
        self.at("point")
        self.play(FadeIn(radius), FadeIn(dot, scale=0.5), run_time=0.3)
        self.at("moving")
        self.drive(theta, TAU + np.radians(55), 2.9)

        one_op = ValueTracker(0.0)          # fade via a tracker so the label keeps following the radius
        one = always_redraw(lambda: label("1", WHITE, 28).move_to(
            C + 0.5 * R * unit(theta.get_value()) + 0.28 * unit(theta.get_value() + PI / 2))
            .set_opacity(one_op.get_value()))
        self.at("1")
        self.add(one)
        self.play(one_op.animate.set_value(1.0), run_time=0.4)

        def arc():
            a = theta.get_value() % TAU
            return Arc(radius=0.5, start_angle=0, angle=max(a, 1e-3), arc_center=C, color=WHITE, stroke_width=3,
                       stroke_opacity=1.0 if a > 0.15 else 0.0)

        def th_label():
            a = theta.get_value() % TAU
            return label("θ", WHITE, 28).move_to(C + 0.8 * unit(a / 2)).set_opacity(1.0 if a > 0.15 else 0.0)

        arc_m, th_m = always_redraw(arc), always_redraw(th_label)
        self.arc_m, self.th_m, self.ring, self.axes, self.radius, self.dot = arc_m, th_m, ring, axes, radius, dot
        readout = always_redraw(lambda: label(f"θ = {np.degrees(theta.get_value() % TAU):.0f}°", WHITE, 28)
                                .move_to([C[0], C[1] + R + 0.75, 0]))
        self.at("angle")
        theta.clear_updaters()
        theta.set_value(theta.get_value() % TAU)
        self.play(FadeIn(arc_m), FadeIn(th_m), FadeIn(readout), run_time=0.5)

        sin_seg = always_redraw(lambda: Line([P()[0], C[1], 0], P(), color=SIN_COLOR, stroke_width=5))
        sin_lab = always_redraw(lambda: label("sin θ", SIN_COLOR, 28).next_to(
            [P()[0], (P()[1] + C[1]) / 2, 0], RIGHT, buff=0.15).add_background_rectangle(opacity=0.85, buff=0.04))
        sin_txt = label("height = sin θ", SIN_COLOR, 36).move_to([0.6, 1.0, 0], aligned_edge=LEFT)
        self.at("height")
        self.play(Create(sin_seg), FadeIn(sin_lab), FadeIn(sin_txt, shift=0.2 * LEFT), run_time=0.6)

        cos_seg = always_redraw(lambda: Line(C, [P()[0], C[1], 0], color=COS_COLOR, stroke_width=6))
        cos_lab = always_redraw(lambda: label("cos θ", COS_COLOR, 28).next_to(
            [(P()[0] + C[0]) / 2, C[1], 0], DOWN, buff=0.18).add_background_rectangle(opacity=0.85, buff=0.04))
        cos_txt = label("horizontal = cos θ", COS_COLOR, 36).move_to([0.6, 0.0, 0], aligned_edge=LEFT)
        self.at("cosine")
        self.play(Create(cos_seg), FadeIn(cos_lab), FadeIn(cos_txt, shift=0.2 * LEFT), run_time=0.5)
        self.sin_seg = sin_seg
        self.s2_extras = [one, readout, sin_lab, sin_txt, cos_seg, cos_lab, cos_txt]
        self.end_section()

    # 3. Plot the height: the sine wave ------------------------------------------------------------
    def s3_wave(self):
        self.section(3)
        theta = self.theta
        for m in self.s2_extras:
            m.clear_updaters()
        axis = Arrow([X0 - 0.2, C[1], 0], [6.35, C[1], 0], buff=0, color=GREY_B, stroke_width=2,
                     max_tip_length_to_length_ratio=0.02)
        axis_name = label("θ", GREY_B, 26).move_to([6.5, C[1] + 0.3, 0])
        self.play(*[FadeOut(m) for m in self.s2_extras], theta.animate.set_value(0.0), run_time=0.45)
        self.play(FadeIn(axis), FadeIn(axis_name), run_time=0.3)

        trace = always_redraw(lambda: ParametricFunction(wave_pt, t_range=[0, max(theta.get_value(), 0.02), 0.02],
                                                         color=SIN_COLOR, stroke_width=4))
        wave_dot = always_redraw(lambda: Dot(wave_pt(theta.get_value()), radius=0.08, color=SIN_COLOR))
        connector = always_redraw(lambda: DashedLine(circle_pt(theta.get_value()), wave_pt(theta.get_value()),
                                                     color=GREY_B, stroke_width=2, dash_length=0.08))
        self.at("growing")
        self.add(trace, connector, wave_dot)
        self.drive(theta, 2 * TAU, 6.3, rate=linear)

        top = DashedLine([C[0], C[1] + R, 0], [X0 + K * 2 * TAU, C[1] + R, 0], color=GREY_C, stroke_width=2,
                         dash_length=0.1)
        bot = DashedLine([C[0], C[1] - R, 0], [X0 + K * 2 * TAU, C[1] - R, 0], color=GREY_C, stroke_width=2,
                         dash_length=0.1)
        top_lab = label("+1", WHITE, 30).move_to([6.25, C[1] + R, 0])
        bot_lab = label("−1", WHITE, 30).move_to([6.25, C[1] - R, 0])
        self.at("1")
        self.play(Create(top), FadeIn(top_lab), run_time=0.4)
        self.at("minus")
        self.play(Create(bot), FadeIn(bot_lab), run_time=0.4)

        cos_wave = ParametricFunction(lambda t: wave_pt(t, np.cos), t_range=[0, 2 * TAU, 0.02], color=COS_COLOR,
                                      stroke_width=4)
        leg = VGroup(
            VGroup(Line(ORIGIN, 0.45 * RIGHT, color=SIN_COLOR, stroke_width=5), label("sin θ", SIN_COLOR, 28)),
            VGroup(Line(ORIGIN, 0.45 * RIGHT, color=COS_COLOR, stroke_width=5), label("cos θ", COS_COLOR, 28)),
        )
        for item in leg:
            item.arrange(RIGHT, buff=0.15)
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([5.3, 2.85, 0])
        self.at("cosine")
        theta.clear_updaters()
        theta.set_value(2 * TAU)
        trace.clear_updaters()
        self.play(Create(cos_wave), FadeIn(leg), run_time=1.0)

        y = C[1] + R + 0.3
        shift = DoubleArrow([X0 + K * TAU, y, 0], [X0 + K * (TAU + PI / 2), y, 0], buff=0, color=WHITE,
                            stroke_width=3, tip_length=0.14)
        shift_lab = label("shifted by ¼ turn", WHITE, 28).next_to(shift, UP, buff=0.12)
        self.at("quarter")
        self.play(GrowFromCenter(shift), FadeIn(shift_lab, shift=0.1 * UP), run_time=0.5)
        self.trace, self.wave_dot, self.connector = trace, wave_dot, connector
        self.s3_extras = [cos_wave, leg, shift, shift_lab]
        self.axis = axis
        self.end_section()

    # 4. Radians: a full turn is 2π ≈ 6.28 -------------------------------------------------------
    def s4_radians(self):
        self.section(4)
        theta = self.theta
        self.play(*[FadeOut(m) for m in self.s3_extras], FadeOut(self.wave_dot), FadeOut(self.connector),
                  run_time=0.5)
        theta.set_value(0.0)          # 4π and 0 are the same point on the circle

        readout = always_redraw(lambda: label(
            f"θ = {np.degrees(theta.get_value()):.0f}° = {theta.get_value():.2f} rad", WHITE, 28)
            .move_to([C[0], C[1] + R + 0.75, 0]))
        self.at("radians")
        self.play(FadeIn(readout), FadeIn(self.wave_dot), FadeIn(self.connector), run_time=0.5)
        self.at("full")
        self.drive(theta, TAU, 1.6)

        full = Text("full turn = 2π ≈ 6.28", font_size=36, color=YELLOW).move_to([2.2, 2.9, 0])
        ticks = VGroup()
        for k, name in [(1, "2π"), (2, "4π")]:
            x = X0 + K * k * TAU
            ticks.add(Line([x, C[1] - 0.1, 0], [x, C[1] + 0.1, 0], color=WHITE, stroke_width=3))
            ticks.add(label(name, WHITE, 26).move_to([x + 0.32, C[1] - 0.32, 0]))
        self.at("pi")
        self.play(FadeIn(full, shift=0.2 * DOWN), FadeIn(ticks), run_time=0.5)
        self.at("so")
        self.drive(theta, 2 * TAU, 1.6)

        yb = C[1] - R - 0.4
        period = DoubleArrow([X0, yb, 0], [X0 + K * TAU, yb, 0], buff=0, color=YELLOW, stroke_width=3,
                             tip_length=0.16)
        period_lab = label("one period = 6.28", YELLOW, 28).next_to(period, DOWN, buff=0.12)
        self.at("units")
        self.play(GrowFromCenter(period), FadeIn(period_lab, shift=0.1 * UP), run_time=0.5)
        self.end_section()

    # 5. Frequency ---------------------------------------------------------------------------------
    def s5_frequency(self):
        self.section(5)
        axes = Axes(x_range=[0, 4 * PI, PI], y_range=[-1.3, 1.3, 1], x_length=11.6, y_length=3.6, tips=False,
                    axis_config={"color": GREY_B, "stroke_width": 2}).move_to([0, -1.0, 0])
        title = Text("frequency = how fast the wave repeats", font_size=34).move_to([0, 3.1, 0])
        g1 = axes.plot(np.sin, x_range=[0, 4 * PI, 0.02], color=BLUE_C, stroke_width=4)
        g2 = axes.plot(lambda x: np.sin(2 * x), x_range=[0, 4 * PI, 0.01], color=YELLOW, stroke_width=4)
        g3 = axes.plot(lambda x: np.sin(x / 2), x_range=[0, 4 * PI, 0.02], color=TEAL, stroke_width=4)

        def entry(name, color, factor=None):
            parts = [Line(ORIGIN, 0.45 * RIGHT, color=color, stroke_width=5), Text(name, font_size=30, color=color)]
            if factor:
                parts.append(Text(factor, font_size=30, color=WHITE))
            return VGroup(*parts).arrange(RIGHT, buff=0.2)

        e1, e2, e3 = entry("sin(x)", BLUE_C), entry("sin(2x)", YELLOW, "×2"), entry("sin(x/2)", TEAL, "×½")
        legend = VGroup(e1, e2, e3).arrange(RIGHT, buff=1.0).move_to([0, 2.0, 0])
        fast = label("faster", YELLOW, 26).next_to(e2, DOWN, buff=0.2)
        slow = label("slower", TEAL, 26).next_to(e3, DOWN, buff=0.2)
        self.play(*self.clear_anims(), FadeIn(title, shift=0.2 * DOWN), FadeIn(axes), FadeIn(e1), Create(g1),
                  run_time=0.8)
        self.at("multiply")
        self.play(FadeIn(e2, shift=0.2 * DOWN), run_time=0.4)
        self.at("faster")
        self.play(Create(g2), FadeIn(fast), g1.animate.set_stroke(opacity=0.45), run_time=0.9)
        self.at("multiply")
        self.play(FadeIn(e3, shift=0.2 * DOWN), run_time=0.4)
        self.at("slows")
        self.play(Create(g3), FadeIn(slow), g2.animate.set_stroke(opacity=0.45), run_time=0.8)
        self.end_section()

    # 6. Several frequencies: every position gets its own combination -----------------------------
    def s6_several(self):
        self.section(6)
        XA, XB = -4.7, 2.1
        SX = (XB - XA) / N_POS
        ROWS = [2.2, 1.0, -0.2, -1.4]
        AMP = 0.42

        def xp(p):
            return XA + p * SX

        base, waves, names = VGroup(), VGroup(), VGroup()
        for y, w, c, n in zip(ROWS, FREQS, FREQ_COLORS, FREQ_NAMES):
            base.add(Line([XA, y, 0], [XB + 0.1, y, 0], color=GREY_D, stroke_width=1.5))
            waves.add(ParametricFunction(lambda p, y=y, w=w: np.array([xp(p), y + AMP * np.sin(w * p), 0]),
                                         t_range=[0, N_POS, 0.04], color=c, stroke_width=4))
            names.add(label(n, c, 24).move_to([XA - 0.3, y, 0], aligned_edge=RIGHT))
        pos_axis = Arrow([XA, -2.2, 0], [XB + 0.4, -2.2, 0], buff=0, color=GREY_B, stroke_width=2,
                         max_tip_length_to_length_ratio=0.03)
        ticks = VGroup(*[label(str(p), GREY_B, 22).move_to([xp(p), -2.5, 0]) for p in (0, 10, 20)])
        pos_name = label("position", GREY_B, 24).next_to(pos_axis, RIGHT, buff=0.15)
        illus = caption("illustrative frequencies")
        self.play(*self.clear_anims(), run_time=0.3)
        self.at("several")
        self.play(FadeIn(base), FadeIn(names), FadeIn(pos_axis), FadeIn(ticks), FadeIn(pos_name), FadeIn(illus),
                  LaggedStart(*[Create(w) for w in waves], lag_ratio=0.2), run_time=1.2)

        pos = ValueTracker(3)

        def p_now():
            return int(round(pos.get_value()))

        line = always_redraw(lambda: Line([xp(p_now()), 2.75, 0], [xp(p_now()), -1.95, 0], color=YELLOW,
                                          stroke_width=3))
        line_lab = always_redraw(lambda: label(f"pos = {p_now()}", YELLOW, 26).move_to([xp(p_now()), 3.1, 0]))

        def dots():
            p = p_now()
            return VGroup(*[Dot([xp(p), y + AMP * np.sin(w * p), 0], radius=0.08, color=c)
                            .set_stroke(WHITE, 1.5, background=False)
                            for y, w, c in zip(ROWS, FREQS, FREQ_COLORS)])

        def col():
            v = column([fmt(np.sin(w * p_now())) for w in FREQS], FREQ_COLORS)
            return v.move_to([3.5, 0.4, 0])

        crossings, values = always_redraw(dots), always_redraw(col)
        head = label("values", GREY_B, 24).move_to([3.5, 1.85, 0])
        self.at("position")
        self.play(FadeIn(line), FadeIn(line_lab), FadeIn(crossings), FadeIn(values), FadeIn(head), run_time=0.5)
        self.at("values")
        self.play(pos.animate.set_value(10), run_time=1.1)

        clock = clock_icon(np.array([5.75, 0.8, 0]))
        clock_lab = label("like clock hands", GREY_B, 22).move_to([5.75, -0.1, 0])
        self.at("clock")
        self.play(FadeIn(clock, scale=0.7), FadeIn(clock_lab), run_time=0.45)
        self.end_section()

    # 7. Rotating a 2-D vector --------------------------------------------------------------------
    def s7_rotate(self):
        self.section(7)
        O, U = np.array([-4.3, -1.9, 0]), 2.5
        axes = VGroup(Line(O + 1.0 * LEFT, O + (U + 0.7) * RIGHT), Line(O + 0.8 * DOWN, O + (U + 0.5) * UP))
        axes.set_stroke(GREY_D, 2)
        ang = ValueTracker(0.0)
        vec = always_redraw(lambda: Arrow(O, O + U * unit(ang.get_value()), buff=0, color=YELLOW, stroke_width=7,
                                          max_tip_length_to_length_ratio=0.12))
        v_lab = label("v = (1, 0)", WHITE, 28).next_to(O + U * RIGHT, DOWN, buff=0.25)
        self.clear(run_time=0.35)
        self.at("rotate")
        self.play(FadeIn(axes), GrowArrow(vec), run_time=0.6)
        self.at("vector")
        self.play(FadeIn(v_lab, shift=0.1 * UP), run_time=0.4)

        head = label("rotate (x, y) by θ:", GREY_B, 28).move_to([0, 3.15, 0])
        formula = Text("(x cos θ − y sin θ,  x sin θ + y cos θ)", font_size=36,
                       t2c={"cos θ": COS_COLOR, "sin θ": SIN_COLOR}).move_to([0, 2.45, 0])
        self.at("mix")
        self.play(FadeIn(head, shift=0.2 * DOWN), FadeIn(formula, shift=0.2 * DOWN), run_time=0.6)

        ghost = Arrow(O, O + U * RIGHT, buff=0, color=GREY_C, stroke_width=4, max_tip_length_to_length_ratio=0.12)
        ghost.set_opacity(0.5)
        arc = always_redraw(lambda: Arc(radius=0.8, start_angle=0, angle=max(ang.get_value(), 1e-3), arc_center=O,
                                        color=WHITE, stroke_width=3))
        deg = always_redraw(lambda: label(f"{np.degrees(ang.get_value()):.0f}°", WHITE, 26)
                            .move_to(O + 1.2 * unit(ang.get_value() / 2))
                            .set_opacity(min(1.0, ang.get_value() / (0.4 * THETA))))
        self.at("angle")
        self.add(ghost, arc, deg)
        self.bring_to_front(vec)
        self.play(ang.animate.set_value(THETA), run_time=0.7)
        tip_lab = label(f"({V_ROT[0]:.3f}, {V_ROT[1]:.1f})", YELLOW, 28)
        tip_lab.next_to(O + U * unit(THETA), UR, buff=0.1)
        self.play(FadeIn(tip_lab, shift=0.1 * UP), run_time=0.3)

        circ = DashedVMobject(Arc(radius=U, start_angle=-0.15, angle=PI / 2 + 0.3, arc_center=O), num_dashes=30)
        circ.set_stroke(GREY_B, 2)
        length = VGroup(label("length = 1 → 1", WHITE, 34), label("unchanged", GREEN, 30))
        length.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([0.9, 0.75, 0], aligned_edge=LEFT)
        self.at("length")
        self.play(Create(circ), FadeIn(length, shift=0.2 * LEFT), run_time=0.6)

        direction = label("direction: 0° → 30°", YELLOW, 34).move_to([0.9, -0.6, 0], aligned_edge=LEFT)
        self.at("direction")
        vec.clear_updaters()
        self.play(FadeIn(direction, shift=0.2 * LEFT), Indicate(vec, color=YELLOW, scale_factor=1.08), run_time=0.6)
        self.end_section()

    # 8. Relative position becomes a relative angle -----------------------------------------------
    def s8_relative(self):
        self.section(8)
        O, L = np.array([-2.8, -1.8, 0]), 2.6
        phi, a1, a2 = ValueTracker(0.0), ValueTracker(0.0), ValueTracker(0.0)

        base = always_redraw(lambda: DashedLine(O, O + L * unit(phi.get_value()), color=GREY_B, stroke_width=3,
                                                dash_length=0.12))
        start = always_redraw(lambda: beyond(label("start", GREY_B, 24), O, unit(phi.get_value()), L + 0.1))
        ticks = always_redraw(lambda: VGroup(*[
            Line(O + 2.75 * unit(phi.get_value() + k * STEP), O + 2.95 * unit(phi.get_value() + k * STEP),
                 color=GREY_C, stroke_width=2) for k in range(1, 8)]))
        hub = Dot(O, radius=0.07, color=WHITE)
        self.play(*self.clear_anims(), FadeIn(base), FadeIn(start), FadeIn(ticks), FadeIn(hub), run_time=0.45)

        def arrow(tr, color):
            return always_redraw(lambda: Arrow(O, O + L * unit(phi.get_value() + tr.get_value()), buff=0,
                                               color=color, stroke_width=7, max_tip_length_to_length_ratio=0.12))

        def tip_label(tr, text, color):
            return always_redraw(lambda: beyond(label(text, color, 28), O, unit(phi.get_value() + tr.get_value()),
                                                L + 0.12).set_opacity(min(1.0, tr.get_value() / STEP)))

        arr1, arr2 = arrow(a1, BLUE_C), arrow(a2, TEAL)
        lab1, lab2 = tip_label(a1, "2θ", BLUE_C), tip_label(a2, "5θ", TEAL)
        note = caption("1 step = θ = 20° (illustrative)")
        self.at("2")
        self.add(arr1, lab1)
        self.play(a1.animate.set_value(2 * STEP), FadeIn(note), run_time=0.8)
        self.at("5")
        self.add(arr2, lab2)
        self.play(a2.animate.set_value(5 * STEP), run_time=0.9)

        def between():
            return Arc(radius=1.3, start_angle=phi.get_value() + a1.get_value(),
                       angle=a2.get_value() - a1.get_value(), arc_center=O, color=YELLOW, stroke_width=5)

        def between_lab():
            mid = phi.get_value() + (a1.get_value() + a2.get_value()) / 2
            return label("3θ", YELLOW, 32).move_to(O + 1.75 * unit(mid))

        gap, gap_lab = always_redraw(between), always_redraw(between_lab)
        diff = Text("5θ − 2θ = 3θ", font_size=38, t2c={"5θ": TEAL, "2θ": BLUE_C, "3θ": YELLOW}).move_to([3.6, 1.2, 0])
        self.at("3")
        self.play(Create(gap), FadeIn(gap_lab), FadeIn(diff, shift=0.2 * LEFT), run_time=0.6)

        same = VGroup(label("start somewhere else:", GREY_B, 28),
                      Text("still 3θ", font_size=34, color=YELLOW)).arrange(DOWN, buff=0.2).move_to([3.6, -0.1, 0])
        self.at("started")
        self.play(phi.animate.set_value(np.radians(50)), FadeIn(same, shift=0.2 * LEFT), run_time=1.0)

        cap = Text("relative position → relative angle", font_size=36, color=YELLOW).move_to([0.8, -2.85, 0])
        self.at("relative")
        self.play(FadeIn(cap, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # 9. The NumPy code ---------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, font_size=26)
        code.move_to([0, 0.6, 0])
        row_h = hl.height
        hl.stretch_to_fit_height(2 * row_h)
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT)
        hl.move_to([hl.get_x(), (code.line_numbers[1].get_y() + code.line_numbers[2].get_y()) / 2, 0])
        self.play(*self.clear_anims(), FadeIn(code, shift=0.3 * UP), run_time=0.5)
        self.at("matrix")
        self.play(Create(hl), run_time=0.4)
        self.at("cosine")
        self.play(Indicate(code.code_lines[1], color=COS_COLOR, scale_factor=1.03),
                  Indicate(code.code_lines[2], color=COS_COLOR, scale_factor=1.03), run_time=0.6)
        self.at("length")
        self.play(hl.animate.stretch_to_fit_height(row_h).match_y(code.line_numbers[5]), run_time=0.4)
        ok = VGroup(Text("✓", font_size=40, color=GREEN), label("length stays exactly 1.0", GREEN, 32))
        ok.arrange(RIGHT, buff=0.25).next_to(code, DOWN, buff=0.5)
        self.at("one")
        self.play(FadeIn(ok, shift=0.2 * UP), run_time=0.4)
        self.end_section()

    # 10. Outro ------------------------------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        card = used_in_card(USED_IN).move_to(ORIGIN)
        self.at("see")
        self.play(*self.clear_anims(), FadeIn(card, shift=0.2 * UP), run_time=0.7)
        self.at("four")
        self.play(Indicate(card[1][0], scale_factor=1.08), run_time=0.6)
        nxt = next_up_card(NEXT)
        self.at("learning")
        self.play(FadeOut(card), FadeIn(nxt, shift=0.2 * UP), run_time=0.5)
