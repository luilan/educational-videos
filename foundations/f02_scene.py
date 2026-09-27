"""Foundations F2 — The Dot Product.

Render from the repo root:  ./render.sh foundations f02
"""
import numpy as np
from manim import *

from common import code_panel, finish, highlight, next_up_card, used_in_card
from f02_script import LABEL, NEXT, TAGLINE, TITLE, USED_IN
from intro import play_token_intro
from voiced_scene import VoicedScene

MONO = "DejaVu Sans Mono"
A_COLOR, B_COLOR = BLUE, TEAL

# ------------------------------------------------------------------ the worked example
A = np.array([3.0, 1.0])
B = np.array([2.0, 2.0])
DOT = float(A @ B)                      # 8
NA, NB = np.linalg.norm(A), np.linalg.norm(B)
COS = DOT / (NA * NB)                   # 0.894
THETA = np.degrees(np.arccos(COS))      # 26.6°
SHADOW = DOT / NA                       # 2.53, length of b's projection onto a
FOOT = DOT / NA ** 2 * A                # (2.4, 0.8), foot of the perpendicular from b's tip
assert DOT == 8 and f"{NA:.2f}" == "3.16" and f"{NB:.2f}" == "2.83"
assert f"{COS:.3f}" == "0.894" and f"{THETA:.1f}" == "26.6" and f"{SHADOW:.2f}" == "2.53"
assert round(round(SHADOW, 2) * round(NA, 2)) == 8
assert A @ np.array([-1, 3]) == 0 and A @ np.array([-2, -1]) == -7 and A @ np.array([3, 2]) == 11

CODE = """a = np.array([3, 1])
b = np.array([2, 2])
a @ b                                     # 8
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
cos                                       # 0.894"""

# ------------------------------------------------------------------ the 2-D plane (sections 4–6)
U = 0.9                                 # scene units per coordinate unit
O = np.array([-3.6, -1.1, 0.0])         # where the origin sits on screen


def p(v):
    """Plane coordinates → scene point."""
    return O + U * np.array([v[0], v[1], 0.0])


# b's journey in section 4 (and back in section 5): waypoints as (angle°, length).
# Positions between waypoints are interpolated in polar form; the readout always uses the drawn b.
WAYPOINTS = [
    (np.degrees(np.arctan2(2, 2)), np.hypot(2, 2)),        # (2, 2)   start: a · b = 8
    (np.degrees(np.arctan2(2, 3)), np.hypot(3, 2)),        # (3, 2)   same way: 11
    (np.degrees(np.arctan2(3, -1)), np.hypot(-1, 3)),      # (−1, 3)  right angle: 0
    (np.degrees(np.arctan2(-1, -2)) + 360, np.hypot(-2, -1)),  # (−2, −1) opposite: −7
    (np.degrees(np.arctan2(2, 2)), np.hypot(2, 2)),        # back to (2, 2), turning clockwise
]


def b_at(t):
    k = int(np.clip(np.floor(t), 0, len(WAYPOINTS) - 2))
    f = t - k
    (a0, r0), (a1, r1) = WAYPOINTS[k], WAYPOINTS[k + 1]
    ang, r = np.radians(a0 + (a1 - a0) * f), r0 + (r1 - r0) * f
    return np.array([r * np.cos(ang), r * np.sin(ang)])


def fmt(v, nd=2):
    s = f"{v:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−")


def label(text, color=GREY_B, font_size=26, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def vec_arrow(start, end, color, width=6):
    return Arrow(start, end, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.2)


def bracket_pair(left, right, top, bottom, color=GREY_B):
    def one(x, d):
        pts = [[x + d, top, 0], [x, top, 0], [x, bottom, 0], [x + d, bottom, 0]]
        return VMobject(stroke_color=color, stroke_width=3).set_points_as_corners(pts)
    return VGroup(one(left, 0.15), one(right, -0.15))


def column(entries, x, ys, color, font_size=44):
    """A column vector: Text entries at (x, y) plus bracket lines. Returns (group, cells, brackets)."""
    cells = VGroup(*[Text(e, font_size=font_size, color=color).move_to([x, y, 0]) for e, y in zip(entries, ys)])
    hw = max(c.width for c in cells) / 2 + 0.25
    br = bracket_pair(x - hw, x + hw, ys[0] + 0.45, ys[-1] - 0.45, color)
    return VGroup(br, cells), cells, br


def strip(n_head, n_tail, color, seed, cell=0.5, pitch=0.56):
    """A long vector drawn as a strip of shaded cells: n_head cells, '…', n_tail cells (left edge at x=0)."""
    rng = np.random.default_rng(seed)
    cells = VGroup()
    xs = [i * pitch for i in range(n_head)] + [(n_head + 1 + i) * pitch for i in range(n_tail)]
    for x in xs:
        cells.add(Square(cell, stroke_color=color, stroke_width=2, fill_color=color,
                         fill_opacity=float(rng.uniform(0.15, 0.85))).move_to([x, 0, 0]))
    dots = Text("…", font_size=36, color=color).move_to([n_head * pitch, 0, 0])
    return VGroup(cells, dots), cells, dots


def two_arrow_icon(ang1, ang2, length=1.3):
    o = np.array([-0.5, -0.45, 0])
    d1 = np.array([np.cos(np.radians(ang1)), np.sin(np.radians(ang1)), 0])
    d2 = np.array([np.cos(np.radians(ang2)), np.sin(np.radians(ang2)), 0])
    return VGroup(vec_arrow(o, o + length * d1, A_COLOR, 5), vec_arrow(o, o + length * d2, B_COLOR, 5),
                  Dot(o, radius=0.05, color=GREY_B))


class DotProductVideo(VoicedScene):
    VIDEO = "f02"

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
        self.s1_question()
        self.s2_recipe()
        self.s3_dimensions()
        self.s4_agreement()
        self.s5_shadow()
        self.s6_formula()
        self.s7_cosine()
        self.s8_everywhere()
        self.s9_code()
        self.s10_outro()
        finish(self)

    # 1. The question ----------------------------------------------------------------------------
    def s1_question(self):
        self.section(1)
        o = np.array([-4.3, -1.7, 0])
        ang = ValueTracker(118.0)
        a_dir = np.array([3, 1, 0]) / np.sqrt(10)
        a_arr = vec_arrow(o, o + 3.2 * a_dir, A_COLOR)
        a_lab = Text("a", font_size=34, color=A_COLOR).move_to(o + 3.55 * a_dir)

        def b_dir():
            t = np.radians(ang.get_value())
            return np.array([np.cos(t), np.sin(t), 0])

        b_arr = always_redraw(lambda: vec_arrow(o, o + 2.9 * b_dir(), B_COLOR))
        b_lab = always_redraw(lambda: Text("b", font_size=34, color=B_COLOR).move_to(o + 3.25 * b_dir()))
        dot = Dot(o, radius=0.07, color=GREY_B)
        self.at("vectors")
        self.play(FadeIn(dot), GrowArrow(a_arr), FadeIn(a_lab), FadeIn(b_arr), FadeIn(b_lab), run_time=0.8)

        question = label("How much do they point the same way?", WHITE, 34).move_to([0, 2.9, 0])
        self.at("how")
        self.play(FadeIn(question, shift=0.2 * DOWN), run_time=0.6)
        self.at("same")
        self.play(ang.animate.set_value(29.0), run_time=1.0)

        name = label("the dot product", font_size=30).move_to([3.0, 0.9, 0])
        self.at("dot")
        self.play(FadeIn(name, shift=0.2 * UP), run_time=0.5)

        box = RoundedRectangle(corner_radius=0.15, width=2.4, height=1.1, stroke_color=YELLOW,
                               fill_color=YELLOW, fill_opacity=0.15)
        badge = VGroup(box, Text("a · b", font_size=44, color=YELLOW).move_to(box)).move_to([3.0, -0.4, 0])
        pointer = Arrow([-0.2, -0.7, 0], badge.get_left() + 0.1 * LEFT, buff=0.1, color=GREY_B, stroke_width=4)
        caption = label("one number", font_size=26).next_to(badge, DOWN, buff=0.3)
        self.at("number")
        self.play(GrowArrow(pointer), FadeIn(badge, scale=0.8), FadeIn(caption), run_time=0.5)
        self.end_section()

    # 2. The recipe ------------------------------------------------------------------------------
    def s2_recipe(self):
        self.section(2)
        ys = [0.9, -0.2]
        a_col, a_cells, a_br = column(["3", "1"], -3.9, ys, A_COLOR)
        b_col, b_cells, b_br = column(["2", "2"], -1.2, ys, B_COLOR)
        a_name = Text("a", font_size=34, color=A_COLOR).next_to(a_br, UP, buff=0.3)
        b_name = Text("b", font_size=34, color=B_COLOR).next_to(b_br, UP, buff=0.3)
        mid = Text("·", font_size=60).move_to([-2.55, (ys[0] + ys[1]) / 2, 0])
        self.play(*self.clear_anims(), run_time=0.4)
        self.play(FadeIn(a_col), FadeIn(b_col), FadeIn(a_name), FadeIn(b_name), FadeIn(mid), run_time=0.5)

        links = VGroup(*[Line([a_br[1].get_right()[0] + 0.12, y, 0], [b_br[0].get_left()[0] - 0.12, y, 0],
                              color=YELLOW, stroke_width=4) for y in ys])
        step1 = label("multiply matching entries,", WHITE, 30)
        step2 = label("then add them up", WHITE, 30)
        steps = VGroup(step1, step2).arrange(RIGHT, buff=0.25).move_to([0, 2.9, 0])
        self.at("multiply")
        self.play(Create(links), FadeIn(step1), run_time=0.6)
        self.at("add")
        self.play(FadeIn(step2), run_time=0.5)

        x0 = 1.6
        prods = VGroup(
            Text("3·2 = 6", font_size=44, t2c={"[0:1]": A_COLOR, "[2:3]": B_COLOR}),
            Text("1·2 = 2", font_size=44, t2c={"[0:1]": A_COLOR, "[2:3]": B_COLOR}))
        for t, y in zip(prods, ys):
            t.move_to([x0, y, 0], aligned_edge=LEFT).set_y(y)
        pointers = VGroup(*[Arrow([b_br[1].get_right()[0] + 0.15, y, 0], [x0 - 0.2, y, 0], buff=0,
                                  color=GREY_B, stroke_width=3, max_tip_length_to_length_ratio=0.12)
                            for y in ys])
        self.at("3")
        self.play(Indicate(VGroup(a_cells[0], b_cells[0]), scale_factor=1.3), run_time=0.5)
        self.at("6")
        self.play(GrowArrow(pointers[0]), FadeIn(prods[0], shift=0.2 * RIGHT), run_time=0.5)
        self.at("1")
        self.play(Indicate(VGroup(a_cells[1], b_cells[1]), scale_factor=1.3), run_time=0.4)
        self.at("2")
        self.play(GrowArrow(pointers[1]), FadeIn(prods[1], shift=0.2 * RIGHT), run_time=0.5)

        rule = Line([x0 - 0.1, -0.85, 0], [prods.get_right()[0] + 0.1, -0.85, 0], color=GREY_B, stroke_width=2)
        total = VGroup(Text("6 + 2 =", font_size=44), Text("8", font_size=44, color=YELLOW))
        total.arrange(RIGHT, buff=0.35).move_to([x0, -1.5, 0], aligned_edge=LEFT).set_y(-1.5)
        ring = SurroundingRectangle(total[1], color=YELLOW, buff=0.12, corner_radius=0.08)
        self.at("8")
        self.play(Create(rule), FadeIn(total, shift=0.2 * DOWN), Create(ring), run_time=0.6)
        self.end_section()

    # 3. Any number of dimensions ----------------------------------------------------------------
    def s3_dimensions(self):
        self.section(3)
        self.play(*self.clear_anims(), run_time=0.5)
        a_strip, a_cells, _ = strip(11, 2, A_COLOR, seed=2)
        b_strip, b_cells, _ = strip(11, 2, B_COLOR, seed=5)
        shift = np.array([-a_strip.get_center()[0] + 0.3, 0, 0])
        a_strip.shift(shift + 1.3 * UP)
        b_strip.shift(shift + 0.0 * UP)
        a_name = Text("a", font_size=34, color=A_COLOR).next_to(a_strip, LEFT, buff=0.4)
        b_name = Text("b", font_size=34, color=B_COLOR).next_to(b_strip, LEFT, buff=0.4)
        head = label("same recipe, any number of dimensions", WHITE, 32).move_to([0, 2.9, 0])
        self.at("dimensions")
        self.play(FadeIn(head), LaggedStart(*[FadeIn(m, shift=0.2 * RIGHT) for m in
                                             [a_name, a_strip, b_name, b_strip]], lag_ratio=0.2), run_time=0.9)

        left_x = a_cells[0].get_left()[0]
        right_x = a_cells[-1].get_right()[0]
        top_y, bot_y = a_strip.get_top()[1] + 0.15, b_strip.get_bottom()[1] - 0.15
        ends = VGroup(DashedLine([left_x, top_y, 0], [left_x, bot_y, 0], color=GREY_B, stroke_width=2),
                      DashedLine([right_x, top_y, 0], [right_x, bot_y, 0], color=GREY_B, stroke_width=2))
        same = label("same length", font_size=26).move_to([(left_x + right_x) / 2, bot_y - 0.35, 0])
        self.at("length")
        self.play(Create(ends), FadeIn(same), run_time=0.6)

        gap_y = (a_strip.get_bottom()[1] + b_strip.get_top()[1]) / 2
        times = VGroup(*[Text("×", font_size=24, color=YELLOW).move_to([c.get_x(), gap_y, 0]) for c in a_cells])
        idx = VGroup(label("1", font_size=22).next_to(a_cells[0], UP, buff=0.12),
                     label("768", font_size=22).next_to(a_cells[-1], UP, buff=0.12))
        mults = label("768 multiplications", WHITE, 36).move_to([0, -1.55, 0])
        self.at("768")
        self.play(LaggedStart(*[FadeIn(t, scale=0.5) for t in times], lag_ratio=0.08), FadeIn(idx),
                  FadeIn(mults, shift=0.2 * UP), run_time=0.8)

        sigma = Text("Σ", font_size=64, color=YELLOW)
        one_sum = label("→ one sum", WHITE, 36)
        total = VGroup(sigma, one_sum).arrange(RIGHT, buff=0.3).move_to([0, -2.75, 0])
        self.at("one")
        self.play(FadeIn(total, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # 4. Agreement ---------------------------------------------------------------------------------
    def s4_agreement(self):
        self.section(4)
        self.play(*self.clear_anims(), run_time=0.5)
        plane = NumberPlane(x_range=[-3, 4, 1], y_range=[-2, 4, 1], x_length=7 * U, y_length=6 * U,
                            background_line_style={"stroke_color": BLUE_E, "stroke_width": 1.5,
                                                   "stroke_opacity": 0.5},
                            axis_config={"stroke_color": GREY_B, "stroke_width": 2})
        plane.shift(O - plane.c2p(0, 0))
        self.plane = plane
        self.bt = ValueTracker(0.0)
        a_arr = vec_arrow(p([0, 0]), p(A), A_COLOR)
        a_lab = Text("a", font_size=32, color=A_COLOR).move_to(p(A * 1.12))
        b_arr = always_redraw(lambda: vec_arrow(p([0, 0]), p(b_at(self.bt.get_value())), B_COLOR))

        def b_label():
            v = b_at(self.bt.get_value())
            return Text("b", font_size=32, color=B_COLOR).move_to(p(v + 0.4 * v / np.linalg.norm(v)))

        b_lab = always_redraw(b_label)
        self.a_arr, self.a_lab, self.b_arr, self.b_lab = a_arr, a_lab, b_arr, b_lab
        self.play(Create(plane), GrowArrow(a_arr), FadeIn(a_lab), FadeIn(b_arr), FadeIn(b_lab), run_time=0.9)

        rx = 1.0  # left edge of the readout column
        head = label("dot product = agreement", WHITE, 32).move_to([3.6, 2.6, 0])
        a_txt = Text("a = (3, 1)", font_size=32, color=A_COLOR)
        a_txt.move_to([rx, 1.4, 0], aligned_edge=LEFT).set_y(1.4)

        def b_txt():
            v = b_at(self.bt.get_value())
            t = Text(f"b = ({fmt(v[0])}, {fmt(v[1])})", font_size=32, color=B_COLOR)
            return t.move_to([rx, 0.7, 0], aligned_edge=LEFT).set_y(0.7)

        def dot_txt():
            v = b_at(self.bt.get_value())
            d = A[0] * v[0] + A[1] * v[1]  # computed from the drawn b
            col = GREEN if d > 0.005 else RED if d < -0.005 else WHITE
            t = VGroup(Text("a · b =", font_size=42), Text(fmt(d), font_size=42, color=col))
            t.arrange(RIGHT, buff=0.25, aligned_edge=DOWN)
            return t.move_to([rx, -0.35, 0], aligned_edge=LEFT).set_y(-0.35)

        b_read, d_read = always_redraw(b_txt), always_redraw(dot_txt)
        self.at("agreement")
        self.play(FadeIn(head), FadeIn(a_txt), FadeIn(b_read), FadeIn(d_read), run_time=0.6)

        cap_pos = [3.6, -1.6, 0]
        cap1 = label("same way → large positive", GREEN, 28).move_to(cap_pos)
        self.at("same")
        self.play(self.bt.animate.set_value(1.0), FadeIn(cap1), run_time=1.0)

        cap2 = label("right angle → exactly 0", WHITE, 28).move_to(cap_pos)
        self.at("right")
        self.play(self.bt.animate.set_value(2.0), FadeOut(cap1), run_time=0.8)
        mark = RightAngle(Line(p([0, 0]), p(A)), Line(p([0, 0]), p(b_at(2.0))), length=0.3,
                          color=YELLOW, stroke_width=3)
        self.at("dot")
        self.play(Create(mark), FadeIn(cap2), run_time=0.5)

        cap3 = label("opposite → negative", RED, 28).move_to(cap_pos)
        self.at("opposite")
        self.play(FadeOut(mark), FadeOut(cap2), run_time=0.3)
        self.play(self.bt.animate.set_value(3.0), run_time=1.0)
        self.at("negative")
        self.play(FadeIn(cap3), Indicate(d_read, scale_factor=1.1, color=RED), run_time=0.6)
        self.end_section()
        self.readout = VGroup(head, a_txt, b_read, d_read, cap3)

    # 5. The shadow ----------------------------------------------------------------------------------
    def s5_shadow(self):
        self.section(5)
        self.play(FadeOut(self.readout), self.bt.animate.set_value(4.0), run_time=1.0)
        tip, foot = p(B), p(FOOT)
        drop = DashedLine(tip, foot, color=GREY_B, stroke_width=3)
        mark = RightAngle(Line(foot, p([0, 0])), Line(foot, tip), length=0.22, color=GREY_B, stroke_width=2)
        shadow = Line(p([0, 0]), foot, color=YELLOW, stroke_width=10)
        self.at("project")
        self.play(Create(drop), run_time=0.6)
        self.play(Create(shadow), FadeIn(mark), run_time=0.6)

        normal = np.array([1, -3]) / np.sqrt(10)
        sh_lab = label("shadow", YELLOW, 28).move_to(p(FOOT / 2 + 1.05 * normal))
        self.at("shadow")
        self.play(FadeIn(sh_lab), run_time=0.4)
        self.at("length")
        self.play(Indicate(shadow, scale_factor=1.05), run_time=0.6)

        formula = Text("a · b = |shadow| × ‖a‖", font_size=36, t2c={"|shadow|": YELLOW, "‖a‖": A_COLOR})
        formula.move_to([3.6, 1.0, 0])
        check = Text(f"8 ≈ {SHADOW:.2f} × {NA:.2f}", font_size=36, color=GREY_B).next_to(formula, DOWN, buff=0.5)
        self.at("times")
        self.play(FadeIn(formula, shift=0.2 * UP), run_time=0.6)
        self.at("arrow")
        self.play(FadeIn(check), run_time=0.4)
        self.end_section()
        self.shadow_bits = VGroup(drop, mark, shadow, sh_lab, formula, check)

    # 6. The second formula ------------------------------------------------------------------------
    def s6_formula(self):
        self.section(6)
        self.play(FadeOut(self.shadow_bits), run_time=0.5)
        pieces = VGroup(Text("a · b =", font_size=38), Text("‖a‖", font_size=38, color=A_COLOR),
                        Text("‖b‖", font_size=38, color=B_COLOR), Text("cos θ", font_size=38, color=YELLOW))
        pieces.arrange(RIGHT, buff=0.3).move_to([3.6, 1.6, 0])
        vals = VGroup(label(f"‖a‖ = √10 ≈ {NA:.2f}", A_COLOR, 30), label(f"‖b‖ = √8 ≈ {NB:.2f}", B_COLOR, 30),
                      label(f"θ ≈ {THETA:.1f}°", YELLOW, 30))
        vals.arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([3.6, -0.5, 0])
        self.at("formula")
        self.play(FadeIn(pieces[0]), run_time=0.5)
        self.at("length")
        self.at("a")
        self.play(FadeIn(pieces[1], shift=0.2 * UP), FadeIn(vals[0]), Indicate(self.a_arr, scale_factor=1, color=WHITE),
                  Indicate(self.a_lab, scale_factor=1.4, color=A_COLOR),
                  run_time=0.6)
        self.at("b")
        self.play(FadeIn(pieces[2], shift=0.2 * UP), FadeIn(vals[1]), Indicate(self.b_arr, scale_factor=1, color=WHITE),
                  Indicate(self.b_lab, scale_factor=1.4, color=B_COLOR),
                  run_time=0.6)
        self.at("cosine")
        self.play(FadeIn(pieces[3], shift=0.2 * UP), run_time=0.5)
        arc = Angle(Line(p([0, 0]), p(A)), Line(p([0, 0]), p(B)), radius=1.0, color=YELLOW, stroke_width=4)
        mid = np.radians((np.degrees(np.arctan2(1, 3)) + 45) / 2)
        theta = Text("θ", font_size=30, color=YELLOW).move_to(p([0, 0]) + 1.35 * np.array([np.cos(mid), np.sin(mid), 0]))
        self.at("angle")
        self.play(Create(arc), FadeIn(theta), FadeIn(vals[2]), run_time=0.6)
        self.end_section()
        self.formula = pieces

    # 7. Cosine similarity ---------------------------------------------------------------------------
    def s7_cosine(self):
        self.section(7)
        self.b_arr.clear_updaters()
        self.b_lab.clear_updaters()
        self.play(*self.clear_anims(keep=[self.formula]),
                  self.formula.animate.scale(0.85).move_to([0, 2.9, 0]), run_time=0.8)

        lhs = Text("cos θ", font_size=40, color=YELLOW)
        rhs = Text("= a · b / (‖a‖ ‖b‖)", font_size=40)
        line1 = VGroup(lhs, rhs).arrange(RIGHT, buff=0.4).move_to([0, 1.75, 0])
        line2 = Text(f"= 8 / (√10 × √8) ≈ {COS:.3f}", font_size=40)
        line2.move_to(rhs, aligned_edge=LEFT).set_y(0.8)
        self.at("divide")
        self.play(FadeIn(line1, shift=0.2 * UP), run_time=0.6)
        self.at("whats")
        self.play(FadeIn(line2, shift=0.2 * UP), run_time=0.6)

        name = label("cosine similarity", YELLOW, 34).move_to([0, -0.45, 0])
        ring = SurroundingRectangle(lhs, color=YELLOW, buff=0.12, corner_radius=0.08)
        y = -2.3
        axis = Line([-5, y, 0], [5, y, 0], color=GREY_B, stroke_width=3)
        ticks = VGroup(*[Line([x, y - 0.14, 0], [x, y + 0.14, 0], color=GREY_B, stroke_width=3)
                         for x in (-5, -2.5, 0, 2.5, 5)])
        ex_x = 5 * COS
        ex_dot = Dot([ex_x, y, 0], radius=0.1, color=YELLOW)
        ex_lab = label(f"our a, b: {COS:.3f}", YELLOW, 22).move_to([ex_x - 1.7, y - 0.95, 0])
        ex_ptr = Line(ex_lab.get_right() + 0.1 * RIGHT, ex_dot.get_bottom() + 0.05 * DOWN, color=YELLOW,
                      stroke_width=2)
        self.at("similarity")
        self.play(FadeIn(name), Create(ring), Create(axis), FadeIn(ticks), run_time=0.7)
        self.play(FadeIn(ex_dot, scale=0.5), FadeIn(ex_lab), Create(ex_ptr), run_time=0.4)

        def mark(x, num, word, color):
            n = label(num, WHITE, 28).move_to([x, y - 0.5, 0])
            w = label(word, color, 30).move_to([x, y + 0.6, 0])
            return n, w

        for cue, x, num, word, color in [("same", 5, "1", "same", GREEN), ("unrelated", 0, "0", "unrelated", GREY_B),
                                         ("opposite", -5, "−1", "opposite", RED)]:
            n, w = mark(x, num, word, color)
            self.at(cue)
            self.play(FadeIn(n), FadeIn(w, shift=0.2 * DOWN), run_time=0.5)
        self.end_section()

    # 8. Everywhere in LLMs ----------------------------------------------------------------------------
    def s8_everywhere(self):
        self.section(8)
        self.play(*self.clear_anims(), run_time=0.5)
        head = label("dot products everywhere", WHITE, 34).move_to([0, 2.9, 0])
        specs = [("Attention", "query · key", 15, 60), ("Output layer", "vector · every word", 35, 5),
                 ("MLP neuron", "weights · input", 10, 75)]
        panels, boxes, contents = VGroup(), [], []
        for k, (title, cap, a1, a2) in enumerate(specs):
            box = RoundedRectangle(corner_radius=0.2, width=4.0, height=3.9, stroke_color=GREY_D,
                                   stroke_width=3, fill_color=YELLOW, fill_opacity=0)
            t = Text(title, font_size=30).move_to([0, 1.4, 0])
            icon = two_arrow_icon(a1, a2).move_to([0, 0.05, 0])
            c = Text(cap, font_size=28).move_to([0, -1.35, 0])
            body = VGroup(t, icon, c)
            panel = VGroup(box, body).move_to([-4.35 + 4.35 * k, -0.4, 0])
            body.set_opacity(0.2)
            panels.add(panel)
            boxes.append(box)
            contents.append(body)
        self.at("everywhere")
        self.play(FadeIn(head), LaggedStart(*[FadeIn(pn) for pn in panels], lag_ratio=0.2), run_time=0.8)
        for k, cue in enumerate(["attention", "output", "neuron"]):
            self.at(cue)
            self.play(boxes[k].animate.set_stroke(YELLOW).set_fill(YELLOW, 0.08),
                      contents[k].animate.set_opacity(1), run_time=0.5)
        self.end_section()

    # 9. Code ----------------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, 26)
        grp = VGroup(code, hl)
        grp.scale(12.4 / code.width).move_to([0, 0.2, 0])
        hl.match_y(code.line_numbers[2])
        self.play(*self.clear_anims(), run_time=0.4)
        self.play(FadeIn(code, shift=0.3 * UP), run_time=0.6)
        self.at("sign")
        self.play(Create(hl), run_time=0.4)
        self.at("similarity")
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.end_section()

    # 10. Outro --------------------------------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        card = used_in_card(USED_IN).move_to(ORIGIN)
        rows = card[1]
        self.at("meet")
        self.play(*self.clear_anims(), FadeIn(card, shift=0.2 * UP), run_time=0.6)
        for k, cue in enumerate(["3", "5", "6", "8", "10"]):
            self.at(cue)
            self.play(Indicate(rows[k], scale_factor=1.08), run_time=0.35)
        nxt = next_up_card(NEXT)
        self.at("matrices")
        self.play(FadeOut(card), FadeIn(nxt, shift=0.2 * UP), run_time=0.7)
        self.end_section()
