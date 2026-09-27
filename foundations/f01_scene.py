"""Foundations F1 — Vectors: Lists of Numbers as Arrows.

Render from the repo root:  ./render.sh foundations f01
"""
import numpy as np
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card, token, token_row, used_in_card
from f01_script import LABEL, NEXT, TAGLINE, TITLE, USED_IN
from intro import play_token_intro
from voiced_scene import VoicedScene

POS_COLOR = BLUE_C
NEG_COLOR = RED_C

# The shared coordinate plane (sections 3, 5, 6, 7): 0.9 frame units per unit, origin left of centre
U = 0.9
PLANE_ORIGIN = np.array([-3.9, -1.35, 0.0])
PANEL_X = 3.9           # centre of the text column to the right of the plane

CODE = """import numpy as np
a = np.array([2, 1])
b = np.array([1, 2])
a + b                               # array([3, 3])
2 * a                               # array([4, 2])
np.linalg.norm(np.array([3, 4]))    # 5.0"""


# ---------------------------------------------------------------------------- helpers
def P(x, y):
    """Plane coordinates -> frame point."""
    return PLANE_ORIGIN + U * np.array([x, y, 0.0])


def make_plane():
    plane = NumberPlane(x_range=[-3, 5, 1], y_range=[-2, 4.5, 1], x_length=8 * U, y_length=6.5 * U,
                        background_line_style={"stroke_color": BLUE_E, "stroke_width": 1.5, "stroke_opacity": 0.7},
                        axis_config={"stroke_color": GREY_B, "stroke_width": 2})
    plane.shift(PLANE_ORIGIN - plane.c2p(0, 0))
    ticks = VGroup(*[Text(str(k), font_size=20, color=GREY_B).next_to(P(k, 0), DOWN, buff=0.1) for k in range(1, 5)],
                   *[Text(str(k), font_size=20, color=GREY_B).next_to(P(0, k), LEFT, buff=0.1) for k in range(1, 5)])
    return plane, ticks


def vec_arrow(x, y, color=BLUE, start=(0, 0)):
    return Arrow(P(*start), P(start[0] + x, start[1] + y), buff=0, color=color, stroke_width=6,
                 max_tip_length_to_length_ratio=0.18, max_stroke_width_to_length_ratio=10)


def caption(text):
    return Text(text, font_size=20, color=GREY_B).to_corner(DR, buff=0.3)


def value_color(v):
    v = max(-1.0, min(1.0, v))
    return interpolate_color(GREY_E, POS_COLOR if v >= 0 else NEG_COLOR, abs(v))


def cell_column(values, side=0.32):
    return VGroup(*[Square(side_length=side, stroke_color=GREY_D, stroke_width=1.5,
                           fill_color=value_color(v), fill_opacity=1) for v in values]).arrange(DOWN, buff=0)


class VectorsVideo(VoicedScene):
    VIDEO = "f01"

    # ------------------------------------------------------------------ stage helpers
    def on_stage(self, keep=()):
        top = [m for m in self.mobjects if not isinstance(m, ValueTracker)]
        inner = set()
        for m in top:
            for d in m.get_family()[1:]:
                inner.add(id(d))
        keep_ids = {id(d) for k in keep for d in k.get_family()}
        return [m for m in top if id(m) not in inner and id(m) not in keep_ids]

    def clear_anims(self, keep=()):
        return [FadeOut(m) for m in self.on_stage(keep)]

    def clear(self, run_time=0.45, keep=()):
        anims = self.clear_anims(keep)
        if anims:
            self.play(*anims, run_time=run_time)

    def construct(self):
        play_token_intro(self, TITLE, 0, TAGLINE, label=LABEL)
        self.s1_hook()
        self.s2_list()
        self.s3_arrow()
        self.s4_dimensions()
        self.s5_add()
        self.s6_scale()
        self.s7_length()
        self.s8_meaning()
        self.s9_code()
        self.s10_outro()
        finish(self)

    # 1. Everything is a vector ---------------------------------------------------------------
    def s1_hook(self):
        self.section(1)
        toks = token_row(["The", "cat", "sat", "on", "the"], buff=0.45).move_to([-0.8, 1.7, 0])
        rng = np.random.default_rng(1)
        cols = VGroup(*[cell_column(rng.uniform(-0.95, 0.95, 4)).next_to(t, DOWN, buff=0.55) for t in toks])
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in toks], lag_ratio=0.12, run_time=1.0))
        self.at("vectors")
        self.play(LaggedStart(*[FadeIn(c, shift=0.2 * DOWN) for c in cols], lag_ratio=0.12, run_time=0.9))
        self.at("token")
        self.play(Circumscribe(toks, color=YELLOW, buff=0.12), run_time=0.7)
        self.at("hidden")
        self.play(Circumscribe(cols, color=YELLOW, buff=0.12), run_time=0.7)
        mat = token("mat", color=YELLOW).move_to([4.6, cols.get_y(), 0])
        pred = Arrow(cols[-1].get_right(), mat.get_left(), buff=0.2, color=GREY_B, stroke_width=4)
        self.at("prediction")
        self.play(GrowArrow(pred), FadeIn(mat, shift=0.2 * RIGHT), run_time=0.6)
        q = Text("vector?", font_size=72, color=YELLOW).move_to([0, -2.4, 0])
        self.at("vector")
        self.play(FadeIn(q, scale=0.8), run_time=0.5)
        self.end_section()

    # 2. A list of numbers --------------------------------------------------------------------
    def s2_list(self):
        self.section(2)
        self.clear(run_time=0.5)
        head = Text("a vector is a list of numbers", font_size=34, color=GREY_B).move_to([0, 2.6, 0])
        v2 = Text("[2, 1]", font_size=60).move_to([-4.0, 0.5, 0])
        v4 = Text("[0.3, −1.2, 0.8, 2.0]", font_size=48).move_to([2.6, 0.5, 0])
        note = caption("illustrative")
        self.at("list")
        self.play(FadeIn(head, shift=0.2 * DOWN), FadeIn(v2, scale=0.8), run_time=0.7)
        self.at("four")
        self.play(FadeIn(v4, scale=0.9), FadeIn(note), run_time=0.7)
        d2 = Text("dimension 2", font_size=32, t2c={"2": YELLOW}).next_to(v2, DOWN, buff=0.6)
        d4 = Text("dimension 4", font_size=32, t2c={"4": YELLOW}).next_to(v4, DOWN, buff=0.6).match_y(d2)
        self.at("dimension")
        self.play(FadeIn(d2, shift=0.2 * UP), FadeIn(d4, shift=0.2 * UP), run_time=0.6)
        self.v2 = v2
        self.end_section()

    # 3. Two numbers -> an arrow ----------------------------------------------------------------
    def s3_arrow(self):
        self.section(3)
        v2 = self.v2
        self.play(*self.clear_anims(keep=[v2]), v2.animate.move_to([PANEL_X, 1.4, 0]), run_time=0.6)
        plane, ticks = make_plane()
        self.at("draw")
        self.play(Create(plane), FadeIn(ticks), run_time=1.0)
        right = DashedLine(P(0, 0), P(2, 0), color=YELLOW, stroke_width=5, dash_length=0.12)
        up = DashedLine(P(2, 0), P(2, 1), color=YELLOW, stroke_width=5, dash_length=0.12)
        right_note = Text("2 → right", font_size=30, t2c={"2": YELLOW}).move_to([PANEL_X, 0.2, 0])
        up_note = Text("1 → up", font_size=30, t2c={"1": YELLOW}).next_to(right_note, DOWN, buff=0.35)
        up_note.align_to(right_note, LEFT)
        self.at("right")
        self.play(Create(right), v2[1].animate.set_color(YELLOW), FadeIn(right_note, shift=0.1 * UP), run_time=0.6)
        self.at("up")
        self.play(Create(up), v2[3].animate.set_color(YELLOW), FadeIn(up_note, shift=0.1 * UP), run_time=0.5)
        arrow = vec_arrow(2, 1)
        label = Text("[2, 1]", font_size=30, color=BLUE).next_to(P(2, 1), UR, buff=0.1)
        self.at("arrow")
        self.play(GrowArrow(arrow), FadeIn(label), run_time=0.7)
        dot0 = Dot(P(0, 0), radius=0.07, color=WHITE)
        self.at("origin")
        self.play(FadeIn(dot0, scale=0.5), run_time=0.3)
        self.end_section()

    # 4. More dimensions ------------------------------------------------------------------------
    def s4_dimensions(self):
        self.section(4)
        o = np.array([-1.1, 0.4, 0])
        ex, ey, ez = np.array([1.0, 0, 0]), np.array([-0.5, -0.4, 0]), np.array([0, 0.85, 0])

        def p3(x, y, z):
            return o + x * ex + y * ey + z * ez

        ax_style = dict(color=GREY_B, stroke_width=3, buff=0, max_tip_length_to_length_ratio=0.08)
        axes3 = VGroup(Arrow(p3(0, 0, 0), p3(3.3, 0, 0), **ax_style), Arrow(p3(0, 0, 0), p3(0, 2.4, 0), **ax_style),
                       Arrow(p3(0, 0, 0), p3(0, 0, 3.1), **ax_style))
        ax_labels = VGroup(Text("x", font_size=26, color=GREY_B).next_to(p3(3.3, 0, 0), RIGHT, buff=0.12),
                           Text("y", font_size=26, color=GREY_B).next_to(p3(0, 2.4, 0), LEFT, buff=0.12),
                           Text("z", font_size=26, color=GREY_B).next_to(p3(0, 0, 3.1), UP, buff=0.12))
        self.play(*self.clear_anims(), Create(axes3), FadeIn(ax_labels), run_time=0.9)
        guides = VGroup(DashedLine(p3(2, 0, 0), p3(2, 1, 0)), DashedLine(p3(0, 1, 0), p3(2, 1, 0)),
                        DashedLine(p3(2, 1, 0), p3(2, 1, 2))).set_stroke(GREY_B, 2)
        arrow3 = Arrow(p3(0, 0, 0), p3(2, 1, 2), buff=0, color=BLUE, stroke_width=6,
                       max_tip_length_to_length_ratio=0.15)
        lab3 = Text("[2, 1, 2]", font_size=30, color=BLUE).next_to(p3(2, 1, 2), RIGHT, buff=0.15)
        self.at("arrow")
        self.play(Create(guides), GrowArrow(arrow3), FadeIn(lab3), run_time=0.8)

        rng = np.random.default_rng(7)
        vals = rng.uniform(-0.99, 0.99, 11)
        cells = VGroup()
        for i, v in enumerate(vals):
            op = 1.0 if i < 6 else max(0.12, 1 - 0.18 * (i - 5))
            box = Rectangle(width=0.95, height=0.6, stroke_color=GREY_B, stroke_width=2, stroke_opacity=op,
                            fill_color=value_color(v), fill_opacity=0.35 * op)
            txt = Text(f"{v:.2f}".replace("-", "−"), font=MONO, font_size=20).set_opacity(op)
            cells.add(VGroup(box, txt.move_to(box)))
        cells.arrange(RIGHT, buff=0.06).move_to([-6.55, -1.45, 0], aligned_edge=LEFT)
        dots = Text("…", font_size=40, color=GREY_B).next_to(cells, RIGHT, buff=0.3)
        note = caption("illustrative values")
        self.at("cant")
        self.play(LaggedStart(*[FadeIn(c, shift=0.15 * RIGHT) for c in cells], lag_ratio=0.1),
                  FadeIn(dots), FadeIn(note), run_time=1.1)
        gpt = Text("GPT-2: 768 numbers per vector", font_size=32, t2c={"768": YELLOW}).move_to([0, -2.6, 0])
        self.at("768")
        self.play(FadeIn(gpt, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # 5. Adding vectors ---------------------------------------------------------------------------
    def s5_add(self):
        self.section(5)
        self.clear(run_time=0.4)
        plane, ticks = make_plane()
        self.play(Create(plane), FadeIn(ticks), run_time=0.8)
        a = vec_arrow(2, 1, BLUE)
        b = vec_arrow(1, 2, TEAL)
        a_lab = Text("a", font_size=32, color=BLUE).move_to(P(1.4, 0.7) + 0.4 * np.array([1, -2, 0]) / np.sqrt(5))
        b_lab = Text("b", font_size=32, color=TEAL).next_to(P(1, 2), UP, buff=0.12)
        self.at("vectors")
        self.at("add")  # the second "add": "add their numbers"
        self.play(GrowArrow(a), GrowArrow(b), FadeIn(a_lab), FadeIn(b_lab), run_time=0.7)
        eq1 = Text("[2, 1] + [1, 2]", font_size=36, t2c={"[2, 1]": BLUE, "[1, 2]": TEAL}).move_to([PANEL_X, 1.0, 0])
        eq2 = Text("= [3, 3]", font_size=36, t2c={"[3, 3]": YELLOW}).next_to(eq1, DOWN, buff=0.4)
        eq2.align_to(eq1, LEFT)
        self.at("position")
        self.play(FadeIn(eq1, shift=0.2 * UP), run_time=0.6)
        self.at("three")
        self.play(FadeIn(eq2, shift=0.2 * UP), run_time=0.5)
        self.at("placing")
        self.play(VGroup(b, b_lab).animate.shift(P(2, 1) - P(0, 0)), run_time=0.8)
        s = vec_arrow(3, 3, YELLOW)
        s.set_z_index(-1)
        s_lab = Text("a + b", font_size=30, color=YELLOW).next_to(P(1.5, 1.5), UL, buff=0.1)
        self.at("tip")
        self.play(GrowArrow(s), FadeIn(s_lab), run_time=0.6)
        self.stage5 = dict(plane=plane, ticks=ticks, a=a, a_lab=a_lab)
        self.end_section()

    # 6. Scaling ----------------------------------------------------------------------------------
    def s6_scale(self):
        self.section(6)
        st = self.stage5
        plane, ticks, a, a_lab = st["plane"], st["ticks"], st["a"], st["a_lab"]
        self.at("single")
        self.play(*self.clear_anims(keep=[plane, ticks, a, a_lab]), run_time=0.5)
        ghost = DashedLine(P(0, 0), P(2, 1), color=GREY_B, stroke_width=3, dash_length=0.1)
        eq1 = Text("2 × [2, 1] = [4, 2]", font_size=32, t2c={"2 ×": YELLOW, "[4, 2]": BLUE})
        eq1.move_to([1.3, 1.0, 0], aligned_edge=LEFT)
        lab2 = Text("2a", font_size=32, color=BLUE).next_to(P(4, 2), UP, buff=0.12)
        self.at("scalar")
        self.add(ghost)
        self.bring_to_front(a)
        self.play(Transform(a, vec_arrow(4, 2)), FadeOut(a_lab), FadeIn(lab2), FadeIn(eq1, shift=0.2 * UP),
                  run_time=1.0)
        eq2 = Text("−1 × [2, 1] = [−2, −1]", font_size=32, t2c={"−1 ×": YELLOW, "[−2, −1]": BLUE})
        eq2.next_to(eq1, DOWN, buff=0.45).align_to(eq1, LEFT)
        lab_neg = Text("−a", font_size=32, color=BLUE).next_to(P(-2, -1), LEFT, buff=0.12)
        self.at("negative")
        self.play(Transform(a, vec_arrow(-2, -1)), FadeOut(lab2), FadeIn(lab_neg), eq1.animate.set_opacity(0.45),
                  FadeIn(eq2, shift=0.2 * UP), run_time=0.9)
        self.end_section()

    # 7. Length -----------------------------------------------------------------------------------
    def s7_length(self):
        self.section(7)
        st = self.stage5
        plane, ticks = st["plane"], st["ticks"]
        self.play(*self.clear_anims(keep=[plane, ticks]), run_time=0.5)
        v = vec_arrow(3, 4)
        leg_x = DashedLine(P(0, 0), P(3, 0), color=YELLOW, stroke_width=5, dash_length=0.12)
        leg_y = DashedLine(P(3, 0), P(3, 4), color=YELLOW, stroke_width=5, dash_length=0.12)
        corner = VMobject().set_points_as_corners([P(3, 0) + 0.25 * LEFT, P(3, 0) + 0.25 * LEFT + 0.25 * UP,
                                                   P(3, 0) + 0.25 * UP]).set_stroke(YELLOW, 2)
        lx = Text("3", font_size=32, color=YELLOW).move_to(P(1.6, 0) + 0.3 * UP)
        ly = Text("4", font_size=32, color=YELLOW).next_to(P(3, 2), RIGHT, buff=0.15)
        self.at("pythagoras")
        self.play(GrowArrow(v), Create(leg_x), Create(leg_y), Create(corner), FadeIn(lx), FadeIn(ly), run_time=0.9)
        eq1 = Text("√(3² + 4²) = √25", font_size=36).move_to([PANEL_X, 1.0, 0])
        eq2 = Text("= 5", font_size=36, t2c={"5": YELLOW})
        eq2.next_to(eq1, DOWN, buff=0.4)
        eq2.shift((eq1[8].get_x() - eq2[0].get_x()) * RIGHT)
        self.at("square")
        self.play(FadeIn(eq1, shift=0.2 * UP), run_time=0.6)
        length = Text("length 5", font_size=30, color=BLUE)
        y_lab = 3.3
        length.move_to(P(0, y_lab)).set_x(P(y_lab * 3 / 4, y_lab)[0] - 0.35 - length.width / 2)
        self.at("5")
        self.play(FadeIn(eq2, shift=0.2 * UP), FadeIn(length), run_time=0.5)
        self.end_section()

    # 8. Directions of meaning --------------------------------------------------------------------
    def s8_meaning(self):
        self.section(8)
        self.clear(run_time=0.5)
        o8 = np.array([-0.8, 0.1, 0])
        axes = VGroup(Arrow([-6.3, o8[1], 0], [6.3, o8[1], 0], buff=0, color=GREY_D, stroke_width=2,
                            max_tip_length_to_length_ratio=0.02),
                      Arrow([o8[0], -3.4, 0], [o8[0], 3.4, 0], buff=0, color=GREY_D, stroke_width=2,
                            max_tip_length_to_length_ratio=0.035))
        shift = np.array([3.9, 0.3, 0])
        sing = {"cat": np.array([-2.9, 1.1, 0]), "dog": np.array([-2.1, 1.9, 0]), "mat": np.array([-2.6, -1.7, 0])}
        pts = {**sing, **{w + "s": p + shift for w, p in sing.items()}}
        dots = {w: Dot(p, radius=0.09, color=BLUE_C) for w, p in pts.items()}
        labels = {w: Text(w, font_size=26).next_to(dots[w], UP, buff=0.12) for w in pts}
        points = VGroup(*[VGroup(dots[w], labels[w]) for w in pts])
        self.at("directions")
        self.play(Create(axes), LaggedStart(*[FadeIn(p, scale=0.7) for p in points], lag_ratio=0.1), run_time=0.9)
        animal = Arrow([-4.5, -1.9, 0], [-4.5, 2.1, 0], buff=0, color=YELLOW, stroke_width=6,
                       max_tip_length_to_length_ratio=0.08)
        animal_lab = Text("animal-ness", font_size=26, color=YELLOW).next_to(animal, UP, buff=0.12)
        self.at("animals")
        self.play(GrowArrow(animal), FadeIn(animal_lab), run_time=0.7)
        plural = VGroup(*[Arrow(pts[w], pts[w + "s"], buff=0.15, color=TEAL, stroke_width=4,
                                max_tip_length_to_length_ratio=0.07) for w in sing])
        plural_lab = Text("plural-ness", font_size=26, color=TEAL).next_to(plural[2], DOWN, buff=0.2)
        self.at("plurals")
        self.play(LaggedStart(*[GrowArrow(p) for p in plural], lag_ratio=0.15), FadeIn(plural_lab), run_time=0.8)
        tok_vec = Arrow(o8, pts["cats"], buff=0.08, color=BLUE, stroke_width=5, max_tip_length_to_length_ratio=0.15)
        self.at("vector")
        self.play(GrowArrow(tok_vec), run_time=0.5)
        self.at("point")
        self.play(Flash(dots["cats"], color=YELLOW, line_length=0.18, flash_radius=0.22), run_time=0.5)
        self.at("meaning")
        self.play(FadeIn(caption("illustrative")), run_time=0.4)
        self.end_section()

    # 9. NumPy ------------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, font_size=28)
        code.move_to(ORIGIN)
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[1])
        self.clear(run_time=0.4)
        self.play(FadeIn(code, shift=0.3 * UP), run_time=0.5)
        self.at("array")
        self.play(Create(hl), run_time=0.3)
        self.at("adding")
        self.play(highlight(hl, code, 3), run_time=0.25)
        self.at("scaling")
        self.play(highlight(hl, code, 4), run_time=0.25)
        self.at("length")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.end_section()

    # 10. Outro -----------------------------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        self.clear(run_time=0.5)
        card = used_in_card(USED_IN)
        rows = card[1]
        self.at("every")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.7)
        box = SurroundingRectangle(rows[1], color=YELLOW, buff=0.12)
        self.at("three")
        self.play(Create(box), run_time=0.4)
        for word, i in [("four", 2), ("nine", 3)]:
            self.at(word)
            self.play(Transform(box, SurroundingRectangle(rows[i], color=YELLOW, buff=0.12)), run_time=0.4)
        nxt = next_up_card(NEXT)
        self.at("product")
        self.play(*self.clear_anims(), FadeIn(nxt, shift=0.2 * UP), run_time=0.6)
        self.end_section()
