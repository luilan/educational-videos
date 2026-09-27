"""Foundations F13 (extra) — PyTorch and Autograd.

Render from the repo root:  ./render.sh foundations f13
"""
import random

import numpy as np
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card, used_in_card
from f13_script import LABEL, NEXT, TAGLINE, TITLE, USED_IN
from intro import play_token_intro
from voiced_scene import VoicedScene

VAR = BLUE_D        # tensors / variables
OP = PURPLE_B       # operation nodes
FWD = TEAL          # forward values
BWD = RED           # backward pass / gradients
MONOKAI = "#272822"

# ---------------------------------------------------------------------------- verified numbers
X0 = 3
SQ, DBL = X0 ** 2, 2 * X0
Y0 = SQ + DBL
assert (SQ, DBL, Y0) == (9, 6, 15)
D_SQ, D_DBL = 2 * X0, 2                       # local slopes d(x²)/dx = 2x, d(2x)/dx = 2
GRAD = D_SQ + D_DBL
assert (D_SQ, D_DBL, GRAD) == (6, 2, 8)
assert 2 * X0 + 2 == GRAD == 8                 # d/dx (x² + 2x) = 2x + 2
_f = lambda x: x ** 2 + 2 * x                  # noqa: E731
assert abs((_f(X0 + 1e-6) - _f(X0 - 1e-6)) / 2e-6 - GRAD) < 1e-6
assert GRAD + GRAD == 16                       # backward twice without zero_grad

# section 2: an illustrative 3 x 4 tensor
T2 = [[0.2, -1.3, 0.7, 2.1],
      [1.5, 0.4, -0.6, 0.9],
      [-0.8, 1.1, 0.3, -1.7]]

# section 7: illustrative gradients for seven weights
G7 = [0.12, -0.40, 0.03, -0.21, 0.35, -0.07, 0.18]

# section 8: four weights, their gradients, and one Adam step (lr 0.01)
W8 = np.array([0.50, -1.20, 0.80, 0.30])
G8 = np.array([0.12, -0.40, 0.25, -0.08])
_lr, _b1, _b2, _eps = 0.01, 0.9, 0.999, 1e-8
_m, _v = (1 - _b1) * G8, (1 - _b2) * G8 ** 2
W8_NEW = W8 - _lr * (_m / (1 - _b1)) / (np.sqrt(_v / (1 - _b2)) + _eps)
assert np.allclose(np.round(W8_NEW, 2), [0.49, -1.19, 0.79, 0.31])

CODE = """import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x          # PyTorch records the graph
y.backward()                # chain rule, in reverse
x.grad                      # tensor(8.)"""


# ---------------------------------------------------------------------------- helpers
def fmt(v):
    """Numbers with two decimals (one for the tensor grid) and a unicode minus."""
    return f"{v:.2f}".replace("-", "−")


def label(text, color=WHITE, font_size=30, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def mono(text, size=30, color=WHITE):
    """Monospace text rendered at 48 and scaled, so the glyph spacing stays even at every size."""
    return Text(text, font=MONO, font_size=48, color=color).scale(size / 48)


def caption(text):
    return Text(text, font_size=20, color=GREY_B).to_corner(DR, buff=0.3)


def glyphs(mob, s, start, end):
    """Glyphs of Text `mob` (built from string s) for s[start:end]; spaces have no glyphs."""
    a = len(s[:start].replace(" ", ""))
    return mob[a:a + len(s[start:end].replace(" ", ""))]


def sub(mob, s, part, nth=0):
    i = -1
    for _ in range(nth + 1):
        i = s.find(part, i + 1)
    return glyphs(mob, s, i, i + len(part))


def cell(text, color=BLUE, w=0.95, h=0.55, size=24, opacity=0.2, text_color=WHITE):
    box = Rectangle(width=w, height=h, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=opacity)
    return VGroup(box, mono(text, size, text_color).move_to(box))


def var_node(text, color=VAR, size=32):
    t = mono(text, size)
    box = RoundedRectangle(corner_radius=0.15, width=t.width + 0.5, height=0.8, stroke_color=color,
                           fill_color=color, fill_opacity=0.25)
    return VGroup(box, t.move_to(box))


def op_node(text, color=OP, r=0.55, size=36):
    c = Circle(radius=r, stroke_color=color, fill_color=color, fill_opacity=0.2)
    return VGroup(c, label(text, WHITE, size).move_to(c))


def exit_point(node, p0, d):
    """Where the ray p0 + t·d (t > 0, p0 inside the node's shape) leaves the shape."""
    shape = node[0]
    c = shape.get_center()
    q = p0 - c
    if isinstance(shape, Circle):
        r = shape.width / 2
        b = np.dot(d, q)
        t = -b + np.sqrt(b * b - (np.dot(q, q) - r * r))
    else:
        hw, hh = shape.width / 2, shape.height / 2
        ts = []
        if abs(d[0]) > 1e-9:
            ts.append((hw - q[0] * np.sign(d[0])) / abs(d[0]))
        if abs(d[1]) > 1e-9:
            ts.append((hh - q[1] * np.sign(d[1])) / abs(d[1]))
        t = min(ts)
    return p0 + t * d


def link_pts(a, b, side=0, off=0.0, gap=0.06):
    """End points of the edge a → b, optionally shifted sideways (side=±1: left/right normal) by off."""
    ca, cb = a[0].get_center(), b[0].get_center()
    u = (cb - ca) / np.linalg.norm(cb - ca)
    n = np.array([-u[1], u[0], 0]) * side * off
    return exit_point(a, ca + n, u) + gap * u, exit_point(b, cb + n, -u) - gap * u


def arrow(p, q, color=GREY_B, sw=4):
    return Arrow(p, q, buff=0, color=color, stroke_width=sw, max_tip_length_to_length_ratio=0.12,
                 max_stroke_width_to_length_ratio=10)


def link(a, b, color=GREY_B):
    return arrow(*link_pts(a, b), color=color)


def build_graph():
    """The forward graph of y = x² + 2x at x = 3."""
    g = {}
    g["x"] = var_node("x = 3").move_to([-5.1, 0, 0])
    g["sq"] = op_node("x²").move_to([-1.4, 1.55, 0])
    g["dbl"] = op_node("× 2", size=32).move_to([-1.4, -1.55, 0])
    g["add"] = op_node("+", r=0.5, size=44).move_to([1.9, 0, 0])
    g["y"] = var_node("y = 15").move_to([5.1, 0, 0])
    g["e_xsq"] = link(g["x"], g["sq"])
    g["e_xdbl"] = link(g["x"], g["dbl"])
    g["e_sqadd"] = link(g["sq"], g["add"])
    g["e_dbladd"] = link(g["dbl"], g["add"])
    g["e_addy"] = link(g["add"], g["y"])

    def inside(a, b, side, text):
        s, e = link_pts(a, b)
        u = (e - s) / np.linalg.norm(e - s)
        n = np.array([-u[1], u[0], 0]) * side
        return label(text, FWD, 32).move_to((s + e) / 2 + 0.36 * n)

    g["v9"] = inside(g["sq"], g["add"], -1, str(SQ))
    g["v6"] = inside(g["dbl"], g["add"], 1, str(DBL))
    return g


def rec_badge():
    dot = Dot(radius=0.1, color=RED)
    row = VGroup(dot, label("recording", RED, 26)).arrange(RIGHT, buff=0.15)
    box = RoundedRectangle(corner_radius=0.15, width=row.width + 0.4, height=row.height + 0.3, stroke_color=RED,
                           stroke_width=2, fill_opacity=0)
    badge = VGroup(box, row.move_to(box))
    clock = {"t": 0.0}

    def blink(m, dt):
        clock["t"] += dt
        m.set_opacity(1 if clock["t"] % 1.0 < 0.6 else 0.2)

    return badge, dot, blink


def history_tag():
    pts = [[-0.95, 0, 0], [-0.62, 0.34, 0], [0.95, 0.34, 0], [0.95, -0.34, 0], [-0.62, -0.34, 0]]
    shape = Polygon(*pts, stroke_color=YELLOW, stroke_width=3, fill_color=YELLOW, fill_opacity=0.12)
    hole = Circle(radius=0.07, stroke_color=YELLOW, stroke_width=2).move_to([-0.66, 0, 0])
    t = label("history", WHITE, 26).move_to([0.17, 0, 0])
    return VGroup(shape, hole, t)


def tape_icon():
    win = RoundedRectangle(corner_radius=0.12, width=2.2, height=2.2, stroke_color=GREY_B, stroke_width=2,
                           fill_color="#1c1c1c", fill_opacity=1)
    title = label("tape", GREY_B, 24).move_to(win.get_top() + 0.32 * DOWN)
    rule = Line(win.get_left() + 0.2 * RIGHT, win.get_right() + 0.2 * LEFT, color=GREY_D, stroke_width=2)
    rule.set_y(title.get_bottom()[1] - 0.12)
    rng = random.Random(5)
    entries = VGroup()
    for k in range(4):
        y = rule.get_y() - 0.32 - 0.33 * k
        x0 = win.get_left()[0] + 0.3
        dot = Dot([x0, y, 0], radius=0.05, color=RED)
        w = rng.uniform(0.8, 1.35)
        entries.add(VGroup(dot, Line([x0 + 0.18, y, 0], [x0 + 0.18 + w, y, 0], color=GREY_B, stroke_width=5)))
    return VGroup(win, title, rule), entries


class AutogradVideo(VoicedScene):
    VIDEO = "f13"

    # ------------------------------------------------------------------ stage helpers
    def clear_anims(self, keep=()):
        anims = []
        for m in list(self.mobjects):
            if any(m is k for k in keep):
                continue
            if isinstance(m, ValueTracker):
                self.remove(m)
                continue
            m.clear_updaters()
            anims.append(FadeOut(m))
        return anims

    def construct(self):
        play_token_intro(self, TITLE, 0, TAGLINE, label=LABEL)
        self.s1_one_line()
        self.s2_tensors()
        self.s3_recording()
        self.s4_graph()
        self.s5_backward()
        self.s6_check()
        self.s7_bigger()
        self.s8_optimizer()
        self.s9_code()
        self.s10_outro()
        self.end_section()
        finish(self)

    # 1. One line does all the calculus -----------------------------------------------------------
    def s1_one_line(self):
        self.section(1)
        src = "loss.backward()"
        line = mono(src, 60)
        sub(line, src, "loss").set_color("#F8F8F2")
        sub(line, src, "backward").set_color("#A6E22E")
        panel = RoundedRectangle(corner_radius=0.2, width=line.width + 1.2, height=line.height + 1.3,
                                 stroke_color=GREY_D, stroke_width=2, fill_color=MONOKAI, fill_opacity=1)
        dots = VGroup(*[Dot(radius=0.06, color=c) for c in (RED, YELLOW, GREEN)]).arrange(RIGHT, buff=0.1)
        dots.move_to(panel.get_corner(UL) + np.array([0.4, -0.25, 0]))
        line.move_to(panel.get_center() + 0.1 * DOWN)
        editor = VGroup(panel, dots, line).move_to([0, 0.35, 0])
        ep = label("Episode 11 · training", GREY_B, 28).move_to([0, 2.55, 0])
        self.at("11")
        self.play(FadeIn(ep, shift=0.15 * DOWN), run_time=0.5)
        self.at("line")
        self.play(FadeIn(editor, shift=0.2 * UP), run_time=0.6)
        cap = label("all the calculus, in one line", WHITE, 36).move_to([0, -1.75, 0])
        self.at("calculus")
        self.play(FadeIn(cap, shift=0.15 * UP), run_time=0.5)

        bw = sub(line, src, "backward()")
        glow = VGroup(*[bw.copy().set_fill(opacity=0).set_stroke(YELLOW, width=w, opacity=o)
                        for w, o in [(5, 0.45), (10, 0.25), (18, 0.12), (28, 0.06)]])
        glow.set_z_index(-1)
        panel.set_z_index(-2)
        self.at("backward")
        self.play(bw.animate.set_color(YELLOW), FadeIn(glow), panel.animate.set_stroke(YELLOW, width=3),
                  run_time=0.6)
        self.play(glow.animate(rate_func=there_and_back).scale(1.06), run_time=0.8)
        self.end_section()

    # 2. Tensors ----------------------------------------------------------------------------------
    def s2_tensors(self):
        self.section(2)
        head = label("PyTorch", WHITE, 44).move_to([0, 3.2, 0])
        self.play(*self.clear_anims(), FadeIn(head, shift=0.2 * DOWN), run_time=0.5)

        grid = VGroup()
        for i in range(3):
            for j in range(4):
                grid.add(cell(f"{T2[i][j]:.1f}".replace("-", "−"), BLUE, 1.15, 0.66, 30)
                         .move_to([(j - 1.5) * 1.22, (1 - i) * 0.76, 0]))
        grid.move_to([-1.8, 0.2, 0])
        tname = label("tensor", BLUE_B, 36).next_to(grid, UP, buff=0.35)
        note = caption("illustrative values")
        self.at("tensors")
        self.play(FadeIn(grid, lag_ratio=0.05), FadeIn(tname), FadeIn(note), run_time=0.7)
        self.grid, self.tname = grid, tname

        numpy = label("≈ NumPy array", GREY_B, 32).next_to(grid, DOWN, buff=0.45)
        self.at("numpy")
        self.play(FadeIn(numpy, shift=0.15 * UP), run_time=0.5)

        tag = history_tag().move_to([3.4, 1.55, 0])
        corner = grid.get_corner(UR)
        string = CubicBezier(corner, corner + np.array([0.7, 0.5, 0]), tag[1].get_center() + np.array([-0.8, 0, 0]),
                             tag[1].get_center(), stroke_color=YELLOW, stroke_width=2)
        self.at("remember")
        self.play(Create(string), FadeIn(tag, shift=0.2 * LEFT), run_time=0.6)
        self.s2_extra = VGroup(numpy, tag, string)
        self.end_section()

    # 3. requires_grad=True starts the recording ---------------------------------------------------
    def s3_recording(self):
        self.section(3)
        self.play(FadeOut(self.s2_extra), run_time=0.4)
        grid = self.grid
        rg = mono("requires_grad=True", 30, YELLOW)
        rg_box = RoundedRectangle(corner_radius=0.12, width=rg.width + 0.4, height=rg.height + 0.35,
                                  stroke_color=YELLOW, stroke_width=2, fill_color=YELLOW, fill_opacity=0.1)
        rg_tag = VGroup(rg_box, rg.move_to(rg_box)).next_to(grid, DOWN, buff=0.45)
        self.at("requires")
        self.play(FadeIn(rg_tag, shift=0.15 * UP), run_time=0.5)

        badge, dot, blink = rec_badge()
        badge.to_corner(UR, buff=0.4)
        self.at("recording")
        self.play(FadeIn(badge, scale=1.2), run_time=0.4)
        dot.add_updater(blink)
        self.badge = badge

        frame, entries = tape_icon()
        tape = VGroup(frame, entries).move_to([3.7, 0.25, 0])
        feed = Arrow(grid.get_right() + 0.1 * RIGHT, frame.get_left() + 0.1 * LEFT, buff=0.05, color=GREY_B,
                     stroke_width=3, max_tip_length_to_length_ratio=0.15)
        self.at("operation")
        self.play(GrowArrow(feed), FadeIn(frame), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(e, shift=0.1 * RIGHT) for e in entries], lag_ratio=0.35), run_time=0.8)
        self.end_section()

    # 4. The forward pass builds a graph ------------------------------------------------------------
    def s4_graph(self):
        self.section(4)
        self.play(*self.clear_anims(keep=[self.badge]), run_time=0.4)
        g = build_graph()
        self.g = g
        self.at("3")
        self.play(FadeIn(g["x"], shift=0.2 * RIGHT), run_time=0.5)

        src = "y = x² + 2x"
        formula = label(src, WHITE, 40).move_to([0, 3.2, 0])
        self.formula = formula
        self.at("compute")
        self.play(FadeIn(formula, shift=0.15 * DOWN), run_time=0.5)
        self.at("builds")
        self.play(Indicate(self.badge[0], color=RED, scale_factor=1.1), run_time=0.6)
        self.at("goes")
        self.play(Indicate(g["x"], scale_factor=1.1), run_time=0.5)

        self.at("square")
        self.play(GrowArrow(g["e_xsq"]), GrowFromCenter(g["sq"]), run_time=0.6)
        self.play(FadeIn(g["v9"], shift=0.1 * RIGHT), run_time=0.3)
        self.at("times")
        self.play(GrowArrow(g["e_xdbl"]), GrowFromCenter(g["dbl"]), run_time=0.6)
        self.play(FadeIn(g["v6"], shift=0.1 * RIGHT), run_time=0.3)
        self.at("added")
        self.play(GrowArrow(g["e_sqadd"]), GrowArrow(g["e_dbladd"]), GrowFromCenter(g["add"]), run_time=0.7)
        self.at("15")
        self.play(GrowArrow(g["e_addy"]), FadeIn(g["y"], shift=0.2 * RIGHT), run_time=0.6)
        self.end_section()

    # 5. backward() walks the graph in reverse ------------------------------------------------------
    def s5_backward(self):
        self.section(5)
        g = self.g
        ycode = mono("y.backward()", 28, YELLOW).next_to(g["y"], DOWN, buff=0.45)
        self.badge.clear_updaters()
        self.play(FadeOut(self.badge), FadeIn(ycode, shift=0.15 * UP), run_time=0.5)

        revs = []
        for a, b, side in [("add", "y", 1), ("sq", "add", 1), ("dbl", "add", -1), ("x", "sq", 1), ("x", "dbl", -1)]:
            p, q = link_pts(g[a], g[b], side=side, off=0.3)
            revs.append(arrow(q, p, BWD, 5))
        self.at("backward")
        self.play(LaggedStart(GrowArrow(revs[0]), AnimationGroup(GrowArrow(revs[1]), GrowArrow(revs[2])),
                              lag_ratio=0.7), run_time=1.2)
        self.at("walks")
        self.play(GrowArrow(revs[3]), GrowArrow(revs[4]), run_time=0.7)

        s_up, s_dn = "d(x²)/dx = 2x = 6", "d(2x)/dx = 2"
        up = label(s_up, RED_B, 30).move_to([-3.75, 2.3, 0])
        dn = label(s_dn, RED_B, 30).move_to([-3.75, -2.3, 0])
        self.at("reverse")
        self.play(FadeIn(up, shift=0.1 * DOWN), FadeIn(dn, shift=0.1 * UP), run_time=0.6)

        chain = label("chain rule at every step", WHITE, 30).move_to([3.65, -2.3, 0])
        self.at("chain")
        self.play(FadeIn(chain, shift=0.15 * UP),
                  *[a.animate(rate_func=there_and_back).set_stroke(width=9) for a in revs], run_time=0.7)

        self.at("lands")
        self.play(g["x"][0].animate.set_stroke(BWD, width=5), run_time=0.4)

        s_sum = "x.grad = 6 + 2 = 8"
        total = mono(s_sum, 34).move_to([-6.2, -3.2, 0], aligned_edge=LEFT)
        six, two = sub(total, s_sum, "6"), sub(total, s_sum, "2")
        eight = sub(total, s_sum, "8")
        six.set_color(RED_B)
        two.set_color(RED_B)
        eight.set_color(YELLOW)
        rest = VGroup(sub(total, s_sum, "x.grad"), sub(total, s_sum, "="), sub(total, s_sum, "+"),
                      sub(total, s_sum, "=", 1), eight)
        self.at("grad")
        self.play(TransformFromCopy(up[-1], six), TransformFromCopy(dn[-1], two), FadeIn(rest), run_time=0.8)
        self.remove(six, two, rest)
        self.add(total)
        self.total = total
        self.end_section()

    # 6. Check against the derivative ---------------------------------------------------------------
    def s6_check(self):
        self.section(6)
        total = self.total
        self.play(*self.clear_anims(keep=[total]), total.animate.move_to([-6.3, -1.25, 0], aligned_edge=LEFT),
                  run_time=0.4)

        s1 = "d/dx (x² + 2x) = 2x + 2"
        l1 = label(s1, WHITE, 36).move_to([-6.3, 1.75, 0], aligned_edge=LEFT)
        lhs = glyphs(l1, s1, 0, s1.index("="))
        rhs = glyphs(l1, s1, s1.index("="), len(s1))
        sub(l1, s1, "2x + 2").set_color(YELLOW)

        ax = Axes(x_range=[-4, 4, 1], y_range=[-3, 24, 3], x_length=4.8, y_length=5.0, tips=False,
                  axis_config={"color": GREY_B, "stroke_width": 2, "tick_size": 0.05})
        ax.move_to([3.75, 0.1, 0])
        curve = ax.plot(_f, x_range=[-4, 3.95], color=BLUE_C, stroke_width=5)
        cname = label("y = x² + 2x", BLUE_B, 26).move_to(ax.c2p(-3.9, 15), aligned_edge=LEFT)
        self.at("slope")
        self.play(FadeIn(lhs, shift=0.15 * UP), Create(ax), Create(curve), FadeIn(cname), run_time=0.8)
        self.at("is")
        self.play(FadeIn(rhs, shift=0.15 * UP), run_time=0.4)

        P = ax.c2p(3, 15)
        dot = Dot(P, radius=0.09, color=YELLOW).set_z_index(3)
        guide = DashedLine(ax.c2p(3, 0), P, color=GREY_B, stroke_width=2)
        tick = label("3", GREY_B, 24).next_to(ax.c2p(3, 0), DOWN, buff=0.15)
        tan = Line(ax.c2p(1.7, 15 - 8 * 1.3), ax.c2p(4.0, 15 + 8 * 1.0), color=YELLOW, stroke_width=5)
        eq = rhs[0]
        at3 = label("at x = 3", GREY_B, 30)
        s2 = "= 2·3 + 2 = 8"
        l2 = label(s2, WHITE, 36)
        l2.move_to([0, 0.55, 0]).align_to(eq, LEFT)
        sub(l2, s2, "8").set_color(YELLOW)
        at3.next_to(l2, LEFT, buff=0.45)
        self.at("3")
        self.play(FadeIn(dot, scale=0.5), Create(guide), FadeIn(tick), Create(tan), FadeIn(at3), run_time=0.6)

        slope8 = label("slope 8", YELLOW, 28).next_to(ax.c2p(3.45, 19.4), LEFT, buff=0.2)
        self.at("8")
        self.play(FadeIn(l2, shift=0.15 * UP), FadeIn(slope8), run_time=0.5)

        ok = mono("x.grad = 8 ✓", 34, GREEN).move_to(total, aligned_edge=LEFT)
        ok_box = SurroundingRectangle(ok, color=GREEN, buff=0.18, corner_radius=0.1)
        self.at("grad")
        self.play(Indicate(total, color=YELLOW, scale_factor=1.05), run_time=0.6)
        self.at("8")
        self.play(FadeTransform(total, ok), Create(ok_box), run_time=0.5)
        self.end_section()

    # 7. A real model: same thing, bigger ----------------------------------------------------------
    def s7_bigger(self):
        self.section(7)
        self.play(*self.clear_anims(), run_time=0.4)

        mini = build_graph()
        mini_grp = VGroup(*[mini[k] for k in ("e_xsq", "e_xdbl", "e_sqadd", "e_dbladd", "e_addy",
                                               "x", "sq", "dbl", "add", "y", "v9", "v6")])
        mini_grp.scale(0.7).move_to([0, 0.4, 0])
        self.at("same")
        self.play(FadeIn(mini_grp), run_time=0.5)

        xs = [-5.4 + 1.4 * k for k in range(7)]
        counts = [3, 4, 5, 5, 4, 3, 2]
        yc, pitch = 1.0, 0.6
        cols = []
        for x, n in zip(xs, counts):
            cols.append(VGroup(*[Circle(radius=0.16, stroke_color=BLUE_C, stroke_width=3, fill_color=BLUE_C,
                                        fill_opacity=0.3).move_to([x, yc + (k - (n - 1) / 2) * pitch, 0])
                                 for k in range(n)]))
        rng = random.Random(13)
        col_edges = []
        for k in range(6):
            a, b = cols[k], cols[k + 1]
            pairs = set()
            for i in range(len(a)):
                for j in rng.sample(range(len(b)), 2):
                    pairs.add((i, j))
            for j in range(len(b)):
                if not any(p[1] == j for p in pairs):
                    pairs.add((rng.randrange(len(a)), j))
            col_edges.append(VGroup(*[Line(a[i].get_center(), b[j].get_center(), buff=0.16, color=GREY_B,
                                           stroke_width=2) for i, j in sorted(pairs)]))
        loss = VGroup(RoundedRectangle(corner_radius=0.15, width=1.4, height=0.75, stroke_color=ORANGE,
                                       fill_color=ORANGE, fill_opacity=0.25),
                      label("loss", WHITE, 28))
        loss[1].move_to(loss[0])
        loss.move_to([4.9, yc, 0])
        loss_edges = VGroup(*[Line(nd.get_center(), loss[0].get_left(), buff=0.16, color=GREY_B, stroke_width=2)
                              for nd in cols[-1]])
        col_edges.append(loss_edges)

        big = []
        for k in range(7):
            big.append(cols[k])
            if k < 6:
                big.append(col_edges[k])
        self.at("bigger")
        self.play(FadeOut(mini_grp, scale=0.6), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(m) if isinstance(m[0], Line) else GrowFromCenter(m) for m in big],
                              lag_ratio=0.25), run_time=0.9)

        top1 = label("millions of steps", GREY_B, 30)
        top2 = label("·  millions of weights", GREY_B, 30)
        top = VGroup(top1, top2).arrange(RIGHT, buff=0.25).move_to([0, 3.25, 0])
        self.at("millions")
        self.play(FadeIn(top1, shift=0.15 * DOWN), run_time=0.4)
        self.at("loss")
        self.play(FadeIn(loss, shift=0.2 * RIGHT), LaggedStart(*[Create(e) for e in loss_edges], lag_ratio=0.2),
                  run_time=0.6)

        wy, gy = -1.45, -2.3
        weights = VGroup(*[cell(f"w{'₁₂₃₄₅₆₇'[k]}", TEAL, 1.15, 0.55, 26).move_to([xs[k], wy, 0])
                           for k in range(7)])
        wires = VGroup(*[DashedLine(weights[k][0].get_top(), cols[k][0].get_center(), buff=0.16, color=GREY_D,
                                    stroke_width=2, dash_length=0.08) for k in range(7)])
        bins = VGroup(*[Rectangle(width=1.15, height=0.55, stroke_color=RED_E, stroke_width=2).move_to([xs[k], gy, 0])
                        for k in range(7)])
        gvals = VGroup(*[mono(fmt(G7[k]), 22).move_to(bins[k]) for k in range(7)])
        rlab = VGroup(label("weights", TEAL, 26).move_to([4.95, wy, 0]),
                      label("gradients", RED_B, 26).move_to([4.95, gy, 0]))
        self.at("weights")
        self.play(FadeIn(top2, shift=0.15 * DOWN), LaggedStart(*[FadeIn(w, shift=0.1 * UP) for w in weights],
                                                               lag_ratio=0.1),
                  FadeIn(wires), FadeIn(bins), FadeIn(rlab), run_time=0.7)

        sweep = VGroup(Line([0, -2.75, 0], [0, 2.65, 0], color=RED, stroke_width=14, stroke_opacity=0.25),
                       Line([0, -2.75, 0], [0, 2.65, 0], color=RED, stroke_width=4)).move_to([5.9, -0.05, 0])
        x_start, x_end, T = 5.9, -6.3, 1.8
        steps = []
        for k in range(6, -1, -1):
            t_k = (x_start - xs[k]) / (x_start - x_end) * T
            anims = [col_edges[k].animate.set_stroke(RED, width=3),
                     *[nd.animate.set_stroke(RED) for nd in cols[k]],
                     bins[k].animate.set_stroke(RED, width=3).set_fill(RED, 0.15),
                     FadeIn(gvals[k], scale=0.7)]
            steps.append(Succession(Wait(run_time=max(t_k - 0.1, 0.01)), AnimationGroup(*anims, run_time=0.35)))
        note = caption("illustrative values")
        self.at("backward")
        self.add(sweep)
        self.play(sweep.animate(rate_func=linear, run_time=T).move_to([x_end, -0.05, 0]), *steps,
                  FadeIn(note, run_time=0.5), loss[0].animate(run_time=0.3).set_stroke(RED))
        self.play(FadeOut(sweep), run_time=0.2)
        self.at("every")
        self.play(LaggedStart(*[b.animate(rate_func=there_and_back).set_fill(RED, 0.45) for b in bins],
                              lag_ratio=0.1), run_time=0.7)
        self.end_section()

    # 8. The optimizer and zero_grad ----------------------------------------------------------------
    def s8_optimizer(self):
        self.section(8)
        xs = [-3.9, -2.5, -1.1, 0.3]
        wy, gy = 2.15, 1.0
        wcells = VGroup(*[cell(fmt(W8[k]), TEAL, 1.2, 0.62, 26).move_to([xs[k], wy, 0]) for k in range(4)])
        gcells = VGroup(*[cell(fmt(G8[k]), RED, 1.2, 0.62, 26, opacity=0.15).move_to([xs[k], gy, 0])
                          for k in range(4)])
        rlab = VGroup(label("weights", TEAL, 26).move_to([-4.75, wy, 0], aligned_edge=RIGHT),
                      label("gradients", RED_B, 26).move_to([-4.75, gy, 0], aligned_edge=RIGHT))
        note = caption("illustrative values")
        self.play(*self.clear_anims(), FadeIn(wcells), FadeIn(gcells), FadeIn(rlab), FadeIn(note), run_time=0.5)

        adam_box = RoundedRectangle(corner_radius=0.18, width=2.3, height=1.3, stroke_color=OP, fill_color=OP,
                                    fill_opacity=0.25).move_to([4.0, 1.55, 0])
        opt = label("optimizer", GREY_B, 26).next_to(adam_box, UP, buff=0.25)
        self.at("optimizer")
        self.play(FadeIn(adam_box), FadeIn(opt, shift=0.1 * DOWN), run_time=0.4)
        adam = label("Adam", WHITE, 42).move_to(adam_box)
        self.at("atom")
        self.play(FadeIn(adam, scale=1.2), run_time=0.4)

        read_arrow = Arrow([0.95, gy, 0], adam_box.get_left() + np.array([0, -0.35, 0]), buff=0.1, color=RED_B,
                           stroke_width=4, max_tip_length_to_length_ratio=0.12)
        copies = VGroup(*[c[1].copy() for c in gcells])
        self.at("reads")
        self.play(GrowArrow(read_arrow),
                  LaggedStart(*[cp.animate.move_to(adam_box.get_center()).scale(0.5).set_opacity(0) for cp in copies],
                              lag_ratio=0.15), run_time=0.9)
        self.remove(copies)

        upd_arrow = Arrow(adam_box.get_left() + np.array([0, 0.35, 0]), [0.95, wy, 0], buff=0.1, color=YELLOW,
                          stroke_width=4, max_tip_length_to_length_ratio=0.12)
        new_vals = [mono(fmt(round(v, 2)), 26).move_to(wcells[k][0]) for k, v in enumerate(W8_NEW)]
        self.at("updates")
        self.play(GrowArrow(upd_arrow),
                  *[FadeTransform(wcells[k][1], new_vals[k]) for k in range(4)],
                  *[wcells[k][0].animate(rate_func=there_and_back).set_fill(YELLOW, 0.6) for k in range(4)],
                  run_time=0.8)

        zg = mono("optimizer.zero_grad()", 26, YELLOW).move_to([3.7, 0.05, 0])
        self.at("zero")
        self.play(FadeIn(zg, shift=0.1 * UP), run_time=0.35)
        zeros = [mono("0", 26, GREY_B).move_to(gcells[k][0]) for k in range(4)]
        self.at("clears")
        self.play(*[FadeTransform(gcells[k][1], zeros[k]) for k in range(4)],
                  *[gcells[k][0].animate.set_stroke(GREY_B).set_fill(GREY_B, 0.08) for k in range(4)],
                  run_time=0.6)

        divider = DashedLine([-6.3, -0.75, 0], [6.3, -0.75, 0], color=GREY_D, stroke_width=2, dash_length=0.12)
        gname = mono("x.grad", 32).move_to([-4.8, -2.15, 0])
        bin_box = RoundedRectangle(corner_radius=0.12, width=3.0, height=0.9, stroke_color=RED, stroke_width=3,
                                   fill_color=RED, fill_opacity=0.12).move_to([-1.75, -2.15, 0])
        old8 = label("8", WHITE, 40).move_to(bin_box.get_center() + np.array([-0.85, 0, 0]))
        plus = label("+", WHITE, 40).move_to(bin_box.get_center())
        new8 = label("8", WHITE, 40).move_to(bin_box.get_center() + np.array([0.85, 0, 0]))
        old_l = label("old", GREY_B, 22).next_to(old8, DOWN, buff=0.45)
        new_l = label("new", GREY_B, 22).next_to(new8, DOWN, buff=0.45)
        self.at("adds")
        self.play(Create(divider), FadeIn(gname), FadeIn(bin_box), FadeIn(old8), FadeIn(old_l), run_time=0.45)
        self.play(FadeIn(VGroup(plus, new8), shift=1.2 * LEFT), FadeIn(new_l), run_time=0.6)

        self.at("8")
        self.play(Indicate(old8, color=YELLOW), run_time=0.4)
        self.at("8")
        self.play(Indicate(new8, color=YELLOW), run_time=0.4)
        res = label("= 16 ✗", RED, 44).next_to(bin_box, RIGHT, buff=0.35)
        why = label("(forgot zero_grad)", RED, 28).next_to(res, RIGHT, buff=0.3).align_to(res, DOWN)
        self.at("16")
        self.play(FadeIn(res, shift=0.15 * LEFT), FadeIn(why), run_time=0.5)
        self.end_section()

    # 9. The code ---------------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, 26)
        code.move_to([0, 0.1, 0])
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[2])
        self.play(*self.clear_anims(), run_time=0.35)
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.at("tensor")
        self.play(Create(hl), run_time=0.3)
        for cue, i in [("compute", 3), ("backward", 4), ("read", 5)]:
            self.at(cue)
            self.play(highlight(hl, code, i), run_time=0.3)
        self.end_section()

    # 10. Where you'll see this + next up ---------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        card = used_in_card(USED_IN).move_to(ORIGIN)
        rows = card[1]
        self.at("see")
        self.play(*self.clear_anims(), FadeIn(card, shift=0.2 * UP), run_time=0.7)
        self.at("11")
        self.play(Indicate(rows[0], scale_factor=1.08), run_time=0.5)
        self.at("12")
        self.play(Indicate(rows[1], scale_factor=1.08), run_time=0.5)
        nxt = next_up_card(NEXT)
        self.at("validation")
        self.play(FadeOut(card), FadeIn(nxt, shift=0.2 * UP), run_time=0.6)
