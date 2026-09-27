"""Foundations F12 (extra) — Neural Networks in Three Minutes.

Render from the repo root:  ./render.sh foundations f12
"""
import random

import numpy as np
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card, used_in_card
from f12_script import LABEL, NEXT, TAGLINE, TITLE, USED_IN
from intro import play_token_intro
from voiced_scene import VoicedScene

INPUT_C = BLUE
NEURON_C = TEAL
BEND_C = YELLOW

# ------------------------------------------------------------------ verified numbers
# section 3: one neuron
X_IN = np.array([2.0, 3.0])
W_N = np.array([0.5, 1.0])
B_N = -1.0
assert X_IN[0] * W_N[0] == 1 and X_IN[1] * W_N[1] == 3
assert X_IN @ W_N + B_N == 3 and max(0.0, X_IN @ W_N + B_N) == 3

# section 4: a layer of three neurons (illustrative weights; column 0 is the neuron above)
W_L = np.array([[0.5, -1.0, 0.2],
                [1.0, 0.3, -0.4]])
B_L = np.array([-1.0, 0.5, 0.0])
Z_L = X_IN @ W_L + B_L
assert np.allclose(W_L[:, 0], W_N) and B_L[0] == B_N
assert np.allclose(Z_L, [3, -0.6, -0.8]) and np.allclose(np.maximum(0, Z_L), [3, 0, 0])

# section 6: the tiny GPT of episode 12
TINY_GPT_PARAMS = 818_241
assert f"{TINY_GPT_PARAMS:,}" == "818,241"

# section 7: an illustrative prediction, before and after one nudge
PRED0, PRED1, TARGET = 0.2, 0.3, 1.0
assert round(TARGET - PRED0, 10) == 0.8 and round(TARGET - PRED1, 10) == 0.7

CODE = """def neuron(x, w, b):
    return max(0.0, x @ w + b)             # weighted sum, then ReLU

x = np.array([2.0, 3.0])
neuron(x, np.array([0.5, 1.0]), -1.0)     # 3.0

def layer(x, W, b):
    return np.maximum(0, x @ W + b)        # many neurons at once"""


def _check_code():
    scope = {"np": np}
    exec(CODE.replace("neuron(x, np.array", "RESULT = neuron(x, np.array"), scope)
    assert scope["RESULT"] == 3.0
    assert np.allclose(scope["layer"](X_IN, W_L, B_L), [3, 0, 0])


_check_code()


# ------------------------------------------------------------------ helpers
def fmt(v):
    """1.0 -> '1', -0.4 -> '−0.4' (unicode minus)."""
    v = float(v)
    s = str(int(v)) if v == int(v) else f"{v:.2f}".rstrip("0")
    return s.replace("-", "−")


def label(text, color=WHITE, font_size=30, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def mono(s, size=22, color=WHITE):
    return Text(s, font=MONO, font_size=size, color=color)


def caption(text):
    return Text(text, font_size=20, color=GREY_B).to_corner(DR, buff=0.3)


def glyphs(mob, s, start, end):
    """Glyphs of Text `mob` (built from string s) for s[start:end]; spaces have no glyphs."""
    a = len(s[:start].replace(" ", ""))
    return mob[a:a + len(s[start:end].replace(" ", ""))]


def rich(text, colors=(), font_size=32, color=WHITE):
    t = Text(text, font_size=font_size, color=color)
    for sub, col in colors:
        i = text.find(sub)
        glyphs(t, text, i, i + len(sub)).set_color(col)
    return t


def kv(name, value, size=28):
    """'name: value' as two Texts sharing a baseline (name[1] has no descender)."""
    n, v = label(name, WHITE, size), label(value, WHITE, size)
    v.next_to(n, RIGHT, buff=0.18).align_to(n[1], DOWN)
    return VGroup(n, v)


def node(center, r, color, opacity=0.25, sw=3):
    return Circle(radius=r, stroke_color=color, stroke_width=sw, fill_color=color,
                  fill_opacity=opacity).move_to(center)


def link(c1, r1, c2, r2, color=GREY_B, sw=4, tip=True):
    """Arrow (or line) between two circles, touching their rims."""
    c1, c2 = np.array(c1, dtype=float), np.array(c2, dtype=float)
    u = (c2 - c1) / np.linalg.norm(c2 - c1)
    a, b = c1 + u * r1, c2 - u * r2
    if tip:
        return Arrow(a, b, buff=0, color=color, stroke_width=sw, max_tip_length_to_length_ratio=0.12,
                     max_stroke_width_to_length_ratio=10)
    return Line(a, b, color=color, stroke_width=sw)


def build_net(xs, counts, y0=0.3, gap=1.05, r=0.26, edge_w=1.5, edge_op=0.5, sw=3):
    """Columns of circles (first column = inputs) and dense edges between neighbours."""
    cols = VGroup()
    for k, (x, n) in enumerate(zip(xs, counts)):
        c = INPUT_C if k == 0 else NEURON_C
        cols.add(VGroup(*[node([x, y0 + ((n - 1) / 2 - i) * gap, 0], r, c, sw=sw) for i in range(n)]))
    edges = VGroup()
    for a, b in zip(cols[:-1], cols[1:]):
        edges.add(VGroup(*[Line(p.get_center(), q.get_center(), buff=r, stroke_color=GREY_B,
                                stroke_width=edge_w, stroke_opacity=edge_op) for p in a for q in b]))
    return cols, edges


def pulse(cols, edges, color, backward=False, run_time=1.0, width=4):
    """A flash travelling through the network, layer by layer."""
    stages = []
    order = range(len(edges) - 1, -1, -1) if backward else range(len(edges))
    for k in order:
        lines = [e.copy().set_stroke(color, width=width, opacity=1) for e in edges[k]]
        if backward:
            for ln in lines:
                ln.reverse_points()
        dest = cols[k] if backward else cols[k + 1]
        stages.append(AnimationGroup(
            AnimationGroup(*[ShowPassingFlash(ln, time_width=0.8) for ln in lines]),
            AnimationGroup(*[n.animate(rate_func=there_and_back).set_fill(color, 0.8) for n in dest]),
            lag_ratio=0.5))
    return LaggedStart(*stages, lag_ratio=0.7, run_time=run_time)


def brick_wall():
    bricks = VGroup()
    bw, bh = 0.62, 0.28
    for r, n in enumerate((4, 3, 4)):
        for k in range(n):
            x = (k - (n - 1) / 2) * (bw + 0.06)
            bricks.add(RoundedRectangle(corner_radius=0.05, width=bw, height=bh, stroke_color=ORANGE, stroke_width=2,
                                        fill_color=ORANGE, fill_opacity=0.35).move_to([x, -r * (bh + 0.06), 0]))
    return bricks


def gear(r=0.34, teeth=8, color=GREY_B):
    body = Circle(radius=r, stroke_color=color, stroke_width=3, fill_color=color, fill_opacity=0.3)
    hole = Circle(radius=r * 0.35, stroke_color=color, stroke_width=3)
    cogs = VGroup()
    for k in range(teeth):
        a = TAU * k / teeth
        cogs.add(Square(0.15, stroke_width=0, fill_color=color, fill_opacity=0.9)
                 .rotate(a).move_to((r + 0.05) * np.array([np.cos(a), np.sin(a), 0])))
    return VGroup(cogs, body, hole)


def relu_graph(center, s=0.62, color=BEND_C):
    """A tiny ReLU plot (no numbers): axes plus the bent line; returns (axes, curve, point(x))."""
    o = np.array(center, dtype=float) + np.array([0, -0.3, 0])
    axes = VGroup(Line(o + [-s - 0.05, 0, 0], o + [s + 0.05, 0, 0], color=GREY_C, stroke_width=2),
                  Line(o + [0, -0.18, 0], o + [0, s + 0.1, 0], color=GREY_C, stroke_width=2))
    curve = VMobject(stroke_color=color, stroke_width=5).set_points_as_corners(
        [o + [-s, 0, 0], o, o + [s, s, 0]])

    def point(t):   # t in [-1, 1] along the x range
        return o + np.array([t * s, max(0.0, t) * s, 0])

    return axes, curve, point


def doc_icon(color=GREY_B):
    page = RoundedRectangle(corner_radius=0.06, width=0.5, height=0.64, stroke_color=color, stroke_width=2,
                            fill_color=GREY_E, fill_opacity=1)
    lines = VGroup(*[Line([-0.15, 0.16 - 0.12 * k, 0], [0.15 - (0.08 if k == 3 else 0), 0.16 - 0.12 * k, 0],
                          color=color, stroke_width=2) for k in range(4)])
    return VGroup(page, lines)


def knob(r=0.5):
    """A dial: body, tick marks, and a pointer (returned separately so it can rotate)."""
    body = Circle(radius=r, stroke_color=GREY_B, stroke_width=3, fill_color=GREY_E, fill_opacity=1)
    ticks = VGroup()
    for a in np.linspace(5 * PI / 4, -PI / 4, 7):
        d = np.array([np.cos(a), np.sin(a), 0])
        ticks.add(Line((r + 0.07) * d, (r + 0.2) * d, color=GREY_B, stroke_width=2))
    a0 = 5 * PI / 4
    pointer = Line(ORIGIN, (r - 0.1) * np.array([np.cos(a0), np.sin(a0), 0]), color=YELLOW, stroke_width=6)
    return VGroup(ticks, body), pointer


class NeuralNetVideo(VoicedScene):
    VIDEO = "f12"

    # ------------------------------------------------------------------ stage helpers
    def on_stage(self, keep=()):
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
        self.s1_what()
        self.s2_neuron()
        self.s3_example()
        self.s4_layer()
        self.s5_stack()
        self.s6_parameters()
        self.s7_two_ways()
        self.s8_shape()
        self.s9_code()
        self.s10_outro()
        finish(self)

    # 1. What is a neural network? ----------------------------------------------------------------
    def s1_what(self):
        self.section(1)
        words = [("neural network", YELLOW, 52, (0, 0.3)), ("AI", BLUE_B, 60, (-4.8, 2.3)),
                 ("brain", PINK, 42, (4.5, 2.4)), ("deep learning", TEAL, 36, (-4.0, -1.5)),
                 ("neurons", GREY_B, 34, (4.6, -1.2)), ("intelligence", PURPLE_B, 32, (-1.1, 2.7)),
                 ("magic?", GOLD, 34, (2.4, 1.6)), ("black box", GREY_B, 30, (-2.0, -2.8)),
                 ("synapses", GREY_B, 30, (2.6, -2.6))]
        cloud = VGroup(*[label(w, c, fs).move_to([x, y, 0]) for w, c, fs, (x, y) in words])
        self.at("neural")
        self.play(LaggedStart(*[FadeIn(m, scale=0.8) for m in cloud], lag_ratio=0.12), run_time=1.3)
        self.at("hype")
        self.play(LaggedStart(*[FadeOut(m, scale=1.2) for m in cloud], lag_ratio=0.04), run_time=0.6)

        fbox = RoundedRectangle(corner_radius=0.2, width=2.6, height=1.7, stroke_color=BLUE, fill_color=BLUE,
                                fill_opacity=0.2).move_to([0, 0.8, 0])
        ftxt = Text("f", font_size=84, slant=ITALIC).move_to(fbox)
        a_in = Arrow([-3.6, 0.8, 0], fbox.get_left(), buff=0.1, color=GREY_B, stroke_width=5)
        a_out = Arrow(fbox.get_right(), [3.6, 0.8, 0], buff=0.1, color=GREY_B, stroke_width=5)
        l_in = label("input", GREY_B, 30).next_to(a_in, LEFT, buff=0.2)
        l_out = label("output", GREY_B, 30).next_to(a_out, RIGHT, buff=0.2)
        head = label("a function", WHITE, 36).move_to([0, 2.7, 0])
        self.at("function")
        self.play(FadeIn(fbox), FadeIn(ftxt), GrowArrow(a_in), GrowArrow(a_out), FadeIn(l_in), FadeIn(l_out),
                  FadeIn(head, shift=0.15 * DOWN), run_time=0.6)

        wall = brick_wall().move_to([0, -1.55, 0])
        gears = VGroup(gear().move_to([-2.0, -1.55, 0]), gear().move_to([2.0, -1.55, 0]))
        parts = label("built from very simple parts", GREY_B, 26).move_to([0, -2.75, 0])
        self.at("parts")
        self.play(LaggedStart(*[FadeIn(b, shift=0.15 * DOWN) for b in wall], lag_ratio=0.06),
                  FadeIn(gears, scale=0.7), FadeIn(parts), run_time=0.55)
        self.end_section()

    # 2. A neuron ------------------------------------------------------------------------------
    IN1, IN2, R_IN = np.array([-5.0, 1.1, 0]), np.array([-5.0, -1.1, 0]), 0.42
    SIG, R_SIG = np.array([-1.4, 0, 0]), 0.6
    BIAS = np.array([-1.4, 2.05, 0])
    BOX = np.array([1.6, 0, 0])
    OUT, R_OUT = np.array([4.5, 0, 0]), 0.45

    def s2_neuron(self):
        self.section(2)
        self.play(*self.clear_anims(), run_time=0.4)
        head = label("a neuron", WHITE, 38).move_to([0, 3.2, 0])
        self.at("neuron")
        self.play(FadeIn(head, shift=0.15 * DOWN), run_time=0.5)

        in1 = node(self.IN1, self.R_IN, INPUT_C)
        in2 = node(self.IN2, self.R_IN, INPUT_C)
        self.in_txt = [label("x₁", WHITE, 32).move_to(in1), label("x₂", WHITE, 32).move_to(in2)]
        self.inputs = VGroup(in1, in2)
        self.at("numbers")
        self.play(FadeIn(self.inputs, scale=0.8), FadeIn(self.in_txt[0]), FadeIn(self.in_txt[1]), run_time=0.5)

        e1 = link(self.IN1, self.R_IN, self.SIG, self.R_SIG)
        e2 = link(self.IN2, self.R_IN, self.SIG, self.R_SIG)
        self.nedges = VGroup(e1, e2)
        self.at("multiplies")
        self.play(GrowArrow(e1), GrowArrow(e2), run_time=0.5)

        def edge_label_pos(a, b, up):
            u = (b - a) / np.linalg.norm(b - a)
            perp = np.array([-u[1], u[0], 0]) * (1 if up else -1)
            return (a + b) / 2 + 0.42 * perp

        self.w_pos = [edge_label_pos(self.IN1, self.SIG, True), edge_label_pos(self.IN2, self.SIG, False)]
        self.w_txt = [label("× w₁", YELLOW, 28).move_to(self.w_pos[0]),
                      label("× w₂", YELLOW, 28).move_to(self.w_pos[1])]
        self.at("weight")
        self.play(FadeIn(self.w_txt[0], scale=0.8), FadeIn(self.w_txt[1], scale=0.8), run_time=0.4)

        self.sig = node(self.SIG, self.R_SIG, NEURON_C, opacity=0.2)
        self.sig_txt = label("Σ", WHITE, 44).move_to(self.SIG)
        self.at("adds")
        self.play(FadeIn(self.sig, scale=0.7), FadeIn(self.sig_txt), run_time=0.4)

        self.bias_box = RoundedRectangle(corner_radius=0.1, width=1.0, height=0.62, stroke_color=PURPLE_B,
                                         fill_color=PURPLE_B, fill_opacity=0.25).move_to(self.BIAS)
        self.bias_txt = label("b", WHITE, 30).move_to(self.bias_box)
        self.bias_arrow = Arrow(self.bias_box.get_bottom(), self.SIG + [0, self.R_SIG, 0], buff=0.05,
                                color=GREY_B, stroke_width=4, max_tip_length_to_length_ratio=0.25)
        self.bias_plus = label("+", WHITE, 32).next_to(self.bias_arrow, RIGHT, buff=0.12)
        self.at("bias")
        self.play(FadeIn(self.bias_box), FadeIn(self.bias_txt), GrowArrow(self.bias_arrow),
                  FadeIn(self.bias_plus), run_time=0.4)

        self.box = RoundedRectangle(corner_radius=0.12, width=1.6, height=1.6, stroke_color=GREY_B,
                                    fill_color=GREY_E, fill_opacity=0.6).move_to(self.BOX)
        self.gaxes, self.gcurve, self.gpoint = relu_graph(self.BOX)
        self.z_arrow = Arrow(self.SIG + [self.R_SIG, 0, 0], self.box.get_left(), buff=0.06, color=GREY_B,
                             stroke_width=4, max_tip_length_to_length_ratio=0.2)
        self.at("bend")
        self.play(GrowArrow(self.z_arrow), FadeIn(self.box), Create(self.gaxes), Create(self.gcurve), run_time=0.6)

        self.relu_lab = label("ReLU", BEND_C, 28).next_to(self.box, DOWN, buff=0.2)
        self.out = node(self.OUT, self.R_OUT, GREY_B, opacity=0.1)
        self.out_arrow = Arrow(self.box.get_right(), self.OUT - [self.R_OUT, 0, 0], buff=0.06, color=GREY_B,
                               stroke_width=4, max_tip_length_to_length_ratio=0.2)
        self.out_lab = label("output", GREY_B, 26).next_to(self.out, DOWN, buff=0.2)
        self.at("re")
        self.play(FadeIn(self.relu_lab), GrowArrow(self.out_arrow), FadeIn(self.out), FadeIn(self.out_lab),
                  run_time=0.5)
        self.end_section()

    # 3. A worked example --------------------------------------------------------------------------
    def swap(self, old, text, color=WHITE, size=32, pos=None):
        new = label(text, color, size).move_to(old.get_center() if pos is None else pos)
        return new, [FadeOut(old, scale=0.8), FadeIn(new, scale=1.2)]

    def s3_example(self):
        self.section(3)
        for k, (cue, val) in enumerate((("2", "2"), ("3", "3"))):
            new, anims = self.swap(self.in_txt[k], val, WHITE, 36)
            self.in_txt[k] = new
            self.at(cue)
            self.play(*anims, run_time=0.3)
        for k, (cue, val) in enumerate((("5", "× 0.5"), ("1", "× 1"))):
            new, anims = self.swap(self.w_txt[k], val, YELLOW, 30)
            self.w_txt[k] = new
            self.at(cue)
            self.play(*anims, run_time=0.3)
        new, anims = self.swap(self.bias_txt, "−1", WHITE, 32)
        self.bias_txt = new
        self.at("minus")
        self.play(*anims, run_time=0.3)

        def term(num, sub):
            n = label(num, WHITE, 42)
            s = label(sub, GREY_B, 22)
            return VGroup(n, s.next_to(n, DOWN, buff=0.2))

        t1, t2, t3 = term("1", "2 × 0.5"), term("3", "3 × 1"), term("1", "bias")
        ops = [label(o, WHITE, 42) for o in ("+", "−", "=")]
        res = label("3", YELLOW, 42)
        row = VGroup(t1, ops[0], t2, ops[1], t3, ops[2], res).arrange(RIGHT, buff=0.3)
        for m in (ops[0], ops[1], ops[2], res):
            m.match_y(t1[0])
        row.move_to([-1.2, -2.5, 0])
        self.at("thats")
        self.at("1")
        self.play(FadeIn(t1, shift=0.15 * UP),
                  self.nedges[0].animate(rate_func=there_and_back).set_color(YELLOW), run_time=0.25)
        self.at("plus")
        self.play(FadeIn(ops[0]), FadeIn(t2, shift=0.15 * UP),
                  self.nedges[1].animate(rate_func=there_and_back).set_color(YELLOW), run_time=0.35)
        self.at("minus")
        self.play(FadeIn(ops[1]), FadeIn(t3, shift=0.15 * UP),
                  self.bias_arrow.animate(rate_func=there_and_back).set_color(YELLOW), run_time=0.3)
        z_lab = label("3", YELLOW, 34).next_to(self.z_arrow, UP, buff=0.12)
        self.at("3")
        self.play(FadeIn(ops[2]), FadeIn(res, scale=1.3), FadeIn(z_lab, shift=0.1 * RIGHT), run_time=0.45)

        dot = Dot(self.gpoint(0.85), radius=0.09, color=WHITE)
        relu3 = rich("ReLU(3) = 3", [("ReLU", BEND_C)], font_size=32).next_to(self.box, UP, buff=0.3)
        self.at("relu")
        self.play(FadeIn(dot, scale=1.5), FadeIn(relu3, shift=0.1 * DOWN), run_time=0.5)

        glow = Circle(radius=self.R_SIG + 0.3, stroke_width=0, fill_color=GREEN, fill_opacity=0.18).move_to(self.SIG)
        out_val = label("3", WHITE, 36).move_to(self.out)
        fires = label("fires!", GREEN, 30).next_to(self.out, UP, buff=0.25)
        self.at("fires")
        self.play(FadeIn(glow, scale=0.6), self.sig.animate.set_stroke(GREEN).set_fill(GREEN, 0.35),
                  self.box.animate.set_stroke(GREEN), self.z_arrow.animate.set_color(GREEN),
                  self.out_arrow.animate.set_color(GREEN), self.out.animate.set_stroke(GREEN).set_fill(GREEN, 0.4),
                  FadeIn(out_val, scale=1.3), FadeIn(fires, shift=0.1 * DOWN), run_time=0.6)
        self.at("3")
        self.play(Indicate(out_val, color=WHITE, scale_factor=1.4), run_time=0.4)
        self.end_section()

    # 4. A layer -------------------------------------------------------------------------------------
    def s4_layer(self):
        self.section(4)
        keep = [self.inputs, *self.in_txt]
        self.play(*self.clear_anims(keep), run_time=0.4)

        ny = [1.9, 0.0, -1.9]
        nx, r = -1.6, 0.42
        neurons = VGroup(*[node([nx, y, 0], r, NEURON_C, opacity=0.2) for y in ny])
        ins = [self.IN1, self.IN2]
        edges = [[Line(ins[i], neurons[j].get_center(), buff=r, stroke_color=GREY_B, stroke_width=3)
                  for j in range(3)] for i in range(2)]
        all_edges = VGroup(*[e for row in edges for e in row])
        self.add(all_edges)
        all_edges.set_opacity(0)
        self.at("side")
        self.play(LaggedStart(*[FadeIn(n, shift=0.2 * LEFT) for n in neurons], lag_ratio=0.2),
                  all_edges.animate.set_stroke(opacity=1), run_time=0.7)

        self.at("inputs")
        self.play(LaggedStart(*[ShowPassingFlash(e.copy().set_stroke(YELLOW, width=6), time_width=0.6)
                                for e in all_edges], lag_ratio=0.08), run_time=0.8)

        brace = Brace(neurons, RIGHT, buff=0.2, color=GREY_B)
        layer_lab = label("layer", WHITE, 34).next_to(brace, RIGHT, buff=0.2)
        self.at("layer")
        self.play(GrowFromCenter(brace), FadeIn(layer_lab, shift=0.1 * RIGHT), run_time=0.5)

        # weight labels sit on the edges near the neurons, then collapse into the matrix W
        wl = [[label(fmt(W_L[i, j]), YELLOW, 26).add_background_rectangle(opacity=1, buff=0.05)
               .move_to(ins[i] + 0.7 * (neurons[j].get_center() - ins[i])) for j in range(3)] for i in range(2)]
        gx, gy, dx, dy = 3.7, 1.0, 1.0, 0.7
        cells = [[np.array([gx + (j - 1) * dx, gy + (0.5 - i) * dy, 0]) for j in range(3)] for i in range(2)]
        top, bot = gy + 0.5 * dy + 0.33, gy - 0.5 * dy - 0.33
        lb, rb = gx - 1.6, gx + 1.6

        def bracket(x, s):
            return VMobject(stroke_color=WHITE, stroke_width=3).set_points_as_corners(
                [[x + s * 0.15, top, 0], [x, top, 0], [x, bot, 0], [x + s * 0.15, bot, 0]])

        brackets = VGroup(bracket(lb, 1), bracket(rb, -1))
        w_name = label("W =", WHITE, 38).next_to(brackets[0], LEFT, buff=0.25)
        w_note = label("row = input, column = neuron", GREY_B, 20).next_to(brackets, DOWN, buff=0.25)
        b_vec = label("b = [−1, 0.5, 0]", WHITE, 30).next_to(w_note, DOWN, buff=0.35)
        note = caption("illustrative weights")
        self.at("matrix")
        self.play(all_edges.animate.set_stroke(opacity=0.35),
                  LaggedStart(*[FadeIn(t, scale=0.8) for row in wl for t in row], lag_ratio=0.08), run_time=0.45)
        for i in range(2):
            for j in range(3):
                wl[i][j].generate_target()
                wl[i][j].target.move_to(cells[i][j])
                wl[i][j].target[0].set_fill(opacity=0)   # drop the backing so it never hides a neuron in transit
        self.play(*[MoveToTarget(wl[i][j]) for i in range(2) for j in range(3)],
                  FadeIn(brackets), FadeIn(w_name), FadeIn(w_note), FadeIn(b_vec), FadeIn(note),
                  all_edges.animate.set_stroke(opacity=1), run_time=0.8)

        cap_s = "layer = ReLU(x · W + b)"
        cap = rich(cap_s, [("ReLU", BEND_C)], font_size=34)
        cap_box = SurroundingRectangle(cap, buff=0.25, corner_radius=0.12, stroke_color=GREY_B, stroke_width=2)
        VGroup(cap, cap_box).move_to([2.9, -2.4, 0])
        self.at("multiplication")
        self.play(FadeIn(cap, shift=0.15 * UP), Create(cap_box), run_time=0.5)

        outs = [label(fmt(v), WHITE, 32).move_to(neurons[j]) for j, v in enumerate(np.maximum(0, Z_L))]
        self.at("bend")
        self.play(neurons[0].animate.set_stroke(GREEN).set_fill(GREEN, 0.4),
                  *[FadeIn(o, scale=1.3) for o in outs],
                  Indicate(glyphs(cap, cap_s, cap_s.find("ReLU"), cap_s.find("ReLU") + 4), color=BEND_C, scale_factor=1.1),
                  run_time=0.45)
        self.end_section()

    # 5. Stacking layers -----------------------------------------------------------------------------
    NET_XS = [-3.6, -1.2, 1.2, 3.6]
    NET_COUNTS = [3, 4, 4, 4]

    def s5_stack(self):
        self.section(5)
        cols, edges = build_net(self.NET_XS, self.NET_COUNTS)
        self.cols, self.edges = cols, edges
        tags = VGroup(label("inputs", GREY_B, 24).next_to(cols[0], UP, buff=0.55),
                      *[label(f"layer {k}", WHITE, 26).next_to(cols[k], UP, buff=0.3) for k in (1, 2, 3)])
        for t in tags:
            t.set_y(cols[1].get_top()[1] + 0.4)
        self.tags = tags
        self.at("stack")
        self.play(*self.clear_anims(), run_time=0.35)
        self.play(FadeIn(cols[0]), FadeIn(tags[0]),
                  LaggedStart(*[AnimationGroup(FadeIn(edges[k - 1]), FadeIn(cols[k], shift=0.2 * LEFT), FadeIn(tags[k]))
                                for k in (1, 2, 3)], lag_ratio=0.45), run_time=1.0)

        self.at("next")
        self.play(pulse(cols, edges, YELLOW, run_time=0.9))

        y_lab = -2.2
        simple = label("simple patterns", WHITE, 28).move_to([self.NET_XS[1], y_lab, 0])
        strokes = VGroup(Line([-0.25, 0, 0], [0.25, 0, 0]), Line([0, -0.25, 0], [0, 0.25, 0]),
                         Line([-0.2, -0.2, 0], [0.2, 0.2, 0]), Arc(radius=0.22, start_angle=0, angle=PI))
        strokes.set_stroke(TEAL, 4)
        strokes.arrange(RIGHT, buff=0.35).next_to(simple, DOWN, buff=0.3)
        self.at("simple")
        self.play(FadeIn(simple, shift=0.15 * UP), LaggedStart(*[Create(s) for s in strokes], lag_ratio=0.2),
                  FadeIn(caption("illustrative")), run_time=0.5)

        complex_ = label("complex patterns", WHITE, 28).move_to([self.NET_XS[3], y_lab, 0])
        house = VGroup(Square(0.5), Polygon([-0.3, 0.25, 0], [0.3, 0.25, 0], [0, 0.55, 0]),
                       Line([-0.08, -0.25, 0], [-0.08, 0.02, 0]), Line([0.08, -0.25, 0], [0.08, 0.02, 0]))
        face = VGroup(Circle(radius=0.3), Dot([-0.11, 0.08, 0], radius=0.04), Dot([0.11, 0.08, 0], radius=0.04),
                      Arc(radius=0.15, start_angle=-PI * 0.85, angle=PI * 0.7).shift(0.02 * DOWN))
        icons = VGroup(house, face).set_stroke(PURPLE_B, 4)
        for d in face[1:3]:
            d.set_fill(PURPLE_B, 1)
        icons.arrange(RIGHT, buff=0.5).next_to(complex_, DOWN, buff=0.3)
        arrow = Arrow(simple.get_right(), complex_.get_left(), buff=0.3, color=GREY_B, stroke_width=4)
        self.at("later")
        self.play(GrowArrow(arrow), run_time=0.5)
        self.at("complex")
        self.play(FadeIn(complex_, shift=0.15 * UP), FadeIn(icons, shift=0.15 * UP), run_time=0.45)
        self.end_section()

    # 6. Parameters ----------------------------------------------------------------------------------
    def s6_parameters(self):
        self.section(6)
        cols, edges = self.cols, self.edges
        flat_edges = [e for g in edges for e in g]
        neurons = [n for c in cols[1:] for n in c]
        self.play(*self.clear_anims(keep=[cols, edges]),
                  *[e.animate.set_stroke(YELLOW, width=2.2, opacity=0.85) for e in flat_edges], run_time=0.5)

        self.b_labs = VGroup(*[label("b", YELLOW, 24).move_to(n) for n in neurons])
        self.at("biases")
        self.play(*[n.animate.set_stroke(YELLOW) for n in neurons], FadeIn(self.b_labs), run_time=0.45)

        head = rich("parameters = weights + biases", [("parameters", YELLOW)], font_size=36).move_to([0, 3.2, 0])
        self.at("parameters")
        self.play(FadeIn(head, shift=0.15 * DOWN), run_time=0.5)

        rng = random.Random(12)
        self.at("changes")
        for _ in range(5):
            self.play(*[e.animate.set_stroke(rng.choice([BLUE, BLUE_B, RED, RED_B]), width=rng.uniform(0.8, 4.2),
                                             opacity=0.9) for e in flat_edges], run_time=0.34, rate_func=smooth)

        tiny = rich(f"tiny GPT: {TINY_GPT_PARAMS:,}", [(f"{TINY_GPT_PARAMS:,}", YELLOW)], font_size=34)
        tiny.move_to([-3.3, -2.55, 0])
        self.at("800")
        self.play(FadeIn(tiny, shift=0.15 * UP), run_time=0.5)
        big = rich("big models: billions", [("billions", YELLOW)], font_size=34).move_to([3.3, -2.55, 0])
        self.at("billions")
        self.play(FadeIn(big, shift=0.15 * UP), run_time=0.5)
        self.end_section()

    # 7. Inference and training ----------------------------------------------------------------------
    def s7_two_ways(self):
        self.section(7)
        cols, edges = self.cols, self.edges
        net = VGroup(edges, cols)
        flat_edges = [e for g in edges for e in g]
        neurons = [n for c in cols[1:] for n in c]
        self.play(*self.clear_anims(keep=[net]),
                  *[e.animate.set_stroke(GREY_B, width=1.5, opacity=0.5) for e in flat_edges],
                  *[n.animate.set_stroke(NEURON_C) for n in neurons], run_time=0.5)
        self.play(net.animate.shift(2.2 * LEFT), run_time=0.6)

        h1 = rich("inference: inputs → prediction", [("inference", YELLOW)], font_size=36).move_to([0, 3.2, 0])
        self.at("inference")
        self.play(FadeIn(h1, shift=0.15 * DOWN), run_time=0.5)
        self.at("flow")
        self.play(pulse(cols, edges, YELLOW, run_time=1.0))

        y_p, y_a, x0, tw, th = 0.3, -1.55, 3.05, 2.8, 0.42
        arrow = Arrow([cols[3].get_right()[0] + 0.1, y_p, 0], [x0 - 0.1, y_p, 0], buff=0, color=GREY_B,
                      stroke_width=4, max_tip_length_to_length_ratio=0.25)

        def track(y):
            return Rectangle(width=tw, height=th, stroke_color=GREY_B, stroke_width=2).move_to(
                [x0 + tw / 2, y, 0])

        def fill(y, a, b, color, op=0.85):
            return Rectangle(width=(b - a) * tw, height=th, stroke_width=0, fill_color=color,
                             fill_opacity=op).move_to([x0 + (a + b) / 2 * tw, y, 0])

        t1 = track(y_p)
        f1 = fill(y_p, 0, PRED0, BLUE)
        p_row = kv("prediction:", fmt(PRED0)).next_to(t1, UP, buff=0.18).align_to(t1, LEFT)
        p_val = p_row[1]
        self.at("prediction")
        self.play(GrowArrow(arrow), FadeIn(t1), FadeIn(f1), FadeIn(p_row), FadeIn(caption("illustrative")),
                  run_time=0.5)

        h2 = rich("training: compare, then nudge", [("training", RED)], font_size=36).move_to([0, 3.2, 0])
        self.at("training")
        self.play(FadeOut(h1), FadeIn(h2), run_time=0.5)

        t2 = track(y_a)
        f2 = fill(y_a, 0, TARGET, GREEN)
        a_row = kv("right answer:", f"{TARGET:.1f}").next_to(t2, UP, buff=0.18).align_to(t2, LEFT)
        err = fill(y_p, PRED0, TARGET, RED, op=0.35)
        err_lab = label(f"error {fmt(TARGET - PRED0)}", RED, 24).next_to(t1, DOWN, buff=0.12).align_to(t1, RIGHT)
        cross = label("✗", RED, 36).next_to(t1, RIGHT, buff=0.2)
        check = label("✓", GREEN, 36).next_to(t2, RIGHT, buff=0.2)
        self.at("compare")
        self.play(FadeIn(t2), FadeIn(f2), FadeIn(a_row), FadeIn(check), run_time=0.4)
        self.play(FadeIn(err), FadeIn(err_lab), FadeIn(cross, scale=1.3), run_time=0.4)

        rng = random.Random(7)
        picks = rng.sample(flat_edges, 10)
        signs = VGroup()
        for k, e in enumerate(picks):
            s = "+" if k % 2 == 0 else "−"
            signs.add(label(s, GREEN if s == "+" else RED_B, 30).move_to(e.point_from_proportion(0.5) + 0.15 * UP))
        self.at("nudge")
        self.play(pulse(cols, edges, RED, backward=True, run_time=1.2, width=5),
                  LaggedStart(*[FadeIn(s, scale=1.4) for s in signs], lag_ratio=0.12, run_time=1.2))

        f1b = fill(y_p, 0, PRED1, BLUE)
        errb = fill(y_p, PRED1, TARGET, RED, op=0.35)
        p_val2 = label(fmt(PRED1), WHITE, 28).move_to(p_val)
        err_lab2 = label(f"error {fmt(TARGET - PRED1)}", RED, 24).move_to(err_lab)
        self.at("better")
        self.play(Transform(f1, f1b), Transform(err, errb), FadeOut(p_val), FadeIn(p_val2),
                  FadeOut(err_lab), FadeIn(err_lab2), FadeOut(signs), run_time=0.5)
        self.end_section()

    # 8. We choose the shape -------------------------------------------------------------------------
    def s8_shape(self):
        self.section(8)
        self.play(*self.clear_anims(), run_time=0.4)
        hy = 2.35
        # column 1: hand-written rules
        c1x = -4.55
        rules_head = label("hand-written rules", WHITE, 28).move_to([c1x, hy, 0])
        card = RoundedRectangle(corner_radius=0.15, width=3.9, height=2.3, stroke_color=GREY_B, stroke_width=2,
                                fill_color=GREY_E, fill_opacity=0.6).move_to([c1x, 0.2, 0])
        src = ['if word == "the":', '    guess = "mat"', 'elif word == "on":', '    guess = "the"']
        char_w = mono("x" * 20, 20).width / 20
        code = VGroup(*[mono(ln.strip(), 20) for ln in src]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        for ln, t in zip(src, code):
            t.shift((len(ln) - len(ln.lstrip())) * char_w * RIGHT)
        code.move_to(card)
        self.play(FadeIn(rules_head), FadeIn(card), FadeIn(code), run_time=0.4)
        cross = VGroup(Line(card.get_corner(UL) + [0.2, -0.15, 0], card.get_corner(DR) + [-0.2, 0.15, 0]),
                       Line(card.get_corner(DL) + [0.2, 0.15, 0], card.get_corner(UR) + [-0.2, -0.15, 0]))
        cross.set_stroke(RED, 9)
        self.at("rules")
        self.play(Create(cross), code.animate.set_opacity(0.4), run_time=0.45)

        # column 2: the knobs
        kx = -1.25
        shape_head = label("we choose the shape", WHITE, 28).move_to([0, hy, 0])
        k1, p1 = knob()
        k2, p2 = knob()
        VGroup(k1, p1).move_to([kx, 1.0, 0])
        VGroup(k2, p2).move_to([kx, -0.75, 0])
        p1.put_start_and_end_on(k1[1].get_center(), p1.get_end())
        p2.put_start_and_end_on(k2[1].get_center(), p2.get_end())
        n1 = label("layers", GREY_B, 26).next_to(k1, RIGHT, buff=0.35).shift(0.2 * UP)
        n2 = label("neurons", GREY_B, 26).next_to(k2, RIGHT, buff=0.35).shift(0.2 * UP)
        v1 = label("3", YELLOW, 30).next_to(n1, DOWN, buff=0.12).align_to(n1, LEFT)
        v2 = label("4 per layer", YELLOW, 28).next_to(n2, DOWN, buff=0.12).align_to(n2, LEFT)
        self.at("shape")
        self.play(FadeIn(shape_head), FadeIn(k1), FadeIn(p1), FadeIn(k2), FadeIn(p2), FadeIn(n1), FadeIn(n2),
                  run_time=0.45)
        self.at("layers")
        self.play(Rotate(p1, angle=-PI / 2, about_point=k1[1].get_center()), FadeIn(v1, shift=0.1 * UP),
                  run_time=0.45)

        # column 3: the (untrained) network
        c3x = 4.55
        ncols, nedges = build_net([c3x - 1.1, c3x - 0.35, c3x + 0.4, c3x + 1.15], [3, 4, 4, 4], y0=0.1, gap=0.42,
                                  r=0.11, edge_w=1.2, edge_op=0.6, sw=2)
        mini = VGroup(nedges, ncols)
        to_net = Arrow([v2.get_right()[0] + 0.15, 0.1, 0], [ncols.get_left()[0] - 0.2, 0.1, 0], buff=0,
                       color=GREY_B, stroke_width=4, max_tip_length_to_length_ratio=0.25)
        net_lab = label("untrained network", GREY_B, 26).move_to([c3x, -1.45, 0])
        self.at("neurons")
        self.play(Rotate(p2, angle=-PI * 0.75, about_point=k2[1].get_center()), FadeIn(v2, shift=0.1 * UP),
                  GrowArrow(to_net), FadeIn(mini), FadeIn(net_lab), run_time=0.5)

        docs = VGroup(*[doc_icon() for _ in range(4)]).arrange(RIGHT, buff=0.18).move_to([c3x, hy, 0])
        self.at("data")
        self.play(FadeIn(docs, shift=0.2 * DOWN), run_time=0.25)
        self.play(LaggedStart(*[d.animate.move_to(mini.get_center()).scale(0.3).set_opacity(0) for d in docs],
                              lag_ratio=0.15), run_time=0.5)
        self.remove(docs)
        trained = label("trained network", GREEN, 28).move_to(net_lab)
        self.at("rest")
        self.play(*[e.animate.set_stroke(GREEN, opacity=0.8) for g in nedges for e in g],
                  *[n.animate.set_stroke(GREEN).set_fill(GREEN, 0.5) for c in ncols[1:] for n in c],
                  FadeOut(net_lab), FadeIn(trained), run_time=0.45)
        self.end_section()

    # 9. The code ------------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, font_size=26)
        code.move_to([0, 0.1, 0])
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[1])
        self.play(*self.clear_anims(), FadeIn(code, shift=0.3 * UP), run_time=0.5)
        self.at("neuron")
        self.play(Create(hl), run_time=0.3)
        self.at("re")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.at("layer")
        self.play(highlight(hl, code, 7), run_time=0.3)
        self.end_section()

    # 10. Outro --------------------------------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        card = used_in_card(USED_IN).move_to(ORIGIN)
        rows = card[1]
        self.at("see")
        self.play(*self.clear_anims(), FadeIn(card, shift=0.2 * UP), run_time=0.7)
        for cue, k in (("eight", 0), ("11", 1), ("12", 2)):
            self.at(cue)
            self.play(Indicate(rows[k], scale_factor=1.08), run_time=0.5)
        nxt = next_up_card(NEXT)
        self.at("gradients")
        self.play(FadeOut(card), FadeIn(nxt, shift=0.2 * UP), run_time=0.6)
        self.end_section()
