"""Foundations F11 — NumPy in Three Minutes.

Render from the repo root:  ./render.sh foundations f11
"""
import random

import numpy as np
from manim import *

from common import code_panel, finish, next_up_card
from f11_script import LABEL, NEXT, TAGLINE, TITLE
from intro import play_token_intro
from voiced_scene import VoicedScene

MONO = "DejaVu Sans Mono"
WORDS = ["The", "cat", "sat", "on", "the"]

# ---------------------------------------------------------------------------- worked examples
# section 2: an illustrative 5 x 4 array (one row per token) and a 4-number vector
X2 = np.array([[0.2, -1.3, 0.7, 2.1],
               [1.5, 0.4, -0.6, 0.9],
               [-0.8, 1.1, 0.3, -1.7],
               [0.6, -0.2, 1.8, 0.5],
               [-1.4, 0.9, -0.5, 1.2]])
V2 = np.array([0.3, -1.1, 2.4, 0.8])
assert X2.shape == (5, 4) and V2.ndim == 1

# section 3: element-wise arithmetic
A3 = np.array([[1, 2, 3], [4, 5, 6]])
B3 = np.array([[10, 20, 30], [40, 50, 60]])
assert (A3 + B3).tolist() == [[11, 22, 33], [44, 55, 66]]
assert (A3 * 2).tolist() == [[2, 4, 6], [8, 10, 12]]

# section 4: shapes of the matrix product
assert (np.random.randn(5, 4) @ np.random.randn(4, 4)).shape == (5, 4)

# section 5: sum along the last axis
Y5 = np.array([[2, 1, 0, 3],
               [1, 4, 2, 1],
               [0, 2, 5, 1],
               [3, 3, 1, 2],
               [1, 0, 2, 2]])
T5 = Y5.sum(axis=-1)
assert T5.tolist() == [6, 8, 8, 9, 5] and T5.shape == (5,)
assert Y5.sum(axis=-1, keepdims=True).shape == (5, 1)

# section 6: broadcasting — divide each row by its own total (as softmax does)
E6 = np.array([[1, 2, 4, 2, 1],
               [1, 1, 1, 1, 1],
               [2, 6, 4, 4, 4],
               [5, 5, 10, 3, 2],
               [6, 1, 1, 1, 1]])
S6 = E6.sum(axis=-1, keepdims=True)
P6 = E6 / S6
assert S6.shape == (5, 1) and S6.ravel().tolist() == [10, 5, 20, 25, 10]
assert np.allclose(P6.sum(axis=-1), 1.0)
assert np.allclose(P6 * 100, np.round(P6 * 100))          # every quotient has at most two decimals

# section 7: reshape and transpose
assert np.arange(768).reshape(12, 64).shape == (12, 64)
assert np.arange(6).reshape(2, 3).tolist() == [[0, 1, 2], [3, 4, 5]]
assert np.arange(6).reshape(2, 3).T.tolist() == [[0, 3], [1, 4], [2, 5]]

# section 8: the causal mask
MASK = np.triu(np.ones((5, 5)), k=1).astype(bool)
_scores = np.zeros((5, 5))
_scores[MASK] = -np.inf
assert all(MASK[i, j] == (j > i) for i in range(5) for j in range(5))
assert np.isneginf(_scores[MASK]).all() and (_scores[~MASK] == 0).all()

CODE = """X = np.random.randn(5, 4)               # (5, 4)
W = np.random.randn(4, 4)
Y = X @ W                               # (5, 4)
totals = Y.sum(axis=-1, keepdims=True)  # (5, 1)
heads = np.arange(768).reshape(12, 64)  # (12, 64)
mask = np.triu(np.ones((5, 5)), k=1).astype(bool)
scores = np.zeros((5, 5))
scores[mask] = -np.inf                  # hide future"""


# ---------------------------------------------------------------------------- helpers
def fmt(v):
    """Numbers as shown in cells: integers without '.0', unicode minus."""
    if isinstance(v, str):
        return v
    v = float(v)
    s = str(int(v)) if v == int(v) else f"{v:.2f}".rstrip("0")
    return s.replace("-", "−")


def num(s, color=WHITE, size=26):
    """Mono cell text; rendered larger and scaled down so the glyphs keep their spacing."""
    return Text(s, font=MONO, font_size=round(size * 1.2), color=color).scale(1 / 1.2)


def mono(s, size=26, color=WHITE):
    return Text(s, font=MONO, font_size=size, color=color)


def label(text, color=WHITE, font_size=30, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def caption(text):
    return Text(text, font_size=20, color=GREY_B).to_corner(DR, buff=0.3)


def glyphs(mob, s, start, end):
    """Glyphs of Text `mob` (built from string s) for s[start:end]; spaces have no glyphs."""
    a = len(s[:start].replace(" ", ""))
    return mob[a:a + len(s[start:end].replace(" ", ""))]


def rich(text, colors=(), font_size=32, color=WHITE, **kw):
    t = Text(text, font_size=font_size, color=color, **kw)
    for sub, col in colors:
        i = text.find(sub)
        glyphs(t, text, i, i + len(sub)).set_color(col)
    return t


class Grid(VGroup):
    """n x m cells (box + optional text), centred on the origin. boxes[i][j], texts[i][j]."""

    def __init__(self, values, cw=0.8, rh=0.55, px=None, py=None, color=BLUE, opacity=0.2, size=26,
                 text_color=WHITE, stroke=2):
        super().__init__()
        n, m = len(values), len(values[0])
        px, py = px or cw + 0.06, py or rh + 0.06
        self.px, self.py, self.n, self.m = px, py, n, m
        self.boxes, self.texts, self.rows = [], [], VGroup()
        x0, y0 = -(m - 1) * px / 2, (n - 1) * py / 2
        self.anchor = VectorizedPoint([x0, y0, 0])
        for i in range(n):
            row, brow, trow = VGroup(), [], []
            for j in range(m):
                box = Rectangle(width=cw, height=rh, stroke_color=color, stroke_width=stroke,
                                fill_color=color, fill_opacity=opacity).move_to([x0 + j * px, y0 - i * py, 0])
                cell = VGroup(box)
                txt = None
                if values[i][j] is not None:
                    txt = num(fmt(values[i][j]), text_color, size).move_to(box)
                    cell.add(txt)
                row.add(cell)
                brow.append(box)
                trow.append(txt)
            self.rows.add(row)
            self.boxes.append(brow)
            self.texts.append(trow)
        self.add(self.rows, self.anchor)

    def pos(self, i, j):
        return self.anchor.get_center() + np.array([j * self.px, -i * self.py, 0])

    def all_boxes(self):
        return [b for r in self.boxes for b in r]

    def all_texts(self):
        return [t for r in self.texts for t in r if t is not None]


def cell(value, color=BLUE, cw=0.8, rh=0.55, size=26, opacity=0.2):
    box = Rectangle(width=cw, height=rh, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=opacity)
    return VGroup(box, num(fmt(value), WHITE, size).move_to(box))


def code_icon(seed):
    """A tiny monokai editor window with coloured 'code' strokes."""
    win = RoundedRectangle(corner_radius=0.1, width=2.3, height=1.45, stroke_color=GREY_B, stroke_width=2,
                           fill_color="#272822", fill_opacity=1)
    dots = VGroup(*[Dot(radius=0.045, color=c) for c in (RED, YELLOW, GREEN)]).arrange(RIGHT, buff=0.07)
    dots.move_to(win.get_corner(UL) + np.array([0.28, -0.17, 0]))
    rng = random.Random(seed)
    palette = ["#F92672", "#66D9EF", "#A6E22E", "#E6DB74", "#F8F8F2", "#AE81FF"]
    strokes = VGroup()
    for k in range(4):
        y = win.get_top()[1] - 0.47 - k * 0.24
        x = win.get_left()[0] + 0.25 + (0.25 if k in (1, 2) and rng.random() < 0.6 else 0)
        for _ in range(rng.randint(2, 3)):
            w = rng.uniform(0.2, 0.55)
            if x + w > win.get_right()[0] - 0.2:
                break
            strokes.add(Line([x, y, 0], [x + w, y, 0], color=rng.choice(palette), stroke_width=5))
            x += w + 0.1
    return VGroup(win, dots, strokes)


# ---------------------------------------------------------------------------- section 10 card icons
def _curve(f, x0, x1, color, n=40):
    pts = [np.array([x, f(x), 0]) for x in np.linspace(x0, x1, n)]
    return VMobject(stroke_color=color, stroke_width=4).set_points_smoothly(pts)


def recap_icon(k):
    if k == 0:    # vectors
        return Arrow([-0.45, -0.28, 0], [0.45, 0.3, 0], buff=0, color=BLUE, stroke_width=5,
                     max_tip_length_to_length_ratio=0.3)
    if k == 1:    # dot product
        a = Arrow([-0.4, -0.25, 0], [0.5, -0.25, 0], buff=0, color=BLUE, stroke_width=5,
                  max_tip_length_to_length_ratio=0.28)
        b = Arrow([-0.4, -0.25, 0], [0.15, 0.3, 0], buff=0, color=TEAL, stroke_width=5,
                  max_tip_length_to_length_ratio=0.3)
        return VGroup(a, b)
    if k == 2:    # matrices
        return VGroup(*[Square(0.2, stroke_color=PURPLE_B, stroke_width=1.5, fill_color=PURPLE_B,
                               fill_opacity=0.5).move_to([(j - 1) * 0.24, (1 - i) * 0.2, 0])
                        for i in range(3) for j in range(3)])
    if k == 3:    # bends (ReLU)
        return VMobject(stroke_color=GREEN, stroke_width=5).set_points_as_corners(
            [[-0.5, -0.2, 0], [0, -0.2, 0], [0.45, 0.3, 0]])
    if k == 4:    # exp & log
        return _curve(lambda x: -0.3 + 0.6 * (np.exp(2.6 * (x + 0.5)) - 1) / (np.exp(2.6) - 1), -0.5, 0.5, YELLOW)
    if k == 5:    # probability
        hs = [0.2, 0.55, 0.35, 0.15]
        return VGroup(*[Rectangle(width=0.18, height=h, stroke_width=0, fill_color=TEAL, fill_opacity=0.9)
                        .move_to([(j - 1.5) * 0.25, -0.3 + h / 2, 0]) for j, h in enumerate(hs)])
    if k == 6:    # softmax
        return label("eˣ / Σ eˣ", ORANGE, 26)
    if k == 7:    # statistics (bell curve)
        return _curve(lambda x: -0.3 + 0.6 * np.exp(-(x / 0.2) ** 2), -0.55, 0.55, BLUE_B)
    if k == 8:    # waves
        return _curve(lambda x: 0.25 * np.sin(2 * PI * x / 0.55), -0.55, 0.55, TEAL)
    if k == 9:    # gradients
        c = _curve(lambda x: -0.3 + 1.6 * x ** 2, -0.5, 0.5, RED_B)
        d = Dot([0.35, -0.3 + 1.6 * 0.35 ** 2, 0], radius=0.06, color=YELLOW)
        a = Arrow(d.get_center(), [0.12, -0.3 + 1.6 * 0.12 ** 2, 0], buff=0.06, color=YELLOW, stroke_width=4,
                  max_tip_length_to_length_ratio=0.5)
        return VGroup(c, a, d)
    return mono("np", 30, YELLOW)    # NumPy


RECAP = [("F1", "vectors", BLUE), ("F2", "dot product", TEAL), ("F3", "matrices", PURPLE_B),
         ("F4", "bends", GREEN), ("F5", "exp & log", YELLOW), ("F6", "probability", TEAL),
         ("F7", "softmax", ORANGE), ("F8", "statistics", BLUE_B), ("F9", "waves", TEAL),
         ("F10", "gradients", RED_B), ("F11", "NumPy", YELLOW)]


def recap_card(k):
    tag, name, color = RECAP[k]
    box = RoundedRectangle(corner_radius=0.14, width=2.0, height=1.55, stroke_color=color, stroke_width=3,
                           fill_color=color, fill_opacity=0.12)
    t = label(tag, GREY_B, 20).move_to(box.get_corner(UL) + np.array([0.3, -0.2, 0]), aligned_edge=LEFT)
    t.align_to(box.get_left() + np.array([0.15, 0, 0]), LEFT)
    icon = recap_icon(k).move_to(box.get_center() + np.array([0, 0.12, 0]))
    name_t = label(name, WHITE, 21).move_to(box.get_center() + np.array([0, -0.5, 0]))
    shade = Rectangle(width=2.1, height=1.65, stroke_width=0, fill_color=BLACK, fill_opacity=0.72)
    shade.move_to(box).set_z_index(3)
    return VGroup(box, t, icon, name_t), shade


class NumPyVideo(VoicedScene):
    VIDEO = "f11"

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
        self.s1_numpy()
        self.s2_arrays()
        self.s3_elementwise()
        self.s4_matmul()
        self.s5_axis()
        self.s6_broadcast()
        self.s7_reshape()
        self.s8_mask()
        self.s9_code()
        self.s10_recap()
        finish(self)

    # 1. NumPy ------------------------------------------------------------------------------------------
    def s1_numpy(self):
        self.section(1)
        title = Text("NumPy", font_size=96)
        title[:3].set_color(BLUE_C)
        title[3:].set_color(YELLOW)
        title.move_to([0, 2.25, 0])
        vals = np.arange(12).reshape(3, 4)
        grid = Grid(vals.tolist(), cw=0.7, rh=0.55, px=0.78, py=0.63, size=28).move_to([-3.0, -0.75, 0])
        gcap = mono("np.arange(12).reshape(3, 4)", 22, GREY_B).next_to(grid, DOWN, buff=0.4)
        self.at("numpy")
        self.play(FadeIn(title, shift=0.2 * DOWN), FadeIn(grid, lag_ratio=0.05), run_time=0.8)
        sub = label("Python's library for fast math on arrays of numbers", GREY_B, 30).move_to([0, 1.05, 0])
        self.at("pythons")
        self.play(FadeIn(sub, shift=0.15 * UP), FadeIn(gcap), run_time=0.5)
        self.at("arrays")
        self.play(LaggedStart(*[b.animate(rate_func=there_and_back).set_fill(YELLOW, 0.6)
                                for b in grid.all_boxes()], lag_ratio=0.08), run_time=1.0)
        icons = VGroup(*[code_icon(s) for s in (4, 7, 9)])
        for k, ic in enumerate(icons):
            ic.move_to([3.0 + (k - 1) * 0.35, -0.6 - (k - 1) * 0.28, 0])
        icap = label("code in episodes 3–13", GREY_B, 22).move_to([3.0, -2.2, 0])
        self.at("snippet")
        self.play(LaggedStart(*[FadeIn(ic, shift=0.2 * UP) for ic in icons], lag_ratio=0.3), FadeIn(icap),
                  run_time=0.6)
        self.end_section()

    # 2. Arrays and shapes ----------------------------------------------------------------------------
    def s2_arrays(self):
        self.section(2)
        head = label("an array: numbers in a grid", GREY_B, 30).move_to([0, 3.3, 0])
        self.play(*self.clear_anims(), run_time=0.4)
        self.play(FadeIn(head), run_time=0.4)

        kw = dict(cw=1.0, rh=0.55, px=1.06, py=0.65, size=24)
        mat = Grid(X2.tolist(), **kw).move_to([-0.6, -0.3, 0])
        vec = Grid([V2.tolist()], color=TEAL, **kw)
        vec.move_to([mat.get_x(), 2.3, 0])
        lx = -5.3
        vlab = VGroup(label("vector", WHITE, 30), label("1-D", GREY_B, 22)).arrange(DOWN, buff=0.1)
        vlab.move_to([lx, vec.get_y(), 0])
        self.at("one")
        self.play(FadeIn(vec, shift=0.15 * RIGHT), FadeIn(vlab), run_time=0.5)

        toks = VGroup(*[label(w, BLUE_B, 26).move_to([mat.get_left()[0] - 0.3, mat.pos(i, 0)[1], 0],
                                                      aligned_edge=RIGHT) for i, w in enumerate(WORDS)])
        mlab = VGroup(label("matrix", WHITE, 30), label("2-D", GREY_B, 22)).arrange(DOWN, buff=0.1)
        mlab.move_to([lx, mat.get_y(), 0])
        note = caption("illustrative values")
        self.at("two")
        self.play(FadeIn(mat, lag_ratio=0.03), FadeIn(toks), FadeIn(mlab), FadeIn(note), run_time=0.7)

        shape = rich("shape (5, 4)", [("(5, 4)", YELLOW)], font_size=34).move_to([mat.get_x(), -2.55, 0])
        self.at("shape")
        self.play(FadeIn(shape, shift=0.15 * UP), run_time=0.5)
        self.at("five")
        self.play(Circumscribe(toks, color=YELLOW, buff=0.1), run_time=0.5)
        self.at("four")
        self.play(Circumscribe(mat.rows[0], color=YELLOW, buff=0.08), run_time=0.5)
        brace = Brace(mat.rows, RIGHT, buff=0.15, color=GREY_B)
        blab = label("5 rows of 4 numbers", WHITE, 28).next_to(brace, RIGHT, buff=0.2)
        self.at("rows")
        self.play(GrowFromCenter(brace), FadeIn(blab, shift=0.1 * RIGHT), run_time=0.5)
        self.end_section()

    # 3. Element-wise arithmetic ----------------------------------------------------------------------
    def s3_elementwise(self):
        self.section(3)
        self.play(*self.clear_anims(), run_time=0.4)
        kw = dict(cw=0.75, rh=0.6, px=0.8, py=0.66, size=28)
        ty, by = 1.25, -1.75
        a = Grid(A3.tolist(), **kw).move_to([-3.6, ty, 0])
        b = Grid(B3.tolist(), color=TEAL, **kw).move_to([0, ty, 0])
        c = Grid((A3 + B3).tolist(), color=YELLOW, opacity=0.08, **kw).move_to([3.6, ty, 0])
        for t in c.all_texts():
            t.set_opacity(0)
        plus = label("+", WHITE, 48).move_to([-1.8, ty, 0])
        eq = label("=", WHITE, 48).move_to([1.8, ty, 0])
        names = VGroup(label("A", WHITE, 30).next_to(a, UP, buff=0.25), label("B", TEAL, 30).next_to(b, UP, buff=0.25))
        cname = label("A + B", YELLOW, 30).next_to(c, UP, buff=0.25)
        head = label("element-wise", GREY_B, 30).move_to([0, 3.3, 0])
        self.play(FadeIn(a), FadeIn(b), FadeIn(plus), FadeIn(names), run_time=0.5)
        self.at("element")
        self.play(FadeIn(head), LaggedStart(*[bx.animate(rate_func=there_and_back).set_fill(YELLOW, 0.5)
                                              for bx in a.all_boxes() + b.all_boxes()], lag_ratio=0.05),
                  run_time=0.7)
        self.at("add")
        self.play(FadeIn(eq), FadeIn(c), FadeIn(cname), run_time=0.4)

        steps = []
        for i in range(2):
            for j in range(3):
                steps.append(AnimationGroup(
                    a.boxes[i][j].animate(rate_func=there_and_back).set_stroke(YELLOW, 5),
                    b.boxes[i][j].animate(rate_func=there_and_back).set_stroke(YELLOW, 5),
                    c.boxes[i][j].animate.set_fill(YELLOW, 0.3),
                    c.texts[i][j].animate.set_opacity(1)))
        self.at("matching")
        self.play(LaggedStart(*steps, lag_ratio=0.35), run_time=1.6)

        a2 = Grid(A3.tolist(), **kw).move_to([-3.6, by, 0])
        d = Grid((A3 * 2).tolist(), color=YELLOW, opacity=0.08, **kw).move_to([3.6, by, 0])
        for t in d.all_texts():
            t.set_opacity(0)
        times = VGroup(label("×", WHITE, 48).move_to([-1.8, by, 0]), label("2", YELLOW, 56).move_to([0, by, 0]))
        eq2 = label("=", WHITE, 48).move_to([1.8, by, 0])
        a2name = label("A", WHITE, 30).next_to(a2, UP, buff=0.25)
        dname = label("A × 2", YELLOW, 30).next_to(d, UP, buff=0.25)
        self.at("multiply")
        self.play(FadeIn(a2), FadeIn(a2name), FadeIn(times), FadeIn(eq2), FadeIn(d), FadeIn(dname), run_time=0.5)
        self.at("doubles")
        self.play(LaggedStart(*[AnimationGroup(d.boxes[i][j].animate.set_fill(YELLOW, 0.3),
                                               d.texts[i][j].animate.set_opacity(1))
                                for i in range(2) for j in range(3)], lag_ratio=0.15), run_time=0.6)
        self.end_section()

    # 4. The @ sign ------------------------------------------------------------------------------------
    def s4_matmul(self):
        self.section(4)
        at_sign = label("@", YELLOW, 84).move_to([-2.9, 2.95, 0])
        self.at("at")
        self.play(*self.clear_anims(), FadeIn(at_sign, scale=0.7), run_time=0.5)
        mm = label("=  matrix multiplication", WHITE, 36).next_to(at_sign, RIGHT, buff=0.35)
        self.at("multiplication")
        self.play(FadeIn(mm, shift=0.1 * RIGHT), run_time=0.4)

        kw = dict(cw=0.42, rh=0.42, px=0.48, py=0.48, stroke=1.5)
        cy = 0.2
        x = Grid([[None] * 4] * 5, color=BLUE, opacity=0.35, **kw).move_to([-3.8, cy, 0])
        w = Grid([[None] * 4] * 4, color=PURPLE_B, opacity=0.35, **kw).move_to([-0.4, cy, 0])
        r = Grid([[None] * 4] * 5, color=GREY_B, opacity=0.08, **kw).move_to([3.0, cy, 0])
        op = label("@", WHITE, 40).move_to([-2.1, cy, 0])
        eq = label("=", WHITE, 40).move_to([1.3, cy, 0])
        ny = x.get_top()[1] + 0.35
        names = VGroup(label("X", BLUE_B, 30).move_to([x.get_x(), ny, 0]),
                       label("W", PURPLE_B, 30).move_to([w.get_x(), ny, 0]),
                       label("X @ W", WHITE, 30).move_to([r.get_x(), ny, 0]))
        self.at("dot")
        self.play(FadeIn(x), FadeIn(w), FadeIn(r), FadeIn(op), FadeIn(eq), FadeIn(names), run_time=0.6)
        rband = SurroundingRectangle(x.rows[0], color=YELLOW, buff=0.05, stroke_width=4)
        cband = SurroundingRectangle(VGroup(*[w.rows[i][0] for i in range(4)]), color=TEAL, buff=0.05, stroke_width=4)
        self.at("rows")
        self.play(Create(rband), run_time=0.3)
        self.at("columns")
        self.play(Create(cband), run_time=0.3)
        one = rich("one row · one column → one cell", [("one row", YELLOW), ("one column", TEAL)], font_size=28)
        one.move_to([0, -1.55, 0])
        self.play(r.boxes[0][0].animate.set_fill(YELLOW, 0.9).set_stroke(YELLOW), FadeIn(one), run_time=0.4)

        fy = -2.6
        parts = [label(s, WHITE, 40) for s in ("(5, 4)", "@", "(4, 4)", "→", "(5, 4)")]
        formula = VGroup(*parts).arrange(RIGHT, buff=0.35).move_to([0, fy, 0])
        parts[0][3].set_color(GREEN)      # inner sizes must match
        parts[2][1].set_color(GREEN)
        self.at("five")
        self.play(FadeIn(parts[0], shift=0.1 * UP), run_time=0.3)
        self.at("at")
        self.play(FadeIn(parts[1], shift=0.1 * UP), run_time=0.25)
        self.at("four")
        self.play(FadeIn(parts[2], shift=0.1 * UP), run_time=0.3)
        self.at("gives")
        self.play(FadeIn(parts[3], shift=0.1 * UP), FadeIn(parts[4], shift=0.1 * UP), FadeOut(rband),
                  FadeOut(cband), *[bx.animate.set_fill(YELLOW, 0.45).set_stroke(YELLOW) for bx in r.all_boxes()],
                  run_time=0.5)
        self.end_section()

    # 5. axis and keepdims --------------------------------------------------------------------------
    def s5_axis(self):
        self.section(5)
        self.play(*self.clear_anims(), run_time=0.4)
        grid = Grid(Y5.tolist(), cw=0.8, rh=0.52, px=0.85, py=0.62, size=26).move_to([-3.4, 0.3, 0])
        ylab = label("Y", WHITE, 34).next_to(grid, LEFT, buff=0.35)
        note = caption("illustrative values")
        self.play(FadeIn(grid), FadeIn(ylab), FadeIn(note), run_time=0.5)

        ay = grid.get_top()[1] + 0.35
        arrow = Arrow([grid.get_left()[0] + 0.05, ay, 0], [grid.get_right()[0] - 0.05, ay, 0], buff=0,
                      color=GREY_B, stroke_width=5, max_tip_length_to_length_ratio=0.08)
        self.at("axis")
        self.play(GrowArrow(arrow), run_time=0.4)
        code1 = mono("Y.sum(axis=-1)", 30).move_to([0, 3.25, 0])
        self.at("summing")
        self.play(FadeIn(code1), run_time=0.4)
        alab = label("axis = −1", YELLOW, 30).next_to(arrow, UP, buff=0.12)
        alab2 = label("(the last axis)", GREY_B, 22).next_to(alab, RIGHT, buff=0.2)
        self.at("one")
        self.play(arrow.animate.set_color(YELLOW), FadeIn(alab), FadeIn(alab2), run_time=0.4)

        bands = VGroup(*[Rectangle(width=grid.rows[i].width + 0.1, height=0.58, stroke_width=0, fill_color=YELLOW,
                                   fill_opacity=0.22).move_to(grid.rows[i]) for i in range(5)])
        bands.set_z_index(-1)
        self.at("along")
        self.play(*[GrowFromEdge(bd, LEFT) for bd in bands], run_time=0.8)

        ry = -2.5
        tot = VGroup(*[cell(v, YELLOW, 0.8, 0.52, 26).move_to([grid.get_x() + (k - 2) * 0.85, ry, 0])
                       for k, v in enumerate(T5)])
        s1 = rich("shape (5,)", [("(5,)", YELLOW)], font_size=30).next_to(tot, RIGHT, buff=0.45)
        self.at("total")
        self.play(LaggedStart(*[AnimationGroup(ReplacementTransform(bands[i], tot[i][0]),
                                               FadeIn(tot[i][1], rate_func=lambda t: smooth(max(0.0, 2 * t - 1))))
                                for i in range(5)], lag_ratio=0.12),
                  *[bx.animate.set_stroke(opacity=0.5).set_fill(opacity=0.1) for bx in grid.all_boxes()],
                  *[t.animate.set_opacity(0.55) for t in grid.all_texts()], run_time=0.9)
        self.remove(*[m for t in tot for m in t])
        self.add(tot)
        self.play(FadeIn(s1), run_time=0.3)

        code2 = mono("Y.sum(axis=-1, keepdims=True)", 30).move_to(code1)
        glyphs(code2, "Y.sum(axis=-1, keepdims=True)", 15, 28).set_color(YELLOW)
        self.at("keep")
        self.play(FadeTransform(code1, code2), run_time=0.4)
        cx = grid.get_right()[0] + 0.95
        s2 = rich("shape (5, 1)", [("(5, 1)", YELLOW)], font_size=30).move_to([cx + 0.75, grid.get_y(), 0],
                                                                              aligned_edge=LEFT)
        self.at("true")
        self.play(*[t.animate.move_to([cx, grid.pos(i, 0)[1], 0]) for i, t in enumerate(tot)],
                  FadeTransform(s1, s2), run_time=0.8)
        self.at("column")
        self.play(Circumscribe(tot, color=YELLOW, buff=0.08), run_time=0.45)
        self.s5 = dict(grid=grid, tot=tot)
        self.end_section()

    # 6. Broadcasting ---------------------------------------------------------------------------------
    def s6_broadcast(self):
        self.section(6)
        g5, tot = self.s5["grid"], self.s5["tot"]
        guides = VGroup(*[DashedLine([g5.get_left()[0] - 0.1, tot[i].get_y(), 0],
                                     [tot[i].get_right()[0] + 0.1, tot[i].get_y(), 0],
                                     color=YELLOW, stroke_width=2, dash_length=0.1) for i in range(5)])
        guides.set_z_index(-1)
        self.at("lines")
        self.play(LaggedStart(*[Create(gd) for gd in guides], lag_ratio=0.12), run_time=0.6)

        kw = dict(cw=0.95, rh=0.52, px=1.0, py=0.6, size=24)
        cy = 0.2
        grid = Grid(E6.tolist(), **kw).move_to([-3.6, cy, 0])
        col = Grid(S6.tolist(), color=YELLOW, **kw).move_to([0.2, cy, 0])
        div = label("÷", WHITE, 48).move_to([-0.55, cy, 0])
        head = label("broadcasting", YELLOW, 36).move_to([0, 3.15, 0])
        clab = label("row totals", GREY_B, 22).next_to(col, UP, buff=0.2)
        note = caption("illustrative values")
        self.at("broadcasting")
        self.play(*self.clear_anims(), run_time=0.35)
        self.play(FadeIn(head), FadeIn(grid), FadeIn(col), FadeIn(clab), FadeIn(note), run_time=0.5)
        self.at("divide")
        self.play(FadeIn(div, scale=0.7), run_time=0.3)

        ly = grid.get_bottom()[1] - 0.35
        sh1 = label("(5, 5)", GREY_A, 28).move_to([grid.get_x(), ly, 0])
        sh2 = label("(5, 1)", GREY_A, 28).move_to([col.get_x(), ly, 0])
        self.at("5")
        self.at("5")
        self.play(FadeIn(sh1), run_time=0.3)
        self.at("1")
        self.play(FadeIn(sh2), run_time=0.3)

        ghosts = VGroup()
        for k in range(1, 5):
            gcol = col.copy()
            for bx in [r[0] for r in gcol.boxes]:
                bx.set_stroke(opacity=0.45).set_fill(opacity=0.06)
            for t in gcol.all_texts():
                t.set_opacity(0.45)
            ghosts.add(gcol.shift(k * col.px * RIGHT))
        self.at("column")
        self.play(LaggedStart(*[FadeIn(g, shift=0.5 * col.px * RIGHT) for g in ghosts], lag_ratio=0.2), run_time=0.7)

        new_texts = [[num(fmt(P6[i, j]), WHITE, 24).move_to(grid.boxes[i][j]) for j in range(5)] for i in range(5)]
        self.at("divided")
        self.play(LaggedStart(*[AnimationGroup(
            *[FadeOut(grid.texts[i][j], shift=0.1 * UP) for j in range(5)],
            *[FadeIn(new_texts[i][j], shift=0.1 * UP) for j in range(5)],
            col.boxes[i][0].animate(rate_func=there_and_back).set_fill(YELLOW, 0.7),
            *[grid.boxes[i][j].animate(rate_func=there_and_back).set_stroke(YELLOW, 4) for j in range(5)])
            for i in range(5)], lag_ratio=0.25), run_time=1.1)
        self.at("number")
        self.play(FadeOut(ghosts), run_time=0.4)

        cap = rich("how softmax normalizes each row", [("softmax", YELLOW)], font_size=32)
        cap.move_to([-1.7, -2.55, 0])
        self.at("softmax")
        self.play(FadeIn(cap, shift=0.15 * UP), run_time=0.5)
        ok = label("every row now sums to 1  ✓", GREEN, 26).move_to([-1.7, -3.2, 0])
        self.at("every")
        self.play(FadeIn(ok, shift=0.1 * UP), run_time=0.4)
        self.end_section()

    # 7. Reshape and transpose ------------------------------------------------------------------------
    def s7_reshape(self):
        self.section(7)
        # small example: 6 numbers → (2, 3) → transposed (3, 2)
        side, x0, y0 = 0.6, -5.7, 2.85
        demo = VGroup(*[cell(k, TEAL, 0.54, 0.54, 26).move_to([x0 + k * side, y0, 0]) for k in range(6)])
        ctext = ["np.arange(6)", "np.arange(6).reshape(2, 3)", "np.arange(6).reshape(2, 3).T"]
        stext = ["shape (6,)", "shape (2, 3)", "shape (2, 3) → (3, 2)"]
        cx = -1.6

        def code_lab(k):
            return mono(ctext[k], 26).move_to([cx, 2.95, 0], aligned_edge=LEFT)

        def shape_lab(k):
            return rich(stext[k], [(stext[k][6:], YELLOW)], font_size=26).move_to([cx, 2.35, 0], aligned_edge=LEFT)

        c0, s0 = code_lab(0), shape_lab(0)
        self.play(*self.clear_anims(), run_time=0.35)
        self.play(FadeIn(demo, lag_ratio=0.1), FadeIn(c0), FadeIn(s0), run_time=0.5)
        c1, s1 = code_lab(1), shape_lab(1)
        self.at("new")
        self.play(*[demo[k].animate.move_to([x0 + (k % 3) * side, y0 - (k // 3) * side, 0]) for k in range(6)],
                  FadeTransform(c0, c1), FadeTransform(s0, s1), run_time=0.7)

        # big example: 768 numbers → 12 heads of 64
        pitch, sw, sy = 0.15, 12.288 / 768, 0.55
        colors = color_gradient([BLUE_D, TEAL_C, GREEN_C, GOLD_C], 12)
        strip = VGroup(*[VGroup(*[Rectangle(width=sw, height=0.32, stroke_width=0,
                                            fill_color=colors[r] if c % 2 == 0 else interpolate_color(colors[r], BLACK, 0.3),
                                            fill_opacity=0.95)
                                  .move_to([-6.144 + (64 * r + c + 0.5) * sw, sy, 0]) for c in range(64)])
                         for r in range(12)])
        slab = label("768 numbers", WHITE, 26).move_to([6.144, sy + 0.45, 0], aligned_edge=RIGHT)
        self.at("768")
        self.play(FadeIn(strip), FadeIn(slab), run_time=0.5)

        gx, gy = 0.8, -1.55
        targets = VGroup(*[VGroup(*[Rectangle(width=pitch - 0.02, height=pitch - 0.02, stroke_width=0,
                                              fill_color=colors[r] if c % 2 == 0 else interpolate_color(colors[r], BLACK, 0.3),
                                              fill_opacity=0.95)
                                    .move_to([gx + (c - 31.5) * pitch, gy + (5.5 - r) * pitch, 0]) for c in range(64)])
                           for r in range(12)])
        code = mono("np.arange(768).reshape(12, 64)", 26, GREY_A).move_to([gx, -3.05, 0])
        self.at("become")
        self.play(LaggedStart(*[Transform(strip[r], targets[r]) for r in range(12)], lag_ratio=0.06),
                  FadeOut(slab), FadeIn(code), run_time=0.9)
        lbrace = Brace(targets, LEFT, buff=0.12, color=GREY_B)
        llab = label("12 heads", WHITE, 28).next_to(lbrace, LEFT, buff=0.15)
        tbrace = Brace(targets, UP, buff=0.1, color=GREY_B)
        tlab = label("64 numbers each", WHITE, 26).next_to(tbrace, UP, buff=0.1)
        self.at("heads")
        self.play(GrowFromCenter(lbrace), FadeIn(llab), run_time=0.3)
        self.at("64")
        self.play(GrowFromCenter(tbrace), FadeIn(tlab), run_time=0.3)

        c2, s2 = code_lab(2), shape_lab(2)
        self.at("transpose")
        self.play(*[demo[k].animate.move_to([x0 + (k // 3) * side, y0 - (k % 3) * side, 0]) for k in range(6)],
                  FadeTransform(c1, c2), FadeTransform(s1, s2), run_time=0.8)
        swap = label("rows ↔ columns", GREY_B, 24).move_to([cx, 1.75, 0], aligned_edge=LEFT)
        self.at("axes")
        self.play(FadeIn(swap, shift=0.1 * UP), run_time=0.4)
        self.end_section()

    # 8. Masks -----------------------------------------------------------------------------------------
    def s8_mask(self):
        self.section(8)
        self.play(*self.clear_anims(), run_time=0.4)
        grid = Grid(np.zeros((5, 5)).tolist(), cw=0.7, rh=0.54, px=0.76, py=0.6, color=BLUE, size=26)
        grid.move_to([-3.4, 0.0, 0])
        rows = VGroup(*[label(w, BLUE_B, 24).move_to([grid.get_left()[0] - 0.2, grid.pos(i, 0)[1], 0],
                                                      aligned_edge=RIGHT) for i, w in enumerate(WORDS)])
        cols = VGroup(*[label(w, BLUE_B, 24).move_to([grid.pos(0, j)[0], grid.get_top()[1] + 0.3, 0])
                        for j, w in enumerate(WORDS)])
        lx = -0.9
        l1 = mono("scores = np.zeros((5, 5))", 24).move_to([lx, 1.35, 0], aligned_edge=LEFT)
        self.at("masks")
        self.play(FadeIn(grid, lag_ratio=0.02), FadeIn(rows), FadeIn(cols), FadeIn(l1), run_time=0.6)

        l2 = mono("np.triu(np.ones((5, 5)), k=1)", 24).move_to([lx, 0.45, 0], aligned_edge=LEFT)
        l2n = label("→ 1s above the diagonal: the mask", GREY_B, 22).next_to(l2, DOWN, buff=0.15, aligned_edge=LEFT)
        self.at("upper")
        self.play(FadeIn(l2), run_time=0.4)
        upper = [(i, j) for i in range(5) for j in range(5) if MASK[i, j]]
        self.at("triangle")
        self.at("triangle")
        self.play(LaggedStart(*[grid.boxes[i][j].animate.set_fill(YELLOW, 0.4).set_stroke(YELLOW)
                                for i, j in upper], lag_ratio=0.08), FadeIn(l2n), run_time=0.8)
        self.at("pick")
        self.play(Circumscribe(VGroup(*[grid.boxes[i][j] for i, j in upper]), color=YELLOW, buff=0.06),
                  run_time=0.6)
        l3 = mono("scores[mask] = -np.inf", 24).move_to([lx, -0.75, 0], aligned_edge=LEFT)
        self.at("set")
        self.play(FadeIn(l3), run_time=0.4)
        infs = [num("−∞", RED, 26).move_to(grid.boxes[i][j]) for i, j in upper]
        self.at("infinity")
        self.play(*[FadeTransform(grid.texts[i][j], t) for (i, j), t in zip(upper, infs)],
                  *[grid.boxes[i][j].animate.set_fill(RED, 0.3).set_stroke(RED) for i, j in upper], run_time=0.6)

        cap = label("the causal mask", WHITE, 34)
        ep = label("(episode 6)", YELLOW, 34)
        VGroup(cap, ep).arrange(RIGHT, buff=0.25).move_to([0, -2.65, 0])
        self.at("causal")
        self.play(FadeIn(cap, shift=0.15 * UP), run_time=0.4)
        self.at("6")
        self.play(FadeIn(ep, shift=0.15 * UP), run_time=0.4)
        self.end_section()

    # 9. Code ------------------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, _ = code_panel(CODE, 26)
        code.move_to([-0.8, 0.1, 0])
        row_h = abs(code.line_numbers[0].get_y() - code.line_numbers[1].get_y())

        def hl_rect(a, b):
            top = code.line_numbers[a].get_y() + row_h / 2
            bot = code.line_numbers[b].get_y() - row_h / 2
            r = Rectangle(width=code.code_lines.width + 0.25, height=top - bot, color=YELLOW, stroke_width=2)
            r.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).set_y((top + bot) / 2)
            return r.set_z_index(1)

        code.code_lines.set_z_index(2)
        bx = code.get_right()[0] + 0.15

        def badge(text, y):
            t = label(text, GREY_B, 22)
            box = RoundedRectangle(corner_radius=0.1, width=t.width + 0.25, height=row_h - 0.06, stroke_color=GREY_D,
                                   stroke_width=2, fill_opacity=0)
            return VGroup(box, t.move_to(box)).move_to([bx, y, 0], aligned_edge=LEFT)

        ln = code.line_numbers
        b2, b3, b4 = badge("Ep 6", ln[2].get_y()), badge("Ep 6·10", ln[3].get_y()), badge("Ep 7", ln[4].get_y())
        span = Line([bx, ln[5].get_y() + 0.15, 0], [bx, ln[7].get_y() - 0.15, 0], color=GREY_D, stroke_width=3)
        b5 = badge("Ep 6", ln[6].get_y()).shift(0.18 * RIGHT)
        badges = VGroup(b2, b3, b4, span, b5)
        self.play(*self.clear_anims(), run_time=0.35)
        self.play(FadeIn(code, shift=0.2 * UP), FadeIn(badges), run_time=0.5)

        hl = hl_rect(0, 1)
        self.at("shapes")
        self.play(Create(hl), run_time=0.3)
        prev = None
        for cue, (a, b), bd in [("at", (2, 2), b2), ("sum", (3, 3), b3), ("reshape", (4, 4), b4),
                                ("mask", (5, 7), b5)]:
            self.at(cue)
            anims = [Transform(hl, hl_rect(a, b)), bd[1].animate.set_color(YELLOW)]
            if prev is not None:
                anims.append(prev[1].animate.set_color(GREY_B))
            self.play(*anims, run_time=0.3)
            prev = bd
        self.end_section()

    # 10. The toolkit ----------------------------------------------------------------------------------
    def s10_recap(self):
        self.section(10)
        self.play(*self.clear_anims(), run_time=0.3)
        head = label("the toolkit", GREY_B, 34).move_to([0, 3.1, 0])
        cards, shades = VGroup(), VGroup()
        for k in range(11):
            c, s = recap_card(k)
            pos = [-5.25 + 2.1 * k, 1.35, 0] if k < 6 else [-4.2 + 2.1 * (k - 6), -0.45, 0]
            c.move_to(pos)
            s.move_to(c[0])
            cards.add(c)
            shades.add(s)
        self.at("toolkit")
        self.play(FadeIn(head), FadeIn(cards), FadeIn(shades), run_time=0.5)
        cues = ["vectors", "dot", "matrices", "bends", "exponentials", "probability", "softmax", "statistics",
                "waves", "gradients", "numpy"]
        for k, cue in enumerate(cues):
            self.at(cue)
            self.play(FadeOut(shades[k]), cards[k][0].animate.set_stroke(width=5).set_fill(opacity=0.22),
                      run_time=0.3 if k < 10 else 0.4)
        ready = rich("you're ready for the main series", [("ready", YELLOW)], font_size=36).move_to([0, -2.45, 0])
        self.at("ready")
        self.play(FadeIn(ready, shift=0.15 * UP), run_time=0.5)
        nxt = next_up_card(NEXT)
        self.at("llm")
        self.play(*self.clear_anims(), FadeIn(nxt, shift=0.2 * UP), run_time=0.6)
        self.end_section()
