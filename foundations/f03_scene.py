"""Foundations F3 — Matrices: Many Dot Products at Once.

Render from the repo root:  ./render.sh foundations f03
"""
import numpy as np
from manim import *

from common import code_panel, finish, highlight, next_up_card, token, used_in_card
from f03_script import LABEL, NEXT, TAGLINE, TITLE, USED_IN
from intro import play_token_intro
from voiced_scene import VoicedScene

MONO = "DejaVu Sans Mono"
WORDS = ["The", "cat", "sat", "on", "the"]
ROW_C, COL_C = YELLOW, TEAL          # row highlight / column and vector colour
W_COLOR = PURPLE_B                   # weights belong to the model

# ---------------------------------------------------------------------------- worked examples
M = np.array([[1, 0, 2],
              [0, 1, -1]])
XV = np.array([3, 1, 2])
MX = M @ XV
assert list(MX) == [7, -1]

STRETCH = np.array([[2, 0], [0, 1]])
ROTATE = np.array([[0, -1], [1, 0]])
SHEAR = np.array([[1, 1], [0, 1]])

# section 6: illustrative integer matrices, product computed exactly
X6 = np.array([[1, -2, 1, 1],
               [2, -1, 0, 2],
               [-1, -2, -2, 1],
               [1, 2, -2, 1],
               [1, 2, 2, 1]])
W6 = np.array([[1, -1, 2, -1],
               [-2, 0, -2, 1],
               [-1, -2, -1, 1],
               [2, 2, 0, 1]])
R6 = X6 @ W6
assert R6[0, 0] == 1 * 1 + (-2) * (-2) + 1 * (-1) + 1 * 2 == 6

# section 8: illustrative query-key score intensities
_rng = np.random.default_rng(8)
SCORE_LIGHT = _rng.uniform(0.15, 0.85, (5, 5))

CODE = """M = np.array([[1, 0, 2],
              [0, 1, -1]])          # 2 x 3
x = np.array([3, 1, 2])
M @ x                               # array([ 7, -1])
X = np.random.randn(5, 4)           # 5 tokens, 4 numbers each
W = np.random.randn(4, 4)
(X @ W).shape                       # (5, 4)
M.T.shape                           # (3, 2)"""


# ---------------------------------------------------------------------------- helpers
def fmt(v):
    return str(int(v)).replace("-", "−")


def num(s, color=WHITE, size=36):
    """Matrix cell text; rendered larger and scaled down so the mono glyphs keep their spacing."""
    return Text(s, font=MONO, font_size=round(size * 1.2), color=color).scale(1 / 1.2)


def label(text, color=GREY_B, font_size=26, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def caption(text):
    return Text(text, font_size=20, color=GREY_B).to_corner(DR, buff=0.3)


def glyphs(mob, s, start, end):
    """Glyphs of Text `mob` (built from string s) for s[start:end]; spaces have no glyphs."""
    a = len(s[:start].replace(" ", ""))
    return mob[a:a + len(s[start:end].replace(" ", ""))]


def bracket_pair(left, right, top, bottom, color=GREY_B, d=0.12, width=2.5):
    def one(x, dd):
        pts = [[x + dd, top, 0], [x, top, 0], [x, bottom, 0], [x + dd, bottom, 0]]
        return VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(pts)
    return VGroup(one(left, d), one(right, -d))


def brackets_around(mob, color=GREY_B, padx=0.12, pady=0.1, d=0.12):
    return bracket_pair(mob.get_left()[0] - padx, mob.get_right()[0] + padx, mob.get_top()[1] + pady,
                        mob.get_bottom()[1] - pady, color, d)


def cell_grid(n, m, px, py, side, color, opacity=0.45, stroke=1.5):
    """n x m grid of filled squares, rows first: grid[i][j]."""
    g = VGroup()
    for i in range(n):
        row = VGroup()
        for j in range(m):
            row.add(Square(side, stroke_width=stroke, stroke_color=color, fill_color=color,
                           fill_opacity=opacity).move_to([j * px, -i * py, 0]))
        g.add(row)
    return g.move_to(ORIGIN)


class Mat(VGroup):
    """A matrix of Text cells with bracket lines, centred on the origin. cells[i][j], right-aligned per column."""

    def __init__(self, values, cw=0.9, rh=0.75, size=36, colors=None, bracket_color=GREY_B, pad=0.18):
        super().__init__()
        values = np.asarray(values)
        n, m = values.shape
        self.cw, self.rh = cw, rh
        cols = colors if colors is not None else [[WHITE] * m for _ in range(n)]
        texts = [[num(fmt(values[i, j]), cols[i][j], size) for j in range(m)] for i in range(n)]
        self.hw = max(t.width for r in texts for t in r) / 2
        x0, y0 = -(m - 1) * cw / 2, (n - 1) * rh / 2
        self.anchor = VectorizedPoint([x0, y0, 0])
        self.cells = VGroup()
        for i in range(n):
            row = VGroup()
            for j in range(m):
                t = texts[i][j]
                t.move_to([x0 + j * cw + self.hw - t.width / 2, y0 - i * rh, 0])
                row.add(t)
            self.cells.add(row)
        half = max(0.3, rh / 2)
        self.brackets = bracket_pair(x0 - self.hw - pad, -x0 + self.hw + pad, y0 + half, -y0 - half, bracket_color)
        self.add(self.brackets, self.cells, self.anchor)

    def cx(self, j):
        return self.anchor.get_x() + j * self.cw

    def ry(self, i):
        return self.anchor.get_y() - i * self.rh

    def row_band(self, i, color, opacity=0.14):
        left, right = self.brackets[0].get_left()[0] + 0.07, self.brackets[1].get_right()[0] - 0.07
        return Rectangle(width=right - left, height=self.rh - 0.06, stroke_color=color, stroke_width=3,
                         fill_color=color, fill_opacity=opacity).move_to(
            [(left + right) / 2, self.ry(i), 0]).set_z_index(-1)

    def col_band(self, j, color, opacity=0.14):
        t = self.brackets.get_top()[1] + 0.05
        b = self.brackets.get_bottom()[1] - 0.05
        return Rectangle(width=min(self.cw, 2 * self.hw + 0.4) - 0.04, height=t - b, stroke_color=color,
                         stroke_width=3, fill_color=color, fill_opacity=opacity).move_to(
            [self.cx(j), (t + b) / 2, 0]).set_z_index(-1)


def window(a, b):
    """Rate function that runs smooth(0 -> 1) only between fractions a and b of the animation."""
    return lambda t: smooth(min(1.0, max(0.0, (t - a) / (b - a))))


def swap(old, new, out_end=0.35, in_start=0.45, in_end=1.0, **kw):
    """Fade `old` out completely before `new` starts fading in (never two captions at once)."""
    return [FadeOut(old, rate_func=window(0, out_end)), FadeIn(new, rate_func=window(in_start, in_end), **kw)]


def dot_icon():
    """A tiny dot product: blue vector · teal vector = one yellow number."""
    def vec(color):
        return VGroup(*[Square(0.16, stroke_width=1.5, stroke_color=color, fill_color=color, fill_opacity=0.55)
                        for _ in range(3)]).arrange(RIGHT, buff=0.04)
    r = Square(0.16, stroke_width=1.5, stroke_color=YELLOW, fill_color=YELLOW, fill_opacity=0.85)
    return VGroup(vec(BLUE), Dot(radius=0.05, color=WHITE), vec(TEAL), Text("=", font_size=24, color=GREY_B),
                  r).arrange(RIGHT, buff=0.1)


def tf_label(name, values):
    """Name + 2 x 2 matrix on a dark backing, for the transformation demos."""
    grp = VGroup(Text(name, font_size=32, color=YELLOW), Mat(values, cw=0.75, rh=0.6, size=32)).arrange(DOWN, buff=0.25)
    bg = BackgroundRectangle(grp, fill_opacity=0.88, buff=0.22)
    return VGroup(bg, grp).to_corner(UL, buff=0.35)


class MatricesVideo(VoicedScene):
    VIDEO = "f03"

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

    def cue_time(self, word, after=0.0):
        """Absolute time of the first `word` spoken at least `after` s into the current section."""
        for w in self.sec["words"]:
            if w["w"] == word and w["t"] >= after:
                return self.sec_start + w["t"]
        raise ValueError(f"cue {word!r} not found after {after} s in {self.sec['file']}")

    def construct(self):
        play_token_intro(self, TITLE, 0, TAGLINE, label=LABEL)
        self.s1_hook()
        self.s2_table()
        self.s3_matvec()
        self.s4_transform()
        self.s5_shapes()
        self.s6_matmat()
        self.s7_tokens()
        self.s8_transpose()
        self.s9_code()
        self.s10_outro()
        late = self.renderer.time - (self.sec_start + self.sec["dur"] + self.SECTION_GAP)
        if late > 0.05:
            print(f"section 10 overran its narration by {late:.2f}s")
        finish(self)

    # 1. Billions of dot products ------------------------------------------------------------
    def s1_hook(self):
        self.section(1)
        rng = np.random.default_rng(3)
        icons = VGroup(*[dot_icon() for _ in range(20)])
        slots = []
        for k in range(20):
            r, c = divmod(k, 5)
            slots.append(([-5.0 + 2.5 * c + rng.uniform(-0.2, 0.2), 1.9 - 1.25 * r + rng.uniform(-0.15, 0.15), 0],
                          rng.uniform(-0.2, 0.2)))
        for icon, (p, a) in zip(icons, slots):
            icon.move_to(p).rotate(a)
        first = icons[7]
        first.save_state()
        first.rotate(-slots[7][1]).scale(1.8).move_to([0, 0.2, 0])
        cap0 = label("one dot product: two vectors → one number", font_size=30).move_to([0, 3.1, 0])
        self.play(FadeIn(first, scale=0.8), FadeIn(cap0), run_time=0.6)

        cap1 = Text("billions of dot products", font_size=34).move_to([0, 3.1, 0])
        others = [ic for k, ic in enumerate(icons) if k != 7]
        self.at("billions")
        self.play(Restore(first), *swap(cap0, cap1, 0.12, 0.17, 0.5),
                  LaggedStart(*[FadeIn(ic, scale=0.5) for ic in others], lag_ratio=0.08), run_time=2.0)

        # one by one they pile up
        heap = []
        for layer, n in enumerate([6, 5, 4, 3, 2]):
            for q in range(n):
                heap.append(([(q - (n - 1) / 2) * 1.55 + rng.uniform(-0.2, 0.2), -3.1 + 0.42 * layer, 0],
                             rng.uniform(-0.45, 0.45)))
        order = rng.permutation(20)
        cap2 = Text("one by one?  hopeless", font_size=34, color=RED).move_to([0, 3.1, 0])
        self.at("hopeless")
        self.play(*[icons[k].animate(rate_func=rush_into).move_to(heap[order[k]][0]).rotate(heap[order[k]][1])
                    for k in range(20)], *swap(cap1, cap2, 0.35, 0.45, 0.9), run_time=1.0)

        # ... and snap into one matrix
        pitch, side, cy = 0.19, 0.15, -0.3
        rows_t = VGroup(*[VGroup(*[Square(side, stroke_width=1.5, stroke_color=BLUE, fill_color=BLUE, fill_opacity=0.55)
                                   .move_to([-1.6 + (j - 1) * pitch, cy + (9.5 - i) * pitch, 0]) for j in range(3)])
                          for i in range(20)])
        vec_t = VGroup(*[Square(side, stroke_width=1.5, stroke_color=TEAL, fill_color=TEAL, fill_opacity=0.55)
                         .move_to([-0.4, cy + (1 - i) * pitch, 0]) for i in range(3)])
        res_t = VGroup(*[Square(side, stroke_width=1.5, stroke_color=YELLOW, fill_color=YELLOW, fill_opacity=0.85)
                         .move_to([1.1, cy + (9.5 - i) * pitch, 0]) for i in range(20)])
        eq = Text("=", font_size=40).move_to([0.35, cy, 0])
        brs = VGroup(brackets_around(rows_t, d=0.08), brackets_around(vec_t, d=0.08), brackets_around(res_t, d=0.08))
        cap3 = Text("a matrix: many dot products at once", font_size=34).move_to([0, 3.1, 0])
        self.remove(*icons)
        parts = [list(ic) for ic in icons]
        for p in parts:
            self.add(*p)
        anims = []
        for i, (a, d, b, e, r) in enumerate(parts):
            anims += [ReplacementTransform(a, rows_t[i]), FadeOut(d), FadeOut(e), ReplacementTransform(r, res_t[i])]
            anims.append(ReplacementTransform(b, vec_t) if i == 0 else FadeOut(b, target_position=vec_t))
        self.at("matrices")
        self.play(*anims, FadeIn(brs), FadeIn(eq), *swap(cap2, cap3, 0.3, 0.4, 0.9), run_time=1.1)
        self.end_section()

    # 2. A table of numbers ---------------------------------------------------------------------
    def s2_table(self):
        self.section(2)
        self.play(*self.clear_anims(), run_time=0.5)
        mat = Mat(M, cw=1.2, rh=0.9, size=48).move_to([0.2, 0.4, 0])
        name = Text("M =", font_size=48).next_to(mat.brackets, LEFT, buff=0.35)
        self.at("table")
        self.play(Create(mat.brackets), FadeIn(name),
                  LaggedStart(*[FadeIn(c) for row in mat.cells for c in row], lag_ratio=0.12), run_time=0.9)
        self.mat2 = mat

        right = mat.brackets[1].get_right()[0]
        top = mat.brackets.get_top()[1]
        rbands = VGroup(*[mat.row_band(i, ROW_C) for i in range(2)])
        rlabs = VGroup(*[label(f"row {i + 1}", ROW_C, 30).move_to([right + 0.85, mat.ry(i), 0]) for i in range(2)])
        self.at("rows")
        self.play(FadeIn(rbands), FadeIn(rlabs, shift=0.2 * LEFT), run_time=0.5)

        cbands = VGroup(*[mat.col_band(j, COL_C) for j in range(3)])
        clabs = VGroup(*[label(f"col {j + 1}", COL_C, 28).move_to([mat.cx(j), top + 0.45, 0]) for j in range(3)])
        self.at("columns")
        self.play(FadeOut(rbands), FadeIn(cbands), FadeIn(clabs, shift=0.2 * DOWN), run_time=0.5)

        self.at("two")
        self.play(FadeOut(cbands), FadeIn(rbands), Indicate(rlabs, color=WHITE, scale_factor=1.25), run_time=0.6)
        self.at("three")
        self.play(FadeOut(rbands), FadeIn(cbands), Indicate(clabs, color=WHITE, scale_factor=1.25), run_time=0.6)

        dims = Text("2 × 3 matrix   (rows × columns)", font_size=38,
                    t2c={"2": ROW_C, "3": COL_C, "rows": ROW_C, "columns": COL_C}).move_to([0, -1.9, 0])
        self.at("2")
        self.play(FadeOut(cbands), FadeIn(dims, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # 3. Matrix times vector ------------------------------------------------------------------------
    def s3_matvec(self):
        self.section(3)
        cy = 1.7
        mat = Mat(M, cw=0.95, rh=0.8, size=42).move_to([-2.2, cy, 0])
        vec = Mat(XV.reshape(3, 1), rh=0.8, size=42, colors=[[COL_C]] * 3).move_to([0.2, cy, 0])
        eq = Text("=", font_size=48).move_to([1.3, cy, 0])
        res = Mat(MX.reshape(2, 1), rh=0.8, size=42).move_to([2.4, cy, 0])
        low = vec.brackets.get_bottom()[1] - 0.4
        m_lab = label("M", WHITE, 32).move_to([mat.get_x(), low, 0])
        x_lab = label("x", COL_C, 32).move_to([vec.get_x(), low, 0])
        r_lab = label("M x", WHITE, 32).move_to([res.get_x(), low, 0])
        self.play(*self.clear_anims(keep=[self.mat2]), ReplacementTransform(self.mat2, mat), run_time=0.7)
        self.add(mat)

        self.at("vector")
        self.play(FadeIn(vec, shift=0.3 * LEFT), FadeIn(m_lab), FadeIn(x_lab), run_time=0.5)

        s1 = "1·3 + 0·1 + 2·2 = 7"
        s2 = "0·3 + 1·1 + (−1)·2 = −1"
        f1 = Text(s1, font_size=40)
        f2 = Text(s2, font_size=40)
        assert len(f1) == len(s1.replace(" ", "")) and len(f2) == len(s2.replace(" ", ""))
        # (formula, M-entry glyph slices, x-entry glyph slices, operator glyphs, "=", result glyphs)
        spec = [
            (f1, [(0, 1), (4, 5), (8, 9)], [(2, 3), (6, 7), (10, 11)], [1, 3, 5, 7, 9], [11], (12, 13)),
            (f2, [(0, 1), (4, 5), (9, 11)], [(2, 3), (6, 7), (13, 14)], [1, 3, 5, 7, 8, 11, 12], [14], (15, 17)),
        ]
        f1.move_to([0.0, -0.75, 0])
        f2.shift(f1[11].get_center() - f2[14].get_center() + 0.95 * DOWN)
        for f, mi, xi, _, _, ri in spec:
            for a, b in mi:
                f[a:b].set_color(ROW_C)
            for a, b in xi:
                f[a:b].set_color(COL_C)

        vband = vec.col_band(0, COL_C)
        band = None
        for r, (f, mi, xi, ops, eqi, ri) in enumerate(spec):
            new_band = mat.row_band(r, ROW_C)
            self.at("each")
            if band is None:
                self.play(FadeIn(new_band), FadeIn(vband), run_time=0.35)
                band = new_band
            else:
                self.play(band.animate.move_to(new_band), run_time=0.35)
            pairs = [AnimationGroup(TransformFromCopy(mat.cells[r][j], f[mi[j][0]:mi[j][1]]),
                                    TransformFromCopy(vec.cells[j][0], f[xi[j][0]:xi[j][1]]))
                     for j in range(3)]
            self.play(LaggedStart(*pairs, lag_ratio=0.35), FadeIn(VGroup(*[f[k] for k in ops])), run_time=0.9)
            self.play(FadeIn(VGroup(f[eqi[0]], f[ri[0]:ri[1]]), shift=0.15 * LEFT), run_time=0.3)
            self.play(TransformFromCopy(f[ri[0]:ri[1]], res.cells[r][0]), run_time=0.4)

        self.at("vector")
        self.play(FadeIn(eq), Create(res.brackets), FadeIn(r_lab), run_time=0.5)
        per = label("one entry per row", ROW_C, 32).move_to([0, -3.0, 0])
        both = VGroup(mat.row_band(0, ROW_C), mat.row_band(1, ROW_C))
        self.at("entry")
        self.play(FadeOut(band), FadeIn(both), FadeIn(per, shift=0.2 * UP),
                  Indicate(res.cells, color=ROW_C, scale_factor=1.3), run_time=0.6)
        self.mat3, self.vec3, self.res3 = mat, vec, res
        self.end_section()

    # 4. A matrix transforms space -------------------------------------------------------------------
    def s4_transform(self):
        self.section(4)
        mat, vec, res = self.mat3, self.vec3, VGroup(self.res3.brackets, self.res3.cells)
        self.play(*self.clear_anims(keep=[mat, vec, res]), run_time=0.4)
        a1 = Arrow([-3.4, 0, 0], [-1.9, 0, 0], buff=0, color=GREY_B)
        a2 = Arrow([1.9, 0, 0], [3.4, 0, 0], buff=0, color=GREY_B)
        one = label("one vector", COL_C, 28).move_to([-4.4, -1.7, 0])
        other = label("another vector", WHITE, 28).move_to([4.4, -1.7, 0])
        self.at("turns")
        self.play(vec.animate.move_to([-4.4, 0, 0]), mat.animate.move_to([0, 0, 0]),
                  res.animate.move_to([4.4, 0, 0]), GrowArrow(a1), GrowArrow(a2), FadeIn(one), run_time=0.9)
        self.at("another")
        self.play(FadeIn(other, shift=0.2 * UP), run_time=0.4)

        bg = NumberPlane(x_range=[-8, 8, 1], y_range=[-5, 5, 1]).set_stroke(GREY, 1, opacity=0.3)
        plane = NumberPlane(x_range=[-12, 12, 1], y_range=[-9, 9, 1],
                            background_line_style={"stroke_color": BLUE_D, "stroke_width": 2, "stroke_opacity": 0.8})
        tips = [VectorizedPoint(np.array([1.0, 2.0, 0.0]))]
        grp = VGroup(plane, tips[0])
        self.at("two")
        self.play(*self.clear_anims(), run_time=0.35)
        self.play(FadeIn(bg), Create(plane), run_time=0.65)

        def live_arrow(tip, color):
            return always_redraw(lambda: Arrow(ORIGIN, tip.get_center(), buff=0, color=color, stroke_width=7,
                                               max_tip_length_to_length_ratio=0.2, max_stroke_width_to_length_ratio=8))

        static = Arrow(ORIGIN, [1, 2, 0], buff=0, color=YELLOW, stroke_width=7, max_tip_length_to_length_ratio=0.2,
                       max_stroke_width_to_length_ratio=8)
        self.at("watch")
        self.play(GrowArrow(static), run_time=0.5)
        self.remove(static)
        arrows = [live_arrow(tips[0], YELLOW)]
        self.add(arrows[0])

        lab_s = tf_label("stretch", STRETCH)
        lab_r = tf_label("rotate 90°", ROTATE)
        lab_h = tf_label("shear", SHEAR)
        self.at("stretch")
        self.play(ApplyMatrix(STRETCH, grp), FadeIn(lab_s), run_time=0.9)
        self.at("rotate")
        self.play(ApplyMatrix(np.linalg.inv(STRETCH), grp), FadeOut(lab_s), run_time=0.3)
        self.play(Rotate(grp, PI / 2, about_point=ORIGIN), FadeIn(lab_r), run_time=0.55)
        self.at("shear")
        self.play(Rotate(grp, -PI / 2, about_point=ORIGIN), FadeOut(lab_r), run_time=0.3)
        self.play(ApplyMatrix(SHEAR, grp), FadeIn(lab_h), run_time=0.55)

        self.at("every")
        self.play(ApplyMatrix(np.linalg.inv(SHEAR), grp), run_time=0.3)
        extra = [([-2, 1, 0], TEAL), ([2, -1, 0], ORANGE)]
        statics = [Arrow(ORIGIN, p, buff=0, color=c, stroke_width=7, max_tip_length_to_length_ratio=0.2,
                         max_stroke_width_to_length_ratio=8) for p, c in extra]
        self.at("arrow")
        self.play(*[GrowArrow(s) for s in statics], run_time=0.3)
        self.remove(*statics)
        for p, c in extra:
            t = VectorizedPoint(np.array(p, dtype=float))
            grp.add(t)
            arrows.append(live_arrow(t, c))
        self.add(*arrows[1:])
        self.remove(lab_h)
        self.add(lab_h)  # keep the label drawn above the new arrows
        note = label("every arrow moves along", WHITE, 30)
        note = VGroup(BackgroundRectangle(note, fill_opacity=0.88, buff=0.15), note).move_to([0, -3.2, 0])
        self.at("moves")
        self.play(ApplyMatrix(SHEAR, grp), FadeIn(note), run_time=0.8)
        for a in arrows:
            a.clear_updaters()
        self.end_section()

    # 5. Shapes must line up ------------------------------------------------------------------------
    def s5_shapes(self):
        self.section(5)
        self.play(*self.clear_anims(), run_time=0.4)
        cy = 1.5
        mb = cell_grid(2, 3, 0.5, 0.5, 0.44, WHITE, 0.15).move_to([-2.9, cy, 0])
        dot = Text("·", font_size=60).move_to([-1.7, cy, 0])
        xb = cell_grid(3, 1, 0.5, 0.5, 0.44, COL_C, 0.4).move_to([-1.0, cy, 0])
        arr = Arrow([0.4, cy, 0], [1.5, cy, 0], buff=0, color=GREY_B)
        rb = cell_grid(2, 1, 0.5, 0.5, 0.44, ROW_C, 0.4).move_to([2.1, cy, 0])
        g_top = Brace(mb, UP, color=GREEN)
        g_top_l = Text("3", font_size=30, color=GREEN).next_to(g_top, UP, buff=0.1)
        g_x = Brace(xb, RIGHT, color=GREEN)
        g_x_l = Text("3", font_size=30, color=GREEN).next_to(g_x, RIGHT, buff=0.1)
        y_m = Brace(mb, LEFT, color=ROW_C)
        y_m_l = Text("2", font_size=30, color=ROW_C).next_to(y_m, LEFT, buff=0.1)
        y_r = Brace(rb, RIGHT, color=ROW_C)
        y_r_l = Text("2", font_size=30, color=ROW_C).next_to(y_r, RIGHT, buff=0.1)

        s = "(2 × 3) · (3) → (2)"
        form = Text(s, font_size=52).move_to([0, -0.7, 0])
        glyphs(form, s, 5, 6).set_color(GREEN)
        glyphs(form, s, 11, 12).set_color(GREEN)
        glyphs(form, s, 1, 2).set_color(ROW_C)
        glyphs(form, s, 17, 18).set_color(ROW_C)
        title = Text("the shapes must line up", font_size=34).move_to([0, 3.25, 0])
        self.at("line")
        self.play(FadeIn(title), FadeIn(mb), FadeIn(dot), FadeIn(xb), GrowArrow(arr), FadeIn(rb), FadeIn(form),
                  GrowFromCenter(g_top), GrowFromCenter(g_x), FadeIn(g_top_l), FadeIn(g_x_l), run_time=0.8)

        self.at("two")
        self.play(Indicate(glyphs(form, s, 0, 7), scale_factor=1.15), Indicate(mb, scale_factor=1.08), run_time=0.7)
        self.at("vector")
        self.at("three")
        self.play(Indicate(glyphs(form, s, 10, 13), scale_factor=1.2), Indicate(xb, scale_factor=1.08), run_time=0.7)
        self.at("two")
        self.play(Indicate(glyphs(form, s, 16, 19), scale_factor=1.2), Indicate(rb, scale_factor=1.08),
                  GrowFromCenter(y_m), GrowFromCenter(y_r), FadeIn(y_m_l), FadeIn(y_r_l), run_time=0.7)

        g = "(m × n) · (n) → (m)"
        gen = Text(g, font_size=52).move_to([0, -2.3, 0])
        n_gl = VGroup(glyphs(gen, g, 5, 6), glyphs(gen, g, 11, 12)).set_color(GREEN)
        m_gl = VGroup(glyphs(gen, g, 1, 2), glyphs(gen, g, 17, 18)).set_color(ROW_C)
        gen_l = label("in general", GREY_B, 26).next_to(gen, UP, buff=0.3)
        self.at("m")
        self.play(FadeIn(gen, shift=0.2 * UP), FadeIn(gen_l), run_time=0.6)
        self.at("turns")
        self.at("n")
        self.play(*[Indicate(g, color=WHITE, scale_factor=1.25) for g in n_gl], run_time=0.6)
        self.at("m")
        self.play(*[Indicate(g, color=WHITE, scale_factor=1.25) for g in m_gl], run_time=0.6)
        self.end_section()

    # 6. Matrix times matrix -------------------------------------------------------------------------
    def s6_matmat(self):
        self.section(6)
        self.play(*self.clear_anims(), run_time=0.4)
        cy = 0.35
        kw = dict(cw=0.66, rh=0.52, size=26)
        xm = Mat(X6, **kw).move_to([-3.3, cy, 0])
        wm = Mat(W6, colors=[[W_COLOR] * 4] * 4, **kw).move_to([0.25, cy, 0])
        rm = Mat(R6, **kw).move_to([3.8, cy, 0])
        dot = Text("·", font_size=56).move_to([(xm.get_right()[0] + wm.get_left()[0]) / 2, cy, 0])
        eq = Text("=", font_size=44).move_to([(wm.get_right()[0] + rm.get_left()[0]) / 2, cy, 0])
        toks = VGroup(*[Text(w, font_size=24, color=BLUE_B) for w in WORDS])
        for i, t in enumerate(toks):
            t.move_to([xm.brackets.get_left()[0] - 0.25 - t.width / 2, xm.ry(i), 0])
        top = xm.brackets.get_top()[1] + 0.4
        names = VGroup(label("X", WHITE, 32).move_to([xm.get_x(), top, 0]),
                       label("W", W_COLOR, 32).move_to([wm.get_x(), top, 0]),
                       label("X · W", WHITE, 32).move_to([rm.get_x(), top, 0]))
        slots = VGroup(*[Rectangle(width=0.58, height=0.44, stroke_color=GREY_D, stroke_width=1.5)
                         .move_to([rm.cx(j), rm.ry(i), 0]) for i in range(5) for j in range(4)])
        rcells = [c for row in rm.cells for c in row]
        for c in rcells:
            c.set_opacity(0)
        rc_group = VGroup(*rcells)
        note = caption("illustrative values")
        self.at("matrices")
        self.play(FadeIn(xm), FadeIn(toks), FadeIn(dot), FadeIn(wm), FadeIn(eq), FadeIn(rm.brackets),
                  FadeIn(slots), FadeIn(names), FadeIn(note), run_time=1.0)
        self.add(rc_group)

        rband = xm.row_band(0, ROW_C)
        cband = wm.col_band(0, COL_C)
        self.at("row")
        self.play(FadeIn(rband), run_time=0.4)
        self.at("column")
        self.play(FadeIn(cband), run_time=0.4)

        s = "row 1 · column 1 = 1·1 + (−2)·(−2) + 1·(−1) + 1·2 = 6"
        arith = Text(s, font_size=28).move_to([0, -2.2, 0])
        glyphs(arith, s, 0, 5).set_color(ROW_C)
        glyphs(arith, s, 8, 16).set_color(COL_C)
        box0 = Rectangle(width=0.58, height=0.44, stroke_color=YELLOW, stroke_width=3).move_to(slots[0])
        self.at("second")
        self.play(rcells[0].animate.set_opacity(1), Create(box0), FadeIn(arith, shift=0.2 * UP), run_time=0.6)

        # sweep: every row of X against every column of W, one cell at a time
        k = ValueTracker(1.0)

        def follow(_):
            c = min(int(k.get_value()), 19)
            i, j = divmod(c, 4)
            rband.set_y(xm.ry(i))
            cband.set_x(wm.cx(j))
            for idx, cell in enumerate(rcells):
                cell.set_opacity(1 if idx <= c else 0)

        rc_group.add_updater(follow)
        t0 = self.renderer.time
        t_end = t0 + 3.0

        def kval(t):
            return 1.0 + 18.99 * (t - t0) / (t_end - t0)

        dim_x = label("5 × 4", GREY_B, 28).move_to([xm.get_x(), xm.brackets.get_bottom()[1] - 0.35, 0])
        dim_w = label("4 × 4", GREY_B, 28).move_to([wm.get_x(), wm.brackets.get_bottom()[1] - 0.35, 0])
        dim_r = label("5 × 4", GREY_B, 28).move_to([rm.get_x(), rm.brackets.get_bottom()[1] - 0.35, 0])
        ta = self.cue_time("5", 8.0)
        tb = self.cue_time("4", 10.0)
        self.play(k.animate(rate_func=linear).set_value(kval(ta)), run_time=ta - t0)
        for t_next, dim in [(tb, dim_x), (t_end, dim_w)]:
            dur = t_next - self.renderer.time
            self.play(k.animate(rate_func=linear).set_value(kval(t_next)),
                      FadeIn(dim, rate_func=lambda a, d=dur: smooth(min(1.0, a * d / 0.5))), run_time=dur)
        rc_group.clear_updaters()

        fs = "(5 × 4) · (4 × 4) → (5 × 4)"
        final = Text(fs, font_size=40).move_to([0, -2.2, 0])
        glyphs(final, fs, 5, 6).set_color(GREEN)
        glyphs(final, fs, 11, 12).set_color(GREEN)
        glyphs(final, fs, 1, 2).set_color(ROW_C)
        glyphs(final, fs, 21, 22).set_color(ROW_C)
        glyphs(final, fs, 15, 16).set_color(COL_C)
        glyphs(final, fs, 25, 26).set_color(COL_C)
        self.at("gives")
        self.at("5")
        self.play(FadeOut(rband), FadeOut(cband), FadeOut(box0), FadeIn(dim_r),
                  *swap(arith, final, 0.4, 0.45, 1.0), run_time=0.8)
        self.end_section()

    # 7. Every token in one step ----------------------------------------------------------------------
    def s7_tokens(self):
        self.section(7)
        self.play(*self.clear_anims(), run_time=0.4)
        toks = VGroup(*[token(w) for w in WORDS]).arrange(RIGHT, buff=0.3).move_to([0, 0.3, 0])
        self.at("llms")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * UP) for t in toks], lag_ratio=0.12), run_time=0.7)

        py, px, side, cy = 0.72, 0.58, 0.48, -0.2
        ys = [cy + (2 - i) * py for i in range(5)]
        xg = cell_grid(5, 4, px, py, side, BLUE).move_to([-3.75, cy, 0])
        xb = brackets_around(xg)
        self.at("tokens")
        self.play(*[t.animate.move_to([-5.75, y, 0]) for t, y in zip(toks, ys)], run_time=0.8)
        top = xb.get_top()[1] + 0.5
        x_name = label("X", WHITE, 32).move_to([xg.get_x(), top, 0])
        self.at("rows")
        self.play(LaggedStart(*[FadeIn(r, shift=0.3 * RIGHT) for r in xg], lag_ratio=0.12), Create(xb),
                  FadeIn(x_name), run_time=0.7)

        wg = cell_grid(4, 4, px, py, side, W_COLOR).move_to([-0.45, cy, 0])
        wb = brackets_around(wg)
        dot = Text("·", font_size=56).move_to([(xb.get_right()[0] + wb.get_left()[0]) / 2, cy, 0])
        w_name = label("W  (weights)", W_COLOR, 30).move_to([wg.get_x(), top, 0])
        self.at("weight")
        self.play(FadeIn(dot), FadeIn(wg), Create(wb), FadeIn(w_name), run_time=0.6)

        rg = cell_grid(5, 4, px, py, side, COL_C).move_to([2.95, cy, 0])
        rb = brackets_around(rg)
        eq = Text("=", font_size=44).move_to([(wb.get_right()[0] + rb.get_left()[0]) / 2, cy, 0])
        r_name = label("X · W", WHITE, 32).move_to([rg.get_x(), top, 0])
        rtoks = VGroup(*[Text(w, font_size=24, color=COL_C) for w in WORDS])
        for t, y in zip(rtoks, ys):
            t.move_to([rb.get_right()[0] + 0.45 + t.width / 2, y, 0])
        self.at("transformed")
        self.play(*[TransformFromCopy(xg[i], rg[i]) for i in range(5)], FadeIn(eq), Create(rb), FadeIn(r_name),
                  FadeIn(rtoks, shift=0.2 * LEFT), run_time=0.8)

        frame = SurroundingRectangle(VGroup(rg, rb), color=YELLOW, buff=0.1)
        cap = Text("every token, one multiplication", font_size=34, color=YELLOW).move_to([0, -2.95, 0])
        self.at("single")
        self.play(Indicate(rg, color=YELLOW, scale_factor=1.05), Create(frame), FadeIn(cap, shift=0.2 * UP),
                  run_time=0.7)
        self.end_section()

    # 8. The transpose -------------------------------------------------------------------------------
    def s8_transpose(self):
        self.section(8)
        self.play(*self.clear_anims(), run_time=0.4)
        p, x0, y0 = 1.0, -0.9, 1.45
        cells = [[num(fmt(M[i, j]), ROW_C if i == 0 else WHITE, 46).move_to([x0 + j * p, y0 - i * p, 0])
                  for j in range(3)] for i in range(2)]
        allc = VGroup(*[c for row in cells for c in row])
        br = bracket_pair(x0 - 0.5, x0 + 2 * p + 0.5, y0 + 0.45, y0 - p - 0.45)
        br_t = bracket_pair(x0 - 0.5, x0 + p + 0.5, y0 + 0.45, y0 - 2 * p - 0.45)
        name = VGroup(Text("M", font_size=48), label("2 × 3", GREY_B, 30)).arrange(DOWN, buff=0.2)
        name.move_to([x0 - 1.7, y0 - p, 0])
        name_t = VGroup(Text("Mᵀ", font_size=48), label("3 × 2", GREY_B, 30)).arrange(DOWN, buff=0.2).move_to(name)
        self.at("transpose")
        self.play(FadeIn(allc), Create(br), FadeIn(name), run_time=0.7)

        diag = DashedLine([x0 - 0.75, y0 + 0.75, 0], [x0 + 2 * p + 0.75, y0 - 2 * p - 0.75, 0], dash_length=0.12,
                          color=GREY_B, stroke_width=2.5).set_z_index(-1)
        d_lab = label("diagonal", GREY_B, 24).next_to(diag.get_end(), RIGHT, buff=0.15)
        self.at("flips")
        self.play(Create(diag), FadeIn(d_lab), run_time=0.5)
        self.at("over")
        self.play(*[cells[i][j].animate.move_to([x0 + i * p, y0 - j * p, 0]) for i in range(2) for j in range(3)],
                  Transform(br, br_t), *swap(name, name_t, 0.3, 0.55, 0.9), run_time=1.2)
        rc = Text("rows ↔ columns", font_size=36, t2c={"rows": ROW_C}).move_to([3.9, y0 - p, 0])
        self.at("rows")
        self.play(FadeIn(rc, shift=0.2 * LEFT), run_time=0.5)

        # Q · Kᵀ: every query against every key (Kᵀ above the result, Q to its left)
        q, qx = 0.5, 0.64                 # row pitch; column pitch of the result / Kᵀ (room for the token labels)
        rx, ry = -0.7, -0.35              # centre of result cell (0, 0)

        def rect(w, color, opacity):
            return Rectangle(width=w, height=0.42, stroke_width=1.5, stroke_color=color, fill_color=color,
                             fill_opacity=opacity)

        res = VGroup(*[VGroup(*[rect(0.56, GREY_B, 0).set_fill(YELLOW, 0).move_to([rx + j * qx, ry - i * q, 0])
                                for j in range(5)]) for i in range(5)])
        qg = VGroup(*[VGroup(*[rect(0.42, BLUE, 0.45).move_to([rx - 0.69 - (3 - j) * q, ry - i * q, 0])
                               for j in range(4)]) for i in range(5)])
        kx = rx + 4 * qx + 1.39
        kg = VGroup(*[VGroup(*[rect(0.42, COL_C, 0.45).move_to([kx + j * q, ry - i * q, 0]) for j in range(4)])
                      for i in range(5)])
        q_labs = VGroup(*[Text(w, font_size=22, color=BLUE_B) for w in WORDS])
        for i, t in enumerate(q_labs):
            t.move_to([qg[i][0].get_left()[0] - 0.2 - t.width / 2, ry - i * q, 0])
        k_labs = VGroup(*[Text(w, font_size=22, color=COL_C) for w in WORDS])
        for i, t in enumerate(k_labs):
            t.move_to([kg[i][-1].get_right()[0] + 0.2 + t.width / 2, ry - i * q, 0])
        q_name = label("Q", BLUE_B, 32).move_to([qg.get_x(), ry + 0.6, 0])
        k_name = label("K", COL_C, 32).move_to([kg.get_x(), ry + 0.6, 0])
        self.at("q")
        self.play(FadeOut(VGroup(allc, br, name_t, diag, d_lab, rc), rate_func=window(0, 0.45)),
                  *[FadeIn(m, rate_func=window(0.5, 1.0)) for m in (qg, q_labs, q_name)], run_time=0.4)
        self.at("k")
        self.play(FadeIn(kg), FadeIn(k_labs), FadeIn(k_name), run_time=0.35)

        def kt_pos(i, j):  # K[i][j] (token i, feature j) -> Kᵀ[j][i]
            return [rx + i * qx, ry + 0.62 + (3 - j) * q, 0]

        kt_top = ry + 0.62 + 3 * q + 0.21
        kt_name = label("Kᵀ", COL_C, 32).move_to([rx - 0.85, ry + 0.62 + 1.5 * q, 0])
        self.at("transposed")
        self.play(*[kg[i][j].animate.stretch_to_fit_width(0.56).move_to(kt_pos(i, j))
                    for i in range(5) for j in range(4)],
                  *[k_labs[i].animate.move_to([rx + i * qx, kt_top + 0.3, 0]) for i in range(5)],
                  *swap(k_name, kt_name, 0.4, 0.5, 1.0), run_time=0.6)

        cap = Text("Q · Kᵀ = every query vs every key", font_size=34).move_to([0.3, 3.3, 0])
        s_name = label("Q · Kᵀ", YELLOW, 30).move_to([rx + 2 * qx, ry - 4 * q - 0.6, 0])
        self.at("compares")
        self.play(FadeIn(res), FadeIn(cap, shift=0.2 * DOWN), FadeIn(s_name), run_time=0.4)

        qband = SurroundingRectangle(qg[0], color=YELLOW, buff=0.05)
        self.at("query")
        self.play(Create(qband), *[res[0][j].animate.set_fill(YELLOW, 0.45) for j in range(5)], run_time=0.4)
        kband = SurroundingRectangle(VGroup(*[kg[0][j] for j in range(4)]), color=YELLOW, buff=0.05)
        self.at("key")
        self.play(Create(kband), *[res[i][0].animate.set_fill(YELLOW, 0.45) for i in range(1, 5)],
                  res[0][0].animate.set_fill(YELLOW, 0.9), run_time=0.3)
        self.at("all")
        self.play(FadeOut(qband), FadeOut(kband),
                  *[res[i][j].animate.set_fill(YELLOW, SCORE_LIGHT[i, j]) for i in range(5) for j in range(5)],
                  run_time=0.5)
        self.end_section()

    # 9. NumPy --------------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, 26)
        VGroup(code, hl).move_to([0, 0.2, 0])
        hl.match_y(code.line_numbers[3])
        self.play(*self.clear_anims(), FadeIn(code, shift=0.3 * UP), run_time=0.7)
        self.at("at")
        self.play(Create(hl), run_time=0.4)
        self.at("t")
        self.play(highlight(hl, code, 7), run_time=0.4)
        self.at("shapes")
        self.play(highlight(hl, code, 6), run_time=0.4)
        self.end_section()

    # 10. Where you'll see this + next up ------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        self.play(*self.clear_anims(), run_time=0.45)
        card = used_in_card(USED_IN)
        rows = card[1]
        self.at("power")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.7)
        for cue, idx in [("5", [0]), ("10", [4]), ("queries", [0]), ("attention", [1, 2]), ("mlp", [3]),
                         ("logits", [4])]:
            self.at(cue)
            self.play(*[Indicate(rows[i], color=YELLOW, scale_factor=1.1) for i in idx], run_time=0.6)
        nxt = next_up_card(NEXT)
        self.at("bend")
        self.play(FadeOut(card), run_time=0.35)
        self.play(FadeIn(nxt, shift=0.2 * UP), run_time=0.6)
        self.end_section()
