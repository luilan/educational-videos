"""Video 6 — Attention II: The Math.

Render from the repo root:  ./render.sh how-llms-work v06
"""
import numpy as np
from manim import *

from common import code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from v06_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

MONO = "DejaVu Sans Mono"
TOKEN_COLOR = BLUE_D
Q_COLOR, K_COLOR, V_COLOR, W_COLOR = YELLOW, TEAL, ORANGE, GREEN
WORDS = ["The", "cat", "sat", "on", "the"]

CODE = """def attention(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv            # project
    scores = Q @ K.T / np.sqrt(K.shape[-1])     # score and scale
    n = len(X)
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    scores[mask] = -np.inf                      # no peeking ahead
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)   # softmax, row by row
    return weights @ V                          # mix the values"""

# ------------------------------------------------------------------ the worked example
# Inputs are drawn once with a fixed seed and rounded to one decimal; everything after that
# is computed exactly from those rounded matrices.
SEED = 182528  # chosen by search: uneven weights, rows of rounded weights sum to 1.00, few rounding surprises
_rng = np.random.default_rng(SEED)
X = np.round(_rng.uniform(-1.5, 1.5, (5, 4)), 1)
WQ = np.round(_rng.uniform(-1.0, 1.0, (4, 4)), 1)
WK = np.round(_rng.uniform(-1.0, 1.0, (4, 4)), 1)
WV = np.round(_rng.uniform(-1.0, 1.0, (4, 4)), 1)
Q, K, V = X @ WQ, X @ WK, X @ WV
SCORES = Q @ K.T
SCALED = SCORES / np.sqrt(4)
MASK = np.triu(np.ones((5, 5), dtype=bool), k=1)
MASKED = np.where(MASK, -np.inf, SCALED)
WEIGHTS = np.exp(MASKED - MASKED.max(-1, keepdims=True))
WEIGHTS /= WEIGHTS.sum(-1, keepdims=True)
OUT = WEIGHTS @ V
assert np.allclose(np.round(WEIGHTS, 2).sum(1), 1.0)  # every displayed row sums to 1.00


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


RAW_THE = softmax(SCORES[4])      # row "the" without the ÷ 2 (sharp)
SCALED_THE = softmax(SCALED[4])   # with it (= the final weights of row "the")

# ------------------------------------------------------------------ layout constants
CW = 1.02    # column pitch of 2-decimal matrices
CW1 = 0.86   # column pitch of 1-decimal matrices
RH = 0.5     # row pitch
HW = 0.41    # half width of the widest 2-decimal cell ("−2.44")
HW1 = 0.32   # half width of the widest 1-decimal cell ("−0.9")
LAB_GAP = 0.55  # bracket-to-row-label distance


def fmt(v, nd=2):
    if v == -np.inf:
        return "−∞"
    s = f"{v:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−")


def num(s, color=WHITE, size=20):
    """Matrix cell text; rendered larger and scaled down so the mono glyphs keep their spacing."""
    return Text(s, font=MONO, font_size=round(size * 1.2), color=color).scale(1 / 1.2)


def label(text, color=GREY_B, font_size=22, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def small_token(word, color=TOKEN_COLOR):
    lab = Text(word, font_size=24).scale(20 / 24)
    box = RoundedRectangle(corner_radius=0.08, width=max(lab.width + 0.24, 0.5), height=0.4,
                           stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.3)
    return VGroup(box, lab.move_to(box))


def sub_sym(base, sub, color=WHITE, fs=30):
    """E.g. W with a subscript Q, built from two Texts (no LaTeX)."""
    b = Text(base, font_size=fs, color=color)
    s = Text(sub, font_size=max(20, round(fs * 0.65)), color=color)
    s.next_to(b, RIGHT, buff=0.04).align_to(b, DOWN).shift(0.1 * DOWN)
    return VGroup(b, s)


def bracket_pair(left, right, top, bottom, color=GREY_B):
    def one(x, d):
        pts = [[x + d, top, 0], [x, top, 0], [x, bottom, 0], [x + d, bottom, 0]]
        return VMobject(stroke_color=color, stroke_width=2.5).set_points_as_corners(pts)
    return VGroup(one(left, 0.12), one(right, -0.12))


def gslice(mob, s, start, end):
    """Glyphs of Text `mob` (built from string s) for s[start:end]; spaces have no glyphs."""
    a = len(s[:start].replace(" ", ""))
    return mob[a:a + len(s[start:end].replace(" ", ""))]


class Mat(VGroup):
    """A matrix of Text cells with bracket lines. cells[i][j]; cells are right-aligned per column."""

    def __init__(self, values, x0, y0, nd=2, cw=CW, rh=RH, size=20, color=WHITE, hw=None,
                 bracket_color=None):
        super().__init__()
        values = np.asarray(values, dtype=float)
        n, m = values.shape
        self.cw, self.rh = cw, rh
        texts = [[num(fmt(values[i, j], nd), color, size) for j in range(m)] for i in range(n)]
        self.hw = hw if hw is not None else max(t.width for r in texts for t in r) / 2
        self.cells = VGroup()
        for i in range(n):
            row = VGroup()
            for j in range(m):
                t = texts[i][j]
                t.move_to([x0 + j * cw + self.hw - t.width / 2, y0 - i * rh, 0])
                row.add(t)
            self.cells.add(row)
        top = y0 + 0.25
        bottom = y0 - (n - 1) * rh - 0.25
        self.brackets = bracket_pair(x0 - self.hw - 0.16, x0 + (m - 1) * cw + self.hw + 0.16, top, bottom,
                                     bracket_color or color)
        self.add(self.brackets, self.cells)

    # geometry read from the current cell positions (valid after moves)
    def cx(self, j):
        return self.cells[0][j].get_right()[0] - self.hw

    def ry(self, i):
        return self.cells[i][0].get_y()

    def cell_box(self, i, j, pad=0.03, **kw):
        return Rectangle(width=self.cw - 2 * pad, height=self.rh - 2 * pad, **kw).move_to(
            [self.cx(j), self.ry(i), 0])

    def row_band(self, i, color, **kw):
        left, right = self.brackets[0].get_left()[0], self.brackets[1].get_right()[0]
        return Rectangle(width=right - left + 0.1, height=self.rh - 0.04, stroke_color=color,
                         stroke_width=kw.get("stroke_width", 3), fill_color=color,
                         fill_opacity=kw.get("fill_opacity", 0.12)).move_to(
            [(left + right) / 2, self.ry(i), 0]).set_z_index(-1)

    def col_band(self, j, color, top=None, **kw):
        t = self.brackets.get_top()[1] + 0.05 if top is None else top
        b = self.brackets.get_bottom()[1] - 0.05
        return Rectangle(width=self.cw - 0.04, height=t - b, stroke_color=color, stroke_width=3,
                         fill_color=color, fill_opacity=kw.get("fill_opacity", 0.12)).move_to(
            [self.cx(j), (t + b) / 2, 0]).set_z_index(-1)

    def row_labels(self, gap=0.6):
        x = self.brackets[0].get_left()[0] - gap
        return VGroup(*[small_token(w).move_to([x, self.ry(i), 0]) for i, w in enumerate(WORDS)])

    def col_labels(self, gap=0.35):
        y = self.brackets.get_top()[1] + gap
        return VGroup(*[small_token(w).move_to([self.cx(j), y, 0]) for j, w in enumerate(WORDS)])


def retext(old, s, color=WHITE, shift=ORIGIN, size=20):
    """A new cell text that sits where `old` sits (same right edge and row), optionally shifted."""
    t = num(s, color, size)
    t.align_to(old, RIGHT).set_y(old.get_y())
    return t.shift(shift)


def gpu_icon():
    pins = VGroup()
    for k in range(6):
        t = -0.7 + k * 0.28
        pins.add(Line([t, 0.9, 0], [t, 1.15, 0]), Line([t, -0.9, 0], [t, -1.15, 0]),
                 Line([0.9, t, 0], [1.15, t, 0]), Line([-0.9, t, 0], [-1.15, t, 0]))
    pins.set_stroke(GREY_B, 4)
    body = RoundedRectangle(corner_radius=0.12, width=1.8, height=1.8, stroke_color=GREY_B, stroke_width=3,
                            fill_color=GREY_E, fill_opacity=1)
    die = RoundedRectangle(corner_radius=0.06, width=1.05, height=1.05, stroke_color=GREEN, stroke_width=2.5,
                           fill_color=BLACK, fill_opacity=1)
    lab = Text("GPU", font_size=30, color=GREEN)
    return VGroup(pins, body, die, lab)


def head_icon(color=BLUE_D):
    shoulders = AnnularSector(inner_radius=0, outer_radius=1.0, angle=PI, start_angle=0, stroke_width=0,
                              fill_color=color, fill_opacity=0.85)
    head = Circle(radius=0.55, stroke_width=0, fill_color=color, fill_opacity=0.85)
    head.next_to(shoulders, UP, buff=0.08)
    return VGroup(shoulders, head)


def thought(anchor, pos, color, text="?"):
    """A thought bubble at `pos` with two small trailing dots pointing back to `anchor`."""
    body = Ellipse(width=1.15, height=0.9, stroke_color=color, stroke_width=3, fill_color=BLACK,
                   fill_opacity=1).move_to(pos)
    mark = Text(text, font_size=44, color=color).move_to(body)
    d = np.array(anchor) - np.array(pos)
    d = d / np.linalg.norm(d)
    edge = np.array(pos) + d * 0.5
    dots = VGroup(Circle(radius=0.09, color=color, stroke_width=2.5).move_to(edge + d * 0.22),
                  Circle(radius=0.055, color=color, stroke_width=2.5).move_to(edge + d * 0.45))
    return VGroup(dots, body, mark)


class AttentionMathVideo(VoicedScene):
    VIDEO = "v06"

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

    def at(self, *words):
        """Same as VoicedScene.at, but also reports a cue reached late (previous animation too long)."""
        super().at(*words)
        late = self.renderer.time - (self.sec_start + self.sec["words"][self.cursor - 1]["t"])
        if late > 0.1:
            print(f"cue {words[0]!r} in {self.sec['file']} reached {late:.2f}s late")

    def construct(self):
        play_token_intro(self, TITLE, 6, TAGLINE)
        self.illus = label("illustrative values", font_size=20).move_to([5.7, -3.62, 0])
        self.s1_intro()
        self.s2_input()
        self.s3_projections()
        self.s4_scores()
        self.s5_scale()
        self.s6_mask()
        self.s7_softmax()
        self.s8_weighted_sum()
        self.s9_formula()
        self.s10_code()
        self.s11_outro()
        late = self.renderer.time - (self.sec_start + self.sec["dur"] + self.SECTION_GAP)
        if late > 0.05:
            print(f"section 11 overran its narration by {late:.2f}s")
        finish(self)

    # 1. Intro ----------------------------------------------------------------------
    def s1_intro(self):
        self.section(1)
        toks = VGroup(*[token(w) for w in WORDS]).arrange(RIGHT, buff=0.3).move_to([0, -0.6, 0])
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * UP) for t in toks], lag_ratio=0.15), run_time=0.9)

        arcs = VGroup()
        for i in range(5):
            for j in range(i):
                p, q = toks[j].get_top() + 0.05 * UP, toks[i].get_top() + 0.05 * UP
                h = 0.32 * abs(q[0] - p[0]) + 0.3
                arcs.add(CubicBezier(p, p + h * UP, q + h * UP, q))
        arcs.set_stroke(GREY_B, 2, opacity=0.45)
        recap = label("last time: each token borrowed meaning from the others", font_size=26).move_to([0, 2.7, 0])
        self.at("asked")
        self.play(FadeIn(recap, shift=0.2 * DOWN), LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.08),
                  run_time=1.0)
        self.at("borrowed")
        flash = arcs.copy().set_stroke(YELLOW, 4, opacity=1)
        self.play(LaggedStart(*[ShowPassingFlash(a, time_width=0.5) for a in flash], lag_ratio=0.08),
                  run_time=1.2)

        now = Text("now: with actual numbers", font_size=32).move_to([0, 3.2, 0])
        self.at("numbers")
        targets = [[-3.65, 1.2 - 0.75 * i, 0] for i in range(5)]
        self.play(FadeOut(arcs), FadeOut(recap), FadeIn(now, shift=0.2 * DOWN),
                  *[t.animate.move_to(p) for t, p in zip(toks, targets)], run_time=1.0)
        self.toks = toks
        self.at("the")
        self.play(LaggedStart(*[Indicate(t, color=YELLOW, scale_factor=1.12) for t in toks], lag_ratio=0.2),
                  run_time=1.0)
        self.end_section()

    # 2. The input matrix X -----------------------------------------------------------
    def s2_input(self):
        self.section(2)
        toks = self.toks
        xm = Mat(X, -2.0, 1.2, nd=1, cw=1.3, rh=0.75, size=28)
        self.play(*self.clear_anims(keep=[toks]), run_time=0.5)
        self.at("input")
        self.play(Create(xm.brackets), FadeIn(self.illus), run_time=0.6)
        self.at("row")
        self.play(LaggedStart(*[FadeIn(r, shift=0.3 * LEFT) for r in xm.cells], lag_ratio=0.3), run_time=1.3)
        self.add(xm)

        big_x = Text("X", font_size=64).move_to([4.0, 0.35, 0])
        self.at("x")
        self.play(FadeIn(big_x, scale=1.3), run_time=0.6)
        rows_lab = label("5 rows", font_size=28, color=WHITE).move_to([4.0, -0.55, 0])
        self.at("rows")
        self.play(FadeIn(rows_lab), LaggedStart(*[Indicate(t, color=YELLOW, scale_factor=1.1) for t in toks],
                                                lag_ratio=0.1), run_time=0.6)
        top = xm.brackets.get_top()[1] + 0.25
        left, right = xm.brackets.get_left()[0] + 0.1, xm.brackets.get_right()[0] - 0.1
        brace = VMobject(stroke_color=GREY_B, stroke_width=2.5).set_points_as_corners(
            [[left, top - 0.1, 0], [left, top, 0], [right, top, 0], [right, top - 0.1, 0]])
        d_lab = Text("d columns", font_size=30).next_to(brace, UP, buff=0.18)
        self.at("d")
        self.play(Create(brace), FadeIn(d_lab, shift=0.1 * DOWN), run_time=0.6)
        d4 = Text("d = 4", font_size=30, color=YELLOW).move_to(d_lab)
        shape = label("5 × 4", font_size=28).move_to([4.0, -1.2, 0])
        self.at("four")
        self.play(Transform(d_lab, d4), FadeIn(shape), run_time=0.7)
        self.xm, self.big_x = xm, big_x
        self.end_section()

    # 3. Projections -------------------------------------------------------------------
    def s3_projections(self):
        self.section(3)
        centers = [-4.53, 0.0, 4.53]
        names = [("Q", Q_COLOR, WQ), ("K", K_COLOR, WK), ("V", V_COLOR, WV)]
        prefixes = VGroup()
        ws, w_labs = [], VGroup()
        for c, (n, col, W) in zip(centers, names):
            m = Mat(W, c - 1.5 * CW1, 2.7, nd=1, cw=CW1, rh=0.46, hw=HW1, bracket_color=GREY_B)
            ws.append(m)
            xdot = Text("X ·", font_size=30)
            wsym = sub_sym("W", n, col, 30)
            tag = label("learned", font_size=20)
            lab = VGroup(xdot, wsym, tag).arrange(RIGHT, buff=0.15)
            wsym.shift((xdot[0].get_bottom()[1] - wsym[0].get_bottom()[1]) * UP)
            tag.match_y(xdot[0])
            lab.move_to([c, 3.42, 0])
            prefixes.add(xdot)
            w_labs.add(VGroup(wsym, tag))

        # X (with its row labels) collapses into the three "X ·" prefixes
        # X (with its row labels) fades up and away; its name flies into the three "X ·" prefixes
        others = self.on_stage(keep=[self.xm, self.toks, self.illus, self.big_x])
        self.play(*[FadeOut(m) for m in others], FadeOut(VGroup(self.xm, self.toks), shift=0.4 * UP),
                  *[TransformFromCopy(self.big_x, p) for p in prefixes], FadeOut(self.big_x), run_time=1.0)
        self.at("three")
        self.play(*[Create(m.brackets) for m in ws], run_time=0.5)
        for (n, _, _), m, lab, cue in zip(names, ws, w_labs, ["wq", "wk", "wv"]):
            self.at(cue)
            self.play(FadeIn(lab, shift=0.1 * DOWN), FadeIn(m.cells, lag_ratio=0.05), run_time=0.55)
            self.add(m)

        res, res_labs = [], VGroup()
        for c, (n, col, _) in zip(centers, names):
            mat = {"Q": Q, "K": K, "V": V}[n]
            m = Mat(mat, c - 1.5 * CW, -0.1, color=col, hw=HW)
            res.append(m)
            res_labs.add(Text(n, font_size=34, color=col).move_to([c, 0.55, 0]))
        for k, cue in enumerate(["q", "k", "v"]):
            self.at(cue)
            src = VGroup(prefixes[k], ws[k].cells).copy()
            self.play(ReplacementTransform(src, res[k].cells), FadeIn(res[k].brackets), FadeIn(res_labs[k]),
                      run_time=0.42)
            self.add(res[k])

        words = VGroup(*[Text(w, font_size=24, color=col).move_to([c, -2.72, 0])
                         for w, c, col in zip(["a query per token", "a key per token", "a value per token"],
                                              centers, [Q_COLOR, K_COLOR, V_COLOR])])
        for k, cue in enumerate(["query", "key", "value"]):
            self.at(cue)
            self.play(FadeIn(words[k], shift=0.15 * UP), run_time=0.4)
        once = Text("all tokens at once", font_size=26).move_to([0, -3.3, 0])
        self.at("all")
        rows = [r for m in res for r in m.cells]
        self.play(FadeIn(once), *[Indicate(r, scale_factor=1.06, color=WHITE) for r in rows], run_time=0.9)
        self.ws, self.prefixes, self.w_labs = ws, prefixes, w_labs
        self.qm, self.km, self.vm = res
        self.res_labs, self.qkv_words, self.once = res_labs, words, once
        self.end_section()

    # 4. Scores = Q Kᵀ -----------------------------------------------------------------
    def s4_scores(self):
        self.section(4)
        qm, km = self.qm, self.km
        q_shift = np.array([-3.88 - qm.cx(0), 0.2 - qm.ry(0), 0])
        k_shift = np.array([(0.6 + 2 * CW - 1.5 * CW) - km.cx(0), 3.1 - km.ry(0), 0])
        q_lab, k_lab = self.res_labs[0], self.res_labs[1]
        keep = [qm, km, q_lab, k_lab, self.illus]
        q_rows = qm.row_labels(gap=LAB_GAP).shift(q_shift)
        late = lambda t: smooth(min(1.0, max(0.0, (t - 0.35) / 0.65)))  # move once the old stage has faded
        self.play(*[FadeOut(m, run_time=0.5) for m in self.on_stage(keep=keep)],
                  qm.animate(run_time=1.0, rate_func=late).shift(q_shift),
                  km.animate(run_time=1.0, rate_func=late).shift(k_shift),
                  FadeIn(q_rows, shift=q_shift * 0.3, run_time=1.0, rate_func=late),
                  q_lab.animate(run_time=1.0, rate_func=late).move_to([-3.88 + 1.5 * CW, 0.85, 0]),
                  k_lab.animate(run_time=1.0, rate_func=late).move_to([0.05, 2.1, 0]))
        self.at("query")
        self.play(Indicate(qm.cells, color=Q_COLOR, scale_factor=1.05), run_time=0.6)
        self.at("key")
        self.play(Indicate(km.cells, color=K_COLOR, scale_factor=1.05), run_time=0.6)

        form_s = "Q · Kᵀ = scores"
        form = Text(form_s, font_size=36, t2c={"Q": Q_COLOR, "K": K_COLOR, "ᵀ": K_COLOR}).move_to([-3.4, 2.45, 0])
        one = label("one matrix multiplication", font_size=24).move_to([-3.4, 3.2, 0])
        self.at("multiplication")
        self.play(FadeIn(one, shift=0.1 * DOWN), run_time=0.5)
        self.at("q")
        self.play(FadeIn(gslice(form, form_s, 0, 3)), run_time=0.3)

        # transpose K: cell (i, j) flies to (j, i)
        kt = Mat(K.T, 0.6, 2.75, color=K_COLOR, hw=HW)
        kt_lab = Text("Kᵀ", font_size=34, color=K_COLOR).move_to([-0.46, 2.0, 0])
        k_cols = kt.col_labels(gap=0.4)
        self.at("transposed")
        self.play(*[km.cells[i][j].animate(path_arc=-0.6).move_to(kt.cells[j][i]) for i in range(5) for j in range(4)],
                  Transform(km.brackets, kt.brackets), Transform(k_lab, kt_lab),
                  FadeIn(gslice(form, form_s, 4, 6)), run_time=1.0)
        self.remove(km)
        self.add(kt)
        self.play(FadeIn(k_cols, shift=0.1 * DOWN), run_time=0.35)

        sm = Mat(SCORES, 0.6, 0.2, hw=HW)
        s_lab = VGroup(Text("scores", font_size=28), label("5 × 5", font_size=24)).arrange(DOWN, buff=0.15)
        s_lab.move_to([6.1, -0.8, 0])
        self.at("result")
        self.play(Create(sm.brackets), FadeIn(gslice(form, form_s, 7, len(form_s))), FadeIn(s_lab), run_time=0.7)
        self.at("grid")
        self.play(LaggedStart(*[FadeIn(r, shift=0.15 * DOWN) for r in sm.cells], lag_ratio=0.35), run_time=1.3)
        self.add(sm)

        i, j = 2, 1
        r_band = VGroup(qm.row_band(i, Q_COLOR), sm.row_band(i, Q_COLOR))
        row_i = Text("row i", font_size=24, color=Q_COLOR).move_to([-6.1, qm.ry(i), 0])
        self.at("row")
        self.play(FadeIn(r_band), FadeIn(row_i), run_time=0.5)
        c_band = kt.col_band(j, K_COLOR, top=k_cols.get_top()[1] + 0.06)
        c_band2 = sm.col_band(j, K_COLOR)
        col_j = Text("column j", font_size=24, color=K_COLOR).move_to([sm.cx(j), -2.4, 0])
        cell_hl = sm.cell_box(i, j, pad=0.0, stroke_color=WHITE, stroke_width=4)
        self.at("column")
        self.play(FadeIn(c_band), FadeIn(c_band2), FadeIn(col_j), Create(cell_hl), run_time=0.6)

        terms = " + ".join(self.prod(Q[i, k], K[j, k]) for k in range(4))
        dot_s = f"q_sat · k_cat = {terms} = {fmt(SCORES[i, j])}"
        dot = Text(dot_s, font_size=22, t2c={"q_sat": Q_COLOR, "k_cat": K_COLOR}).move_to([0, -2.95, 0])
        self.at("says")
        self.play(FadeIn(dot, shift=0.15 * UP), Indicate(sm.cells[i][j], color=WHITE, scale_factor=1.3),
                  run_time=0.9)
        self.at("question")
        self.play(Indicate(qm.cells[i], color=Q_COLOR, scale_factor=1.1), run_time=0.6)
        self.at("key")
        self.play(Indicate(VGroup(*[kt.cells[k][j] for k in range(4)]), color=K_COLOR, scale_factor=1.1),
                  run_time=0.5)
        self.sm, self.q_rows, self.k_cols = sm, q_rows, k_cols
        self.end_section()

    @staticmethod
    def prod(a, b):
        """a·b as text; negative factors in parentheses, written side by side."""
        sa, sb = fmt(a), fmt(b)
        if sa.startswith("−") and sb.startswith("−"):
            return f"({sa})({sb})"
        if sb.startswith("−"):
            return f"{sa}·({sb})"
        if sa.startswith("−"):
            return f"({sa})·{sb}"
        return f"{sa}·{sb}"

    # 5. Scale by √d ---------------------------------------------------------------------
    def s5_scale(self):
        self.section(5)
        sm, q_rows, k_cols = self.sm, self.q_rows, self.k_cols
        shift = np.array([-4.0 - sm.cx(0), 0.8 - sm.ry(0), 0])
        new_rows = q_rows.copy()
        for t, i in zip(new_rows, range(5)):
            t.move_to([-4.0 - HW - 0.16 - LAB_GAP, 0.8 - i * RH, 0])
        new_cols = k_cols.copy().shift(shift + np.array([0, (sm.brackets.get_top()[1] + 0.35) - k_cols.get_y(), 0]))
        heading = Text("÷ √4 = ÷ 2", font_size=40).move_to([3.8, 2.45, 0])
        self.at("divide")
        anims = [Transform(sm.cells[i][j], retext(sm.cells[i][j], fmt(SCALED[i, j]), shift=shift))
                 for i in range(5) for j in range(5)]
        self.play(*self.clear_anims(keep=[sm, q_rows, k_cols, self.illus]),
                  sm.brackets.animate.shift(shift), Transform(q_rows, new_rows), Transform(k_cols, new_cols),
                  *anims, FadeIn(heading, shift=0.2 * DOWN), run_time=1.2)
        why = label("√(key size) = √d = √4 = 2", font_size=24).move_to([3.8, 1.8, 0])
        self.at("size")
        self.play(FadeIn(why), run_time=0.6)
        grow = label("without it, scores grow with d", font_size=22).move_to([3.8, 1.25, 0])
        self.at("grow")
        self.play(FadeIn(grow), run_time=0.6)

        # inset: softmax of row "the", unscaled (sharp) vs scaled (softer)
        frame = RoundedRectangle(corner_radius=0.15, width=5.2, height=3.5, stroke_color=GREY_B, stroke_width=2)
        frame.move_to([3.95, -1.15, 0])
        title = label('softmax of row "the"', font_size=22, color=WHITE).move_to([3.95, 0.3, 0])
        base = -2.0

        def chart(probs, cx, color):
            bars = VGroup()
            for k, p in enumerate(probs):
                b = Rectangle(width=0.26, height=max(1.7 * p, 0.02), stroke_width=0, fill_color=color,
                              fill_opacity=0.9)
                b.move_to([cx + (k - 2) * 0.38, base, 0], aligned_edge=DOWN)
                bars.add(b)
            axis = Line([cx - 1.05, base, 0], [cx + 1.05, base, 0], color=GREY_B, stroke_width=2)
            top = int(np.argmax(probs))
            val = Text(fmt(probs[top]), font=MONO, font_size=20, color=color).next_to(bars[top], UP, buff=0.08)
            return VGroup(axis, bars, val)

        left = chart(RAW_THE, 2.65, RED_C)
        right = chart(SCALED_THE, 5.25, W_COLOR)
        l_cap = label("unscaled", font_size=22).move_to([2.65, base - 0.3, 0])
        r_cap = label("÷ 2", font_size=22).move_to([5.25, base - 0.3, 0])
        row_hl = sm.row_band(4, WHITE, fill_opacity=0.08, stroke_width=2)
        self.at("extreme")
        self.play(Create(frame), FadeIn(title), FadeIn(left), FadeIn(right), FadeIn(l_cap), FadeIn(r_cap),
                  FadeIn(row_hl), run_time=0.9)
        peak = left[1][int(np.argmax(RAW_THE))]
        sharp = Text("too sharp", font_size=22, color=RED_C)
        sharp.move_to(peak.get_corner(UR) + np.array([0.12 + sharp.width / 2, -0.45, 0]))
        self.at("struggles")
        self.play(FadeIn(sharp, shift=0.1 * UP), Indicate(peak, color=RED, scale_factor=1.1), run_time=0.6)
        self.end_section()

    # 6. Causal mask ----------------------------------------------------------------------
    def s6_mask(self):
        self.section(6)
        sm = self.sm
        heading = Text("causal mask", font_size=38).move_to([3.8, 2.45, 0])
        self.play(*self.clear_anims(keep=[sm, self.q_rows, self.k_cols, self.illus]),
                  FadeIn(heading, shift=0.2 * DOWN), run_time=0.8)
        shade = VGroup(*[sm.cell_box(i, j, pad=0.0, stroke_width=0, fill_color=RED_E, fill_opacity=0.35)
                         for i in range(5) for j in range(5) if j > i]).set_z_index(-1)
        future = label("no looking at later tokens", font_size=24).move_to([3.8, 1.75, 0])
        self.at("future")
        self.play(FadeIn(shade), FadeIn(future), run_time=0.8)
        diag = VGroup(*[sm.cell_box(i, i, pad=0.02, stroke_color=WHITE, stroke_width=3) for i in range(5)])
        diag_lab = label("above the diagonal", font_size=24).move_to([3.8, 1.2, 0])
        self.at("diagonal")
        self.play(LaggedStart(*[Create(b) for b in diag], lag_ratio=0.15), FadeIn(diag_lab), run_time=0.7)
        inf_lab = Text("→ −∞", font_size=30, color=RED_B).next_to(diag_lab, RIGHT, buff=0.2)
        self.at("infinity")
        self.play(*[Transform(sm.cells[i][j], retext(sm.cells[i][j], "−∞", GREY_B, size=26))
                    for i in range(5) for j in range(5) if j > i], FadeIn(inf_lab), run_time=0.9)
        after = label("after softmax:", font_size=24).move_to([3.1, -0.3, 0])
        arrow = Text("−∞  →  0", font_size=34).next_to(after, RIGHT, buff=0.3)
        arrow[-1].set_color(W_COLOR)
        self.at("zero")
        self.play(FadeIn(after), FadeIn(arrow, shift=0.2 * LEFT), run_time=0.6)
        self.shade, self.diag = shade, diag
        self.end_section()

    # 7. Softmax -----------------------------------------------------------------------------
    def s7_softmax(self):
        self.section(7)
        sm = self.sm
        heading = Text("softmax, row by row", font_size=38).move_to([3.8, 2.45, 0])
        self.play(*self.clear_anims(keep=[sm, self.q_rows, self.k_cols, self.illus, self.shade]),
                  FadeIn(heading, shift=0.2 * DOWN), run_time=0.8)
        rule = Text("weight = exp(score) ÷ row total", font_size=26).move_to([3.8, 1.7, 0])
        self.at("weights")
        self.play(FadeIn(rule), run_time=0.6)

        heat = VGroup()
        for i in range(5):
            for j in range(5):
                w = WEIGHTS[i, j]
                heat.add(sm.cell_box(i, j, pad=0.03, stroke_width=0, fill_color=W_COLOR,
                                     fill_opacity=0.0 if j > i else 0.12 + 0.68 * w))
        heat.set_z_index(-1)
        self.heat = heat
        self.at("exponentiate")
        row_anims, heat_rows = [], [VGroup(*heat[5 * i:5 * i + 5]) for i in range(5)]
        for i in range(5):
            cells = [Transform(sm.cells[i][j], retext(sm.cells[i][j], fmt(WEIGHTS[i, j]),
                                                      GREY_D if j > i else WHITE)) for j in range(5)]
            row_anims.append(AnimationGroup(*cells, FadeIn(heat_rows[i])))
        self.play(LaggedStart(*row_anims, lag_ratio=0.3), FadeOut(self.shade), run_time=2.2)
        self.remove(*heat_rows)
        self.add(heat)

        sums = VGroup(*[Text("Σ = 1.00", font=MONO, font_size=26, color=W_COLOR).scale(22 / 26)
                        .move_to([sm.brackets[1].get_right()[0] + 0.3, sm.ry(i), 0], aligned_edge=LEFT)
                        for i in range(5)])
        for i, s in enumerate(sums):  # the check really is on the displayed values
            assert abs(sum(float(fmt(WEIGHTS[i, j])) for j in range(5)) - 1.0) < 1e-9
        self.at("total")
        self.play(LaggedStart(*[FadeIn(s, shift=0.2 * LEFT) for s in sums], lag_ratio=0.15), run_time=1.0)
        pos = label("every weight ≥ 0", font_size=24).next_to(sums, DOWN, buff=0.4, aligned_edge=LEFT)
        self.at("positive")
        self.play(FadeIn(pos), run_time=0.5)
        self.at("one")
        self.play(Indicate(sums, color=YELLOW, scale_factor=1.1), run_time=0.6)
        self.end_section()

    # 8. Weighted sum of values -------------------------------------------------------------
    def s8_weighted_sum(self):
        self.section(8)
        sm, heat = self.sm, self.heat
        shift = np.array([-5.0 - sm.cx(0), 0.1 - sm.ry(0), 0])
        grid = VGroup(sm, heat, self.q_rows, self.k_cols)
        w_lab = Text("weights", font_size=28, color=W_COLOR).move_to([sm.cx(2) + shift[0], 1.25, 0])
        self.play(*self.clear_anims(keep=[grid, self.illus]), grid.animate.shift(shift), FadeIn(w_lab),
                  run_time=0.8)

        vm = Mat(V, 0.5, 3.2, color=V_COLOR, hw=HW)
        v_rows = vm.row_labels(gap=LAB_GAP)
        v_lab = Text("V", font_size=36, color=V_COLOR).move_to([-1.55, 2.2, 0])
        self.at("multiply")
        self.play(FadeIn(vm, shift=0.3 * LEFT), FadeIn(v_rows), FadeIn(v_lab), run_time=0.8)
        om = Mat(OUT, 0.5, 0.1, hw=HW)
        o_lab = VGroup(Text("output", font_size=28), label("weights · V", font_size=22)).arrange(DOWN, buff=0.15)
        o_lab.move_to([5.45, -0.8, 0])
        self.at("v")
        self.play(Create(om.brackets), FadeIn(o_lab), run_time=0.6)
        self.at("output")
        self.play(LaggedStart(*[FadeIn(r, shift=0.15 * DOWN) for r in om.cells], lag_ratio=0.3), run_time=1.0)
        self.add(om)

        def blend(i):
            bands = VGroup(sm.row_band(i, W_COLOR, fill_opacity=0.0), om.row_band(i, WHITE, fill_opacity=0.08))
            for k in range(i + 1):
                w = WEIGHTS[i, k]
                bands.add(vm.row_band(k, V_COLOR, fill_opacity=0.1 + 0.4 * w, stroke_width=1 + 4 * w))
            return bands

        i = 2
        terms = " + ".join(f"{fmt(WEIGHTS[i, k])}·v_{WORDS[k]}" for k in range(i + 1))
        eq_s = f"out_{WORDS[i]} = {terms}"
        t2c = {f"v_{WORDS[k]}": V_COLOR for k in range(i + 1)}
        t2c.update({fmt(WEIGHTS[i, k]): W_COLOR for k in range(i + 1)})
        eq = Text(eq_s, font_size=28, t2c=t2c).move_to([-0.5, -2.85, 0])
        bands = blend(i)
        self.at("blend")
        self.play(FadeIn(bands), FadeIn(eq, shift=0.15 * UP), run_time=0.9)

        eq0_s = "out_The = 1.00·v_The = v_The"
        eq0 = Text(eq0_s, font_size=28, t2c={"1.00": W_COLOR, "v_The": V_COLOR}).move_to([-0.5, -2.85, 0])
        bands0 = blend(0)
        self.at("first")
        self.play(ReplacementTransform(bands, bands0), FadeOut(eq, shift=0.15 * UP), FadeIn(eq0, shift=0.15 * UP),
                  run_time=0.8)
        self.at("itself")
        self.play(Indicate(sm.cells[0][0], color=YELLOW, scale_factor=1.4), run_time=0.6)
        same = Text("= v_The", font_size=26, color=V_COLOR).next_to(om.brackets, RIGHT, buff=0.25).match_y(om.cells[0])
        self.at("value")
        # V's first row flies onto output row 1: the old output row fades out first, then the copy lands
        # exactly on it (same text, same place) and is swapped for the real row, so nothing is double-printed
        flyer, landing = vm.cells[0].copy(), om.cells[0].copy()
        self.play(om.cells[0].animate(rate_func=lambda t: min(1.0, 4 * t)).set_opacity(0),
                  Transform(flyer, landing), FadeIn(same), run_time=0.7)
        om.cells[0].set_opacity(1)
        self.remove(flyer)
        self.play(Indicate(om.cells[0], color=V_COLOR, scale_factor=1.1), run_time=0.4)
        self.end_section()

    # 9. The whole formula ---------------------------------------------------------------------
    def s9_formula(self):
        self.section(9)
        self.play(*self.clear_anims(), run_time=0.6)
        fs = "Attention(Q, K, V) = softmax(Q Kᵀ / √d) · V"
        f = Text(fs, font_size=40, t2c={"Q": Q_COLOR, "K": K_COLOR, "ᵀ": K_COLOR, "V": V_COLOR})
        f.move_to([0, 2.55, 0])
        i_sm = fs.index("softmax")
        i_q = fs.index("Q", i_sm)
        i_k = fs.index("Kᵀ")
        i_sl = fs.index("/")
        i_rt = fs.index("√d")
        i_cl = fs.index(")", i_rt)
        i_dot = fs.index("·", i_cl)
        parts = [gslice(f, fs, a, b) for a, b in [(0, i_sm), (i_sm, i_sm + 8), (i_cl, i_cl + 1), (i_q, i_q + 1),
                                                  (i_k, i_k + 2), (i_sl, i_sl + 1), (i_rt, i_rt + 2),
                                                  (i_dot, len(fs))]]
        self.at("one")
        self.play(FadeIn(parts[0], shift=0.2 * DOWN), run_time=0.7)
        self.at("softmax")
        self.play(FadeIn(parts[1]), FadeIn(parts[2]), run_time=0.4)
        self.at("q")
        self.play(FadeIn(parts[3], scale=1.3), run_time=0.3)
        self.at("transposed")
        self.play(FadeIn(parts[4], scale=1.3), run_time=0.4)
        self.at("over")
        self.play(FadeIn(parts[5]), run_time=0.3)
        self.at("root")
        self.play(FadeIn(parts[6], scale=1.3), run_time=0.4)
        self.at("v")
        self.play(FadeIn(parts[7], scale=1.3), run_time=0.4)
        self.remove(*parts)
        self.add(f)

        # every token in parallel: all rows of the weights light at once
        sq = 0.46
        heat = VGroup()
        for i in range(5):
            for j in range(5):
                heat.add(Square(side_length=sq - 0.04, stroke_width=1, stroke_color=GREY_D, fill_color=W_COLOR,
                                fill_opacity=0.0 if j > i else 0.12 + 0.68 * WEIGHTS[i, j])
                         .move_to([-4.3 + (j - 2) * sq, -0.7 - (i - 2) * sq, 0]))
        labs = VGroup(*[small_token(w).move_to([-4.3 - 2.5 * sq - 0.5, -0.7 - (i - 2) * sq, 0])
                        for i, w in enumerate(WORDS)])
        self.at("every")
        self.play(FadeIn(heat), FadeIn(labs), run_time=0.6)
        row_boxes = VGroup(*[Rectangle(width=5 * sq + 0.06, height=sq, stroke_color=YELLOW, stroke_width=3)
                             .move_to([-4.3, -0.7 - (i - 2) * sq, 0]) for i in range(5)])
        par = label("every row at once", font_size=24, color=YELLOW).move_to([-4.6, -2.3, 0])
        self.at("parallel")
        self.play(*[Create(b) for b in row_boxes], FadeIn(par), run_time=0.6)

        badge_specs = [(sub_sym("X · W", "Q", WHITE, 26), Q_COLOR), (sub_sym("X · W", "K", WHITE, 26), K_COLOR),
                       (sub_sym("X · W", "V", WHITE, 26), V_COLOR),
                       (Text("Q · Kᵀ", font_size=26), WHITE), (Text("weights · V", font_size=26), W_COLOR)]
        badges = VGroup()
        for k, (content, col) in enumerate(badge_specs):
            box = RoundedRectangle(corner_radius=0.1, width=2.3, height=0.5, stroke_color=col, stroke_width=2.5,
                                   fill_color=col, fill_opacity=0.12)
            box.move_to([0.3, 0.55 - k * 0.62, 0])
            badges.add(VGroup(box, content.move_to(box)))
        hand = label("a handful of matrix multiplications", font_size=22).move_to([0.3, 1.27, 0])
        self.at("handful")
        self.play(FadeIn(hand), LaggedStart(*[FadeIn(b, shift=0.2 * LEFT) for b in badges], lag_ratio=0.2),
                  run_time=1.1)
        self.at("transformers")
        self.play(Indicate(f, color=WHITE, scale_factor=1.04), run_time=0.8)
        gpu = gpu_icon().move_to([4.5, -0.7, 0])
        self.at("gpus")
        self.play(GrowFromCenter(gpu), *[Indicate(b[0], scale_factor=1.05) for b in badges], run_time=0.8)
        self.end_section()

    # 10. Code -----------------------------------------------------------------------------------
    def s10_code(self):
        self.section(10)
        code, hl = code_panel(CODE, 20)
        grp = VGroup(code, hl)
        grp.scale(12.4 / code.width).move_to([0, 0.1, 0])
        hl.match_y(code.line_numbers[1])
        self.play(*self.clear_anims(), FadeIn(code, shift=0.3 * UP), run_time=0.8)
        self.at("project")
        self.play(Create(hl), run_time=0.4)
        for cue, line in [("score", 2), ("mask", 5), ("softmax", 7), ("mix", 8)]:
            self.at(cue)
            self.play(highlight(hl, code, line), run_time=0.4)
        self.end_section()

    # 11. Outro --------------------------------------------------------------------------------------
    def s11_outro(self):
        self.section(11)
        self.play(*self.clear_anims(), run_time=0.6)
        head = head_icon().move_to([0, -1.1, 0])
        anchor = head[1].get_center()
        first = thought(anchor, [1.9, 1.0, 0], YELLOW)
        cap = label("one layer: one kind of question", font_size=26).move_to([0, -2.9, 0])
        self.at("question")
        self.play(FadeIn(head, shift=0.2 * UP), FadeIn(first, scale=0.6), FadeIn(cap), run_time=0.8)
        more = VGroup(thought(anchor, [-1.9, 1.0, 0], TEAL), thought(anchor, [-3.0, -0.6, 0], ORANGE),
                      thought(anchor, [3.0, -0.6, 0], PURPLE_B), thought(anchor, [0.0, 1.9, 0], GREEN))
        cap2 = label("several things at once?", font_size=26).move_to(cap)
        self.at("several")
        self.play(LaggedStart(*[FadeIn(b, scale=0.6) for b in more], lag_ratio=0.25),
                  ReplacementTransform(cap, cap2), run_time=1.1)
        self.at("next")
        self.play(*self.clear_anims(), run_time=0.6)
        card = next_up_card(NEXT)
        self.at("attention")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.9)
        self.end_section()
