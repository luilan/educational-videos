"""Video 10 — From Vectors Back to Words.

Render from the repo root:  ./render.sh how-llms-work v10
"""
import numpy as np
from manim import *

from common import code_panel, finish, highlight, next_up_card, token, token_row
from intro import play_token_intro
from v10_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

MONO = "DejaVu Sans Mono"
MODEL_COLOR = PURPLE_B
POS_COLOR, NEG_COLOR = BLUE_C, RED_C

# ---------------------------------------------------------------------------- toy example
WORDS = ["mat", "floor", "sofa", "bed", "roof", "table", "car", "banana"]
LOGITS = np.array([3.0, 2.4, 1.9, 1.6, 1.1, 0.8, -0.5, -2.0])


def softmax(x, T=1.0):
    z = np.asarray(x, dtype=float) / T
    e = np.exp(z - z.max())
    return e / e.sum()


P1 = softmax(LOGITS)                                  # T = 1
TOP_K = 3
P_TOPK = P1[:TOP_K] / P1[:TOP_K].sum()                # renormalised top-3
CUM = np.cumsum(P1)
N_TOPP = int(np.searchsorted(CUM, 0.9) + 1)           # smallest top set with cumulative >= 90 % -> 5
P_TOPP = P1[:N_TOPP] / P1[:N_TOPP].sum()

# probability chart layout (sections 4-7)
ROW_H, BAR_H = 0.52, 0.36
PSCALE = 6.0                  # scene units for 100 %
CHART_X, CHART_Y = -3.2, 1.8  # bar start x and first-row y of the main chart
CUM_RIGHT = 3.7               # right edge of the cumulative column

# section 1-2 vectors (6 illustrative dimensions)
D = 6
_rng = np.random.default_rng(10)
TOKEN_VECS = [np.clip(_rng.normal(0, 0.55, D), -0.95, 0.95) for _ in range(5)]
H_RAW = TOKEN_VECS[-1]
H_LN = np.clip((H_RAW - H_RAW.mean()) / H_RAW.std() * 0.5, -0.95, 0.95)
W_COLS = {  # unembedding columns: " mat" lines up with h, " banana" points away
    " the": np.clip(_rng.normal(0, 0.5, D), -0.95, 0.95),
    " cat": np.clip(_rng.normal(0, 0.5, D), -0.95, 0.95),
    " mat": np.clip(0.9 * H_LN + _rng.normal(0, 0.12, D), -0.95, 0.95),
    " floor": np.clip(0.6 * H_LN + _rng.normal(0, 0.3, D), -0.95, 0.95),
    " banana": np.clip(-0.8 * H_LN + _rng.normal(0, 0.12, D), -0.95, 0.95),
}
COL_NAMES = [" the", " cat", " mat", " floor", None, " banana"]   # None = the "⋯" gap
COL_LOGITS = {" the": -1.3, " cat": 0.4, " mat": 3.0, " floor": 2.4, " banana": -2.0}
CELL_W, CELL_H, CELL_GAP = 0.42, 0.34, 0.06

CODE = """def next_token(h, W_unembed, temperature=0.8, top_k=50):
    logits = h @ W_unembed / temperature    # a score per token
    top = np.argsort(logits)[-top_k:]       # keep the k best
    p = np.exp(logits[top] - logits[top].max())
    p /= p.sum()                            # softmax
    return np.random.choice(top, p=p)       # sample one"""


# ---------------------------------------------------------------------------- helpers
def fmt(v, nd=1):
    s = f"{v:.{nd}f}"
    return s.replace("-", "−")


def pct(p):
    return "<0.1%" if p < 0.0005 else f"{100 * p:.1f}%"


def caption(text, font_size=20):
    return Text(text, font_size=font_size, color=GREY_B).to_corner(DR, buff=0.3)


def value_color(v):
    v = max(-1.0, min(1.0, float(v)))
    return interpolate_color(GREY_E, POS_COLOR if v >= 0 else NEG_COLOR, abs(v))


def vec_cells(values, direction=DOWN, stroke=GREY_C):
    """A vector as a strip of coloured cells (blue = positive, red = negative)."""
    cells = VGroup(*[Rectangle(width=CELL_W, height=CELL_H, stroke_color=stroke, stroke_width=1.5,
                               fill_color=value_color(v), fill_opacity=1) for v in values])
    return cells.arrange(direction, buff=CELL_GAP)


def recolor(cells, values):
    return [c.animate.set_fill(value_color(v), 1) for c, v in zip(cells, values)]


def brackets(mob, buff=0.15, color=GREY_B):
    top, bot = mob.get_top()[1] + buff, mob.get_bottom()[1] - buff
    xl, xr = mob.get_left()[0] - buff, mob.get_right()[0] + buff
    left = VMobject().set_points_as_corners([[xl + 0.15, top, 0], [xl, top, 0], [xl, bot, 0], [xl + 0.15, bot, 0]])
    right = VMobject().set_points_as_corners([[xr - 0.15, top, 0], [xr, top, 0], [xr, bot, 0], [xr - 0.15, bot, 0]])
    return VGroup(left, right).set_stroke(color, 2.5)


def row_y(i, y0=CHART_Y):
    return y0 - i * ROW_H


def word_labels(right_x, y0=CHART_Y, font_size=24):
    return VGroup(*[Text(w, font_size=font_size).move_to([right_x, row_y(i, y0), 0], aligned_edge=RIGHT)
                    for i, w in enumerate(WORDS)])


def prob_bar(p, i, x0=CHART_X, y0=CHART_Y, color=TEAL):
    bar = Rectangle(width=max(p * PSCALE, 0.02), height=BAR_H, stroke_width=0, fill_color=color, fill_opacity=0.85)
    return bar.move_to([x0, row_y(i, y0), 0], aligned_edge=LEFT)


def prob_val(p, bar, color=GREY_A):
    return Text(pct(p), font_size=22, color=color).next_to(bar, RIGHT, buff=0.15)


def prob_bars(probs, x0=CHART_X, y0=CHART_Y, colors=None):
    colors = colors or [TEAL] * len(probs)
    bars = VGroup(*[prob_bar(p, i, x0, y0, c) for i, (p, c) in enumerate(zip(probs, colors))])
    vals = VGroup(*[prob_val(p, b) for p, b in zip(probs, bars)])
    return bars, vals


def cutoff_line(after_row, label_text, x_left=-5.9, x_right=4.0):
    y = row_y(after_row) - ROW_H / 2
    line = DashedLine([x_left, y, 0], [x_right, y, 0], dash_length=0.12, color=RED_C, stroke_width=3)
    label = Text(label_text, font_size=24, color=RED_C).next_to(line, RIGHT, buff=0.2)
    return VGroup(line, label)


def superscript(base, sup, font_size=30, color=WHITE):
    b = Text(base, font_size=font_size, color=color)
    s = Text(sup, font_size=int(font_size * 0.7), color=color).next_to(b, UR, buff=0.03).shift(0.12 * DOWN)
    return VGroup(b, s)


def small_box(text, color=GREY_B, font_size=22):
    label = Text(text, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.12, width=label.width + 0.4, height=0.7,
                           stroke_color=color, fill_color=color, fill_opacity=0.15)
    return VGroup(box, label.move_to(box))


def matrix_grid(values, cell=0.3, noise=False):
    """A square grid of cells; noise=True draws random grey values."""
    n = values.shape[0]
    grid = VGroup()
    for r in range(n):
        for c in range(n):
            v = values[r, c]
            color = interpolate_color(BLACK, WHITE, v) if noise else value_color(v)
            grid.add(Square(side_length=cell, stroke_color=GREY_D, stroke_width=1,
                            fill_color=color, fill_opacity=1))
    return grid.arrange_in_grid(n, n, buff=0)


class UnembedVideo(VoicedScene):
    VIDEO = "v10"

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

    def clear(self, run_time=0.45, keep=()):
        anims = self.clear_anims(keep)
        if anims:
            self.play(*anims, run_time=run_time)

    def construct(self):
        play_token_intro(self, TITLE, 10, TAGLINE)
        self.s1_last_vector()
        self.s2_unembedding()
        self.s3_dot_products()
        self.s4_softmax()
        self.s5_greedy()
        self.s6_temperature()
        self.s7_topk_topp()
        self.s8_training()
        self.s9_code()
        self.s10_outro()
        finish(self)

    # 1. The last vector ---------------------------------------------------------------
    def s1_last_vector(self):
        self.section(1)
        row = token_row(["The", "cat", "sat", "on", "the"]).move_to([-0.6, -2.75, 0])
        block = RoundedRectangle(corner_radius=0.15, width=row.width + 1.0, height=0.8, stroke_color=MODEL_COLOR,
                                 fill_color=MODEL_COLOR, fill_opacity=0.25).move_to([row.get_x(), -1.5, 0])
        block_lab = Text("last transformer block", font_size=24).move_to(block)
        ins = VGroup(*[Line(t.get_top(), [t.get_x(), block.get_bottom()[1], 0], color=GREY_C, stroke_width=3)
                       for t in row])
        cols = VGroup(*[vec_cells(v).move_to([t.get_x(), 0.55, 0]) for t, v in zip(row, TOKEN_VECS)])
        outs = VGroup(*[Arrow([c.get_x(), block.get_top()[1], 0], c.get_bottom(), buff=0.06, color=GREY_B,
                              stroke_width=3, max_tip_length_to_length_ratio=0.35) for c in cols])
        side = Text("one vector\nper token", font_size=24, color=GREY_B, line_spacing=0.8)
        side.next_to(cols, LEFT, buff=0.8)
        slot = DashedVMobject(RoundedRectangle(corner_radius=0.12, width=0.9, height=0.65, color=YELLOW))
        slot.next_to(row, RIGHT, buff=0.3)
        qmark = Text("?", font_size=32, color=YELLOW).move_to(slot)

        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * UP) for t in row], lag_ratio=0.12),
                  FadeIn(block), FadeIn(block_lab), Create(ins), run_time=1.0)
        self.at("vector")
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(c, shift=0.3 * UP)) for a, c in zip(outs, cols)],
                              lag_ratio=0.15), FadeIn(side), run_time=1.1)
        self.at("predict")
        self.play(Create(slot), FadeIn(qmark), run_time=0.6)
        self.at("the")
        self.play(LaggedStart(*[Indicate(t, color=WHITE, scale_factor=1.08) for t in row], lag_ratio=0.18),
                  run_time=1.0)
        self.at("last")
        last, last_tok = cols[-1], row[-1]
        glow = SurroundingRectangle(last, color=YELLOW, buff=0.1, stroke_width=4)
        dim = VGroup(*cols[:-1], *outs[:-1], *row[:-1], *ins[:-1])
        self.play(dim.animate.set_opacity(0.25), side.animate.set_opacity(0.4), Create(glow),
                  last.animate.set_stroke(YELLOW, 2), last_tok[0].animate.set_stroke(YELLOW).set_fill(YELLOW, 0.3),
                  run_time=0.6)
        self.at("word")
        self.at("the")
        self.play(VGroup(last, glow).animate.shift(0.55 * UP).scale(1.08), run_time=0.6)
        self.h_vec, self.h_glow = last, glow
        self.end_section()

    # 2. Unembedding ---------------------------------------------------------------
    def s2_unembedding(self):
        self.section(2)
        h, glow = self.h_vec, self.h_glow
        y_mid = 0.25
        ln = RoundedRectangle(corner_radius=0.12, width=0.75, height=h.height / 1.08 + 0.5, stroke_color=GREEN_C,
                              fill_color=GREEN_E, fill_opacity=0.85).move_to([-5.7, y_mid, 0]).set_z_index(3)
        ln_lab = Text("LN", font_size=28, weight=BOLD).move_to(ln).set_z_index(4)
        note = caption("illustrative values")
        self.play(*self.clear_anims(keep=(h, glow)), FadeOut(glow),
                  h.animate.scale(1 / 1.08).move_to([-6.5, y_mid, 0]), run_time=0.8)
        self.play(FadeIn(ln), FadeIn(ln_lab), FadeIn(note), run_time=0.4)
        self.at("norm")
        self.play(h.animate.move_to([-4.85, y_mid, 0]), run_time=0.7)
        h_lab = Text("h", font_size=30, color=YELLOW).next_to(h, UP, buff=0.3)
        self.play(*recolor(h, H_LN), FadeIn(h_lab), Indicate(ln_lab, color=WHITE), run_time=0.5)

        # the unembedding matrix: one column per vocabulary token
        col_x = [-3.45 + 1.2 * j for j in range(len(COL_NAMES))]
        cols, col_labels = VGroup(), VGroup()
        for x, name in zip(col_x, COL_NAMES):
            if name is None:
                cols.add(Text("⋯", font_size=32, color=GREY_B).move_to([x, y_mid, 0]))
                col_labels.add(Text("⋯", font_size=24, color=GREY_B).move_to([x, 2.0, 0]))
            else:
                cols.add(vec_cells(W_COLS[name]).move_to([x, y_mid, 0]))
                col_labels.add(Text(f'"{name}"', font=MONO, font_size=20, color=GREY_A).move_to([x, 2.0, 0]))
        w_br = brackets(cols, buff=0.22)
        w_title = Text("unembedding matrix", font_size=28).move_to([np.mean(col_x), 2.7, 0])
        times = Text("×", font_size=36, color=GREY_B).move_to([-4.3, y_mid, 0])
        self.at("unembedding")
        self.play(FadeIn(times), Create(w_br), Write(w_title),
                  LaggedStart(*[FadeIn(VGroup(c, l), shift=0.2 * DOWN) for c, l in zip(cols, col_labels)],
                              lag_ratio=0.12), run_time=1.2)
        mat_i = COL_NAMES.index(" mat")
        col_box = SurroundingRectangle(VGroup(cols[mat_i], col_labels[mat_i]), color=YELLOW, buff=0.1)
        self.at("column")
        self.play(Create(col_box), run_time=0.5)
        self.at("vocabulary")
        self.play(FadeOut(col_box), LaggedStart(*[Indicate(l, color=YELLOW, scale_factor=1.12) for l in col_labels],
                                                lag_ratio=0.12), run_time=0.9)

        # one score per column
        scores = VGroup()
        for x, name in zip(col_x, COL_NAMES):
            txt = "⋯" if name is None else fmt(COL_LOGITS[name])
            scores.add(Text(txt, font=MONO, font_size=24, color=YELLOW if name == " mat" else WHITE)
                       .move_to([x, -1.4, 0]))
        self.at("score")
        self.play(LaggedStart(*[AnimationGroup(Indicate(c, color=YELLOW, scale_factor=1.06), FadeIn(s, shift=0.3 * DOWN))
                                for c, s in zip(cols, scores)], lag_ratio=0.2), run_time=1.4)
        brace = Brace(VGroup(scores, cols), DOWN, color=GREY_B).set_y(-1.85)
        count = Text("50,257", font_size=28).next_to(brace, DOWN, buff=0.15)
        count2 = Text("50,257 logits", font_size=28).next_to(brace, DOWN, buff=0.15)
        self.at("50")
        self.play(GrowFromCenter(brace), FadeIn(count, shift=0.2 * UP), run_time=0.6)
        logits_lab = Text("logits", font_size=28, color=YELLOW).move_to([-4.85, -1.4, 0])
        self.at("logits")
        self.play(FadeIn(logits_lab, shift=0.2 * RIGHT), ReplacementTransform(count, count2), run_time=0.6)

        # tied weights: the same matrix as the embedding table
        w_core = VGroup(cols, w_br)
        w_core.generate_target()
        w_core.target.scale(0.62).move_to([-3.6, 0.45, 0])
        w_title2 = Text("unembedding matrix", font_size=26).next_to(w_core.target, UP, buff=0.35)
        gpt = Text("GPT-2", font_size=26, color=GREY_B).to_corner(DL, buff=0.35)
        self.at("gpt")
        self.play(FadeOut(VGroup(h, h_lab, ln, ln_lab, times, scores, brace, count2, logits_lab, col_labels)),
                  MoveToTarget(w_core), ReplacementTransform(w_title, w_title2), FadeIn(gpt), run_time=0.9)
        e_rows = VGroup()
        for k, name in enumerate(COL_NAMES):
            y = 1.55 - 0.44 * k
            if name is None:
                lab = Text("⋮", font_size=24, color=GREY_B)
                cells = Text("⋮", font_size=24, color=GREY_B)
            else:
                lab = Text(f'"{name}"', font=MONO, font_size=20, color=GREY_A)
                cells = vec_cells(W_COLS[name], direction=RIGHT)
            cells.move_to([3.65, y, 0])
            lab.move_to([2.0, y, 0], aligned_edge=RIGHT)
            e_rows.add(VGroup(lab, cells))
        e_br = brackets(VGroup(*[r[1] for r in e_rows]), buff=0.15)
        e_title = Text("embedding matrix E", font_size=28).next_to(e_br, UP, buff=0.3)
        eq = Text("=", font_size=48).move_to([-0.55, 0.45, 0])
        eq_note = Text("transposed", font_size=20, color=GREY_B).next_to(eq, DOWN, buff=0.15)
        self.at("embedding")
        self.play(FadeIn(eq), FadeIn(eq_note), Write(e_title), Create(e_br),
                  LaggedStart(*[FadeIn(r, shift=0.2 * LEFT) for r in e_rows], lag_ratio=0.1), run_time=0.9)
        same = Text("same matrix (tied weights)", font_size=30, color=YELLOW).move_to([0.3, -2.3, 0])
        self.at("reused")
        self.play(FadeIn(same, shift=0.2 * UP), Circumscribe(cols, color=YELLOW, run_time=0.7),
                  Circumscribe(e_rows, color=YELLOW, run_time=0.7), run_time=0.7)
        self.end_section()

    # 3. Logits as dot products ---------------------------------------------------------
    def s3_dot_products(self):
        self.section(3)
        self.clear(run_time=0.4)
        axes = Axes(x_range=[-1.4, 2.8, 1], y_range=[-1.0, 1.9, 1], x_length=4.2 * 1.4, y_length=2.9 * 1.4,
                    axis_config={"include_ticks": False, "stroke_color": GREY_D, "stroke_width": 2,
                                 "tip_width": 0.15, "tip_height": 0.15}).move_to([-3.4, -0.25, 0])
        origin = axes.c2p(0, 0)
        h_vec = (2.0, 1.0)
        dirs = {"mat": (1.1, 0.8), "floor": (0.6, 1.2), "banana": (-0.8, -0.4)}
        h_arrow = Arrow(origin, axes.c2p(*h_vec), buff=0, color=WHITE, stroke_width=6)
        h_lab = Text("h", font_size=30).next_to(h_arrow.get_end(), RIGHT, buff=0.12)
        inset_note = Text("illustrative 2-D picture", font_size=20, color=GREY_B).next_to(axes, DOWN, buff=0.2)
        self.play(Create(axes), GrowArrow(h_arrow), FadeIn(h_lab), FadeIn(inset_note), run_time=0.6)
        formula = Text("logit = h · w_token", font_size=32).move_to([0, 3.1, 0])
        formula[8:].set_color(TEAL_C)
        self.at("dot")
        self.play(Write(formula), run_time=0.8)
        self.at("final")
        self.play(Indicate(VGroup(h_arrow, h_lab), color=YELLOW, scale_factor=1.05), run_time=0.6)
        t_arrows, t_labs = {}, {}
        lab_dirs = {"mat": UP, "floor": UP, "banana": DL}
        for w, d in dirs.items():
            t_arrows[w] = Arrow(origin, axes.c2p(*d), buff=0, color=TEAL_C, stroke_width=5,
                                max_tip_length_to_length_ratio=0.2)
            t_labs[w] = Text(w, font_size=24, color=TEAL_C).next_to(t_arrows[w].get_end(), lab_dirs[w], buff=0.1)
        self.at("lines")
        self.play(*[GrowArrow(a) for a in t_arrows.values()], *[FadeIn(l) for l in t_labs.values()], run_time=0.8)

        # the resulting logits as bars (dot products of the 2-D vectors: 3.0, 2.4, -2.0)
        zero_y, scale = -0.5, 0.6
        bar_x = {"mat": 2.2, "floor": 3.7, "banana": 5.2}
        zero = Line([1.3, zero_y, 0], [6.1, zero_y, 0], color=GREY_B, stroke_width=2)
        header = Text("logits", font_size=28).move_to([3.7, 2.2, 0])
        bars, vals, names = {}, {}, {}
        for w, d in dirs.items():
            logit = float(np.dot(h_vec, d))
            bar = Rectangle(width=0.8, height=abs(logit) * scale, stroke_width=0,
                            fill_color=POS_COLOR if logit >= 0 else NEG_COLOR, fill_opacity=0.85)
            bar.move_to([bar_x[w], zero_y, 0], aligned_edge=DOWN if logit >= 0 else UP)
            bars[w] = bar
            vals[w] = Text(fmt(logit), font=MONO, font_size=24).next_to(bar, UP if logit >= 0 else DOWN, buff=0.12)
            names[w] = Text(w, font_size=24, color=TEAL_C).next_to(zero, DOWN if logit >= 0 else UP, buff=0.2).set_x(bar_x[w])
        self.at("direction")
        self.play(Create(zero), FadeIn(header), *[GrowFromEdge(bars[w], DOWN if w != "banana" else UP) for w in dirs],
                  *[FadeIn(vals[w]) for w in dirs], *[FadeIn(names[w]) for w in dirs], run_time=0.9)
        self.at("match")
        self.play(t_arrows["mat"].animate.set_color(YELLOW).set_stroke(width=8), t_labs["mat"].animate.set_color(YELLOW),
                  run_time=0.5)
        self.at("higher")
        self.play(bars["mat"].animate.set_fill(YELLOW, 0.9), names["mat"].animate.set_color(YELLOW),
                  Indicate(vals["mat"], color=YELLOW, scale_factor=1.3), run_time=0.6)
        self.end_section()

    # 4. Softmax ------------------------------------------------------------------------
    def s4_softmax(self):
        self.section(4)
        self.clear(run_time=0.4)
        zero_x, lscale = -3.3, 0.5
        l_labels = word_labels(-4.7)
        zero = Line([zero_x, 2.1, 0], [zero_x, -2.15, 0], color=GREY_B, stroke_width=2)
        zero_lab = Text("0", font_size=20, color=GREY_B).next_to(zero, DOWN, buff=0.1)
        l_header = Text("logits", font_size=28).move_to([-3.3, 2.6, 0])
        l_bars, l_vals = VGroup(), VGroup()
        for i, l in enumerate(LOGITS):
            bar = Rectangle(width=abs(l) * lscale, height=BAR_H, stroke_width=0,
                            fill_color=POS_COLOR if l >= 0 else NEG_COLOR, fill_opacity=0.85)
            bar.move_to([zero_x, row_y(i), 0], aligned_edge=LEFT if l >= 0 else RIGHT)
            l_bars.add(bar)
            x = max(bar.get_right()[0], zero_x) + 0.15
            l_vals.add(Text(fmt(l), font=MONO, font_size=22, color=GREY_A).move_to([x, row_y(i), 0], aligned_edge=LEFT))
        note = caption("toy vocabulary, illustrative logits")
        self.play(FadeIn(l_header), FadeIn(l_labels), Create(zero), FadeIn(zero_lab), FadeIn(note), run_time=0.6)
        pos = [i for i, l in enumerate(LOGITS) if l >= 0]
        neg = [i for i, l in enumerate(LOGITS) if l < 0]
        self.at("positive")
        self.play(LaggedStart(*[AnimationGroup(GrowFromEdge(l_bars[i], LEFT), FadeIn(l_vals[i])) for i in pos],
                              lag_ratio=0.12), run_time=0.7)
        self.at("negative")
        self.play(*[AnimationGroup(GrowFromEdge(l_bars[i], RIGHT), FadeIn(l_vals[i])) for i in neg], run_time=0.6)

        px0 = 2.4
        p_labels = word_labels(px0 - 0.2)
        p_header = Text("probabilities", font_size=28).move_to([px0 + 1.4, 2.6, 0])
        p_bars, p_vals = prob_bars(P1, x0=px0)
        arrow = Arrow([-0.95, 0.25, 0], [0.75, 0.25, 0], buff=0, color=GREY_B)
        sm_lab = Text("softmax", font_size=26, color=YELLOW).next_to(arrow, UP, buff=0.12)
        self.at("softmax")
        self.play(GrowArrow(arrow), FadeIn(sm_lab), FadeIn(p_header), FadeIn(p_labels),
                  LaggedStart(*[AnimationGroup(GrowFromEdge(b, LEFT), FadeIn(v)) for b, v in zip(p_bars, p_vals)],
                              lag_ratio=0.1), run_time=1.2)
        exp_lab = superscript("e", "logit", font_size=32).next_to(arrow, DOWN, buff=0.3)
        self.at("exponentiate")
        self.play(FadeIn(exp_lab, shift=0.2 * DOWN), run_time=0.5)
        div_lab = Text("÷ sum", font_size=28).next_to(exp_lab, DOWN, buff=0.3)
        self.at("total")
        self.play(FadeIn(div_lab, shift=0.2 * DOWN), run_time=0.5)
        self.at("probability")
        self.play(LaggedStart(*[Indicate(v, color=TEAL_A, scale_factor=1.15) for v in p_vals], lag_ratio=0.08),
                  run_time=0.8)
        total = Text("sum = 100%", font_size=28, color=YELLOW).move_to([px0 + 1.4, -2.55, 0])
        self.at("one")
        self.play(FadeIn(total, shift=0.2 * UP), run_time=0.5)
        self.chart = dict(labels=p_labels, header=p_header, bars=p_bars, vals=p_vals, note=note)
        self.s4_extra = VGroup(l_header, l_labels, zero, zero_lab, l_bars, l_vals, arrow, sm_lab, exp_lab, div_lab, total)
        self.end_section()

    # 5. Greedy ------------------------------------------------------------------------
    def s5_greedy(self):
        self.section(5)
        ch = self.chart
        moving = VGroup(ch["labels"], ch["header"], ch["bars"], ch["vals"])
        self.play(FadeOut(self.s4_extra), moving.animate.shift((CHART_X - 2.4) * RIGHT), run_time=0.9)
        mat_row = VGroup(ch["labels"][0], ch["bars"][0], ch["vals"][0])
        box = SurroundingRectangle(mat_row, color=YELLOW, buff=0.1)
        greedy = VGroup(Text("greedy:", font_size=24, color=GREY_B),
                        Text("always pick the max", font_size=28, color=YELLOW)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        greedy.next_to(box, RIGHT, buff=0.5).align_to(box, UP).shift(0.3 * UP)
        self.at("greedy")
        self.play(Create(box), FadeIn(greedy, shift=0.2 * LEFT), run_time=0.6)
        loop = Text("the cat sat on the mat. the cat sat on the mat. the cat sat on the mat.", font_size=24)
        loop.move_to([0, -2.75, 0])
        self.at("repetitive")
        self.play(AddTextLetterByLetter(loop, time_per_char=0.01), run_time=0.7)
        self.at("dull")
        self.play(loop.animate.set_color(GREY_D), run_time=0.4)
        self.s5_extra = VGroup(box, greedy, loop)
        self.end_section()

    # 6. Temperature -------------------------------------------------------------------
    def s6_temperature(self):
        self.section(6)
        ch = self.chart
        self.play(FadeOut(self.s5_extra), run_time=0.5)
        temp = ValueTracker(1.0)
        glow = ValueTracker(0.0)

        def live_chart():
            p = softmax(LOGITS, temp.get_value())
            g = glow.get_value()
            colors = [TEAL] * 6 + [interpolate_color(TEAL, YELLOW, g)] * 2
            bars, vals = prob_bars(p, colors=colors)
            return VGroup(bars, vals)

        live = always_redraw(live_chart)
        self.remove(*ch["bars"], *ch["vals"])
        self.add(live)
        readout = always_redraw(lambda: Text(f"T = {temp.get_value():.1f}", font_size=36, color=YELLOW)
                                .move_to([2.9, 1.55, 0], aligned_edge=LEFT))
        self.at("temperature")
        self.play(FadeIn(readout), run_time=0.5)
        formula = Text("softmax(logits / T)", font=MONO, font_size=28).move_to([4.2, 2.6, 0])
        self.at("divide")
        self.play(Write(formula), run_time=0.7)
        low = Text("low T → sharper", font_size=24, color=GREY_B).move_to([2.9, 0.75, 0], aligned_edge=LEFT)
        high = Text("high T → flatter", font_size=24, color=GREY_B).move_to([2.9, 0.75, 0], aligned_edge=LEFT)
        self.at("sharpens")
        self.play(temp.animate.set_value(0.5), FadeIn(low), run_time=1.3)
        self.at("flattens")
        self.play(temp.animate.set_value(1.4), FadeOut(low), FadeIn(high), run_time=0.9)
        rare = VGroup(ch["labels"][6], ch["labels"][7])
        self.at("rarer")
        self.play(temp.animate.set_value(2.0), glow.animate.set_value(1.0), rare.animate.set_color(YELLOW),
                  run_time=0.8)
        self.at("chance")
        self.play(Indicate(rare, color=YELLOW, scale_factor=1.15), run_time=0.6)
        self.s6_state = dict(temp=temp, glow=glow, live=live, readout=readout, extra=VGroup(formula, high))
        self.end_section()

    # 7. Top-k and top-p ---------------------------------------------------------------
    def s7_topk_topp(self):
        self.section(7)
        ch, st = self.chart, self.s6_state
        labels = ch["labels"]
        self.play(st["temp"].animate.set_value(1.0), st["glow"].animate.set_value(0.0),
                  labels[6].animate.set_color(WHITE), labels[7].animate.set_color(WHITE), run_time=0.8)
        st["readout"].clear_updaters()
        self.play(FadeOut(st["readout"]), FadeOut(st["extra"]), run_time=0.4)
        st["live"].clear_updaters()
        self.remove(st["live"], st["temp"], st["glow"])
        bars, vals = prob_bars(P1)
        bars, vals = list(bars), list(vals)
        self.add(*bars, *vals)

        def set_probs(probs, rows):
            anims = []
            for i, p in zip(rows, probs):
                nb, nv = prob_bar(p, i), prob_val(p, prob_bar(p, i))
                anims += [Transform(bars[i], nb), FadeTransform(vals[i], nv)]
                vals[i] = nv
            return anims

        def grey_rows(rows, color=GREY_D):
            return [m for i in rows for m in (labels[i].animate.set_color(color), bars[i].animate.set_fill(color, 0.85),
                                              vals[i].animate.set_color(color))]

        cut_rows = range(TOP_K, len(WORDS))
        topk = cutoff_line(TOP_K - 1, f"top-k (k = {TOP_K})")
        self.at("k")
        self.play(Create(topk[0]), FadeIn(topk[1]), *grey_rows(cut_rows), run_time=0.7)
        self.at("likely")
        self.play(*set_probs(P_TOPK, range(TOP_K)), run_time=0.7)
        self.at("p")
        restore = set_probs(P1, range(len(WORDS)))
        self.play(FadeOut(topk), *restore, *[labels[i].animate.set_color(WHITE) for i in cut_rows], run_time=0.6)

        cum_header = Text("cumulative", font_size=24, color=GREY_B).move_to([CUM_RIGHT - 0.55, CHART_Y + 0.65, 0])
        cums = [Text(f"{100 * c:.1f}%", font_size=22, color=GREY_A).move_to([CUM_RIGHT, row_y(i), 0], aligned_edge=RIGHT)
                for i, c in enumerate(CUM)]
        self.at("probabilities")
        self.play(FadeIn(cum_header), LaggedStart(*[FadeIn(c, shift=0.15 * DOWN) for c in cums], lag_ratio=0.12),
                  run_time=0.9)
        topp = cutoff_line(N_TOPP - 1, "top-p (p = 0.9)", x_right=CUM_RIGHT + 0.3)
        cut_rows = range(N_TOPP, len(WORDS))
        self.at("90")
        self.play(Create(topp[0]), FadeIn(topp[1]), cums[N_TOPP - 1].animate.set_color(YELLOW),
                  *grey_rows(cut_rows), *[cums[i].animate.set_color(GREY_D) for i in cut_rows], run_time=0.7)
        cut = VGroup(*[VGroup(labels[i], bars[i], vals[i], cums[i]) for i in cut_rows])
        self.at("cut")
        self.play(*[m.animate.set_color(RED_C) for i in cut_rows for m in (labels[i], vals[i], cums[i])],
                  *[bars[i].animate.set_fill(RED_C, 0.85) for i in cut_rows], run_time=0.2)
        self.play(LaggedStart(*[FadeOut(r, shift=1.2 * DOWN) for r in cut], lag_ratio=0.15), run_time=0.5)
        self.at("sample")
        self.play(*set_probs(P_TOPP, range(N_TOPP)), FadeOut(cum_header), *[FadeOut(c) for c in cums[:N_TOPP]],
                  run_time=0.7)
        self.end_section()

    # 8. Training uses every position ----------------------------------------------------
    def s8_training(self):
        self.section(8)
        self.clear(run_time=0.4)
        words = ["The", "cat", "sat", "on", "the", "mat"]
        row = token_row(words, buff=0.4).move_to([0.3, 1.3, 0])
        head = Text("during training", font_size=26, color=GREY_B).move_to([0.3, 2.6, 0])
        self.play(FadeIn(head), LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in row], lag_ratio=0.08), run_time=0.6)
        preds, arrows = VGroup(), VGroup()
        for i in range(5):
            p = token(words[i + 1], color=TEAL).move_to([row[i].get_x(), -0.65, 0])
            preds.add(p)
            arrows.add(Arrow(row[i].get_bottom(), p.get_top(), buff=0.12, color=GREY_B, stroke_width=4,
                             max_tip_length_to_length_ratio=0.2))
        pred_lab = Text("predicts", font_size=24, color=GREY_B).next_to(preds, LEFT, buff=0.5)
        self.at("every")
        self.play(*[GrowArrow(a) for a in arrows], *[FadeIn(p, shift=0.3 * DOWN) for p in preds], FadeIn(pred_lab),
                  run_time=0.8)
        self.at("same")
        self.play(*[ShowPassingFlash(a.copy().set_color(YELLOW).set_stroke(width=8), time_width=0.6) for a in arrows],
                  *[Indicate(p, color=YELLOW, scale_factor=1.1) for p in preds], run_time=0.8)
        checks = VGroup(*[Text("✓", font_size=40, color=GREEN).next_to(p, DOWN, buff=0.25) for p in preds])
        self.at("checked")
        self.play(LaggedStart(*[FadeIn(c, scale=1.5) for c in checks], lag_ratio=0.1), run_time=0.5)
        self.at("real")
        self.play(*[Indicate(row[i + 1], color=YELLOW, scale_factor=1.12) for i in range(5)],
                  *[Indicate(p, color=GREEN, scale_factor=1.08) for p in preds], run_time=0.7)
        lesson = Text("5 predictions from one sentence", font_size=32, color=YELLOW).move_to([0.3, -2.7, 0])
        self.at("lessons")
        self.play(FadeIn(lesson, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # 9. Code ------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        # Pango quantises glyph advances, so font 26 (auto-fit to 12.8 wide) renders glyphs >= font 24
        code, hl = code_panel(CODE, font_size=26)
        code.move_to(ORIGIN)
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[1])
        self.clear(run_time=0.4)
        self.play(FadeIn(code, shift=0.3 * UP), run_time=0.5)
        self.at("logits")
        self.play(Create(hl), run_time=0.4)
        self.at("temperature")
        self.play(Indicate(hl, color=YELLOW, scale_factor=1.03), run_time=0.5)
        self.at("top")
        self.play(highlight(hl, code, 2), run_time=0.4)
        self.at("softmax")
        self.play(highlight(hl, code, 4), run_time=0.4)
        self.at("sample")
        self.play(highlight(hl, code, 5), run_time=0.4)
        self.end_section()

    # 10. Outro -------------------------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        self.clear(run_time=0.5)
        cycle = VGroup(*[Text(w, font_size=32, color=GREY_D) for w in ["predict", "pick", "append", "repeat"]])
        cycle.arrange(RIGHT, buff=0.9).move_to([0, 2.3, 0])
        cyc_arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.12, color=GREY_D, stroke_width=3)
                              for a, b in zip(cycle[:-1], cycle[1:])])
        back = CurvedArrow(cycle[3].get_top() + 0.1 * UP, cycle[0].get_top() + 0.1 * UP, angle=TAU / 7, color=GREY_B,
                           stroke_width=3)
        sent = token_row(["The", "cat", "sat", "on", "the"]).move_to([-0.6, 0.2, 0])
        slot = DashedVMobject(RoundedRectangle(corner_radius=0.12, width=0.9, height=0.65, color=YELLOW))
        slot.next_to(sent, RIGHT, buff=0.12)
        self.at("loop")
        self.play(FadeIn(cycle), FadeIn(cyc_arrows), LaggedStart(*[FadeIn(t) for t in sent], lag_ratio=0.08),
                  Create(slot), run_time=0.6)
        self.at("predict")
        self.play(cycle[0].animate.set_color(YELLOW), Indicate(slot, color=YELLOW), run_time=0.4)
        mat = token("mat", color=YELLOW).move_to(slot)
        self.at("pick")
        self.play(cycle[0].animate.set_color(GREY_B), cycle[1].animate.set_color(YELLOW), FadeIn(mat, scale=0.6),
                  run_time=0.4)
        self.at("append")
        self.play(cycle[1].animate.set_color(GREY_B), cycle[2].animate.set_color(YELLOW), FadeOut(slot),
                  mat[0].animate.set_stroke(BLUE_D).set_fill(BLUE_D, 0.3), run_time=0.4)
        self.at("repeat")
        self.play(cycle[2].animate.set_color(GREY_B), cycle[3].animate.set_color(YELLOW), Create(back), run_time=0.5)

        names = ["tokens", "embed", "+ position", "blocks × N", "unembed", "sample"]
        stages = VGroup(*[small_box(n, color=YELLOW if n in ("unembed", "sample") else GREY_B) for n in names])
        stages.arrange(RIGHT, buff=0.42).move_to([0, 0.0, 0])
        links = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.05, color=GREY_C, stroke_width=3,
                               max_tip_length_to_length_ratio=0.4) for a, b in zip(stages[:-1], stages[1:])])
        fp = Text("the whole forward pass", font_size=24, color=GREY_B).next_to(stages, DOWN, buff=0.35)
        self.at("whole")
        self.play(FadeOut(VGroup(sent, mat)),
                  LaggedStart(*[FadeIn(s, shift=0.15 * RIGHT) for s in stages], lag_ratio=0.1),
                  LaggedStart(*[GrowArrow(l) for l in links], lag_ratio=0.1), FadeIn(fp), run_time=0.8)

        rng = np.random.default_rng(4)
        n = 7
        xs = np.linspace(-1, 1, n)
        learned = [np.outer(np.sin(2 * xs), np.cos(3 * xs)) * 0.9,
                   np.clip(np.eye(n) * 0.9 - 0.25 + 0.15 * np.outer(xs, xs), -0.9, 0.9),
                   np.clip(np.outer(xs, np.ones(n)) * 0.8 + 0.2 * np.sin(4 * np.add.outer(xs, xs)), -0.9, 0.9)]
        grids = VGroup(*[matrix_grid(m) for m in learned]).arrange(RIGHT, buff=1.4).move_to([0, -0.2, 0])
        g_labels = VGroup(*[Text(t, font_size=24, color=GREY_B).next_to(g, DOWN, buff=0.25)
                            for t, g in zip(["embeddings", "attention", "MLP"], grids)])
        self.at("matrices")
        self.play(FadeOut(VGroup(stages, links, fp)),
                  LaggedStart(*[FadeIn(g, scale=0.9) for g in grids], lag_ratio=0.15), FadeIn(g_labels), run_time=0.7)
        noisy = VGroup(*[matrix_grid(rng.uniform(0.1, 0.85, (n, n)), noise=True).move_to(g) for g in grids])
        noise_lab = Text("random noise", font_size=28, color=GREY_A).next_to(g_labels, DOWN, buff=0.35)
        self.at("noise")
        self.play(*[Transform(g, ng) for g, ng in zip(grids, noisy)], FadeIn(noise_lab), run_time=0.8)
        qmark = Text("?", font_size=64, color=YELLOW).next_to(grids, RIGHT, buff=0.4)
        self.at("learn")
        self.play(FadeIn(qmark, scale=0.6), run_time=0.4)
        card = next_up_card(NEXT)
        self.at("training")
        self.play(*self.clear_anims(), FadeIn(card, shift=0.2 * UP), run_time=0.8)
        self.end_section()
