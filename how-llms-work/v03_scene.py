"""Video 3 — Embeddings: Words as Vectors.

Render from the repo root:  ./render.sh how-llms-work v03
"""
import random

import numpy as np
from manim import *

from intro import play_token_intro
from v03_script import TAGLINE, TITLE
from voiced_scene import VoicedScene

TOKEN_COLOR = BLUE_D
MONO = "DejaVu Sans Mono"
POS_COLOR = BLUE_C      # colour of positive values in vector cells / strips
NEG_COLOR = RED_C       # colour of negative values

CAT_VEC = [0.21, -0.47, 0.83, 0.05]   # illustrative embedding of " cat"
SENTENCE = [("The", 464), ("cat", 3797), ("sat", 3332), ("on", 319), ("the", 262)]

# Illustrative 2-D map: word -> (position, cluster)
MAP = {
    "cat": ((-3.4, 1.4), 0), "dog": ((-2.5, 2.1), 0), "kitten": ((-3.9, 0.6), 0),
    "mat": ((2.3, 1.3), 1), "rug": ((3.2, 2.1), 1), "carpet": ((3.6, 0.8), 1),
    "one": ((-3.7, -1.6), 2), "two": ((-2.6, -2.1), 2), "three": ((-3.9, -2.6), 2),
    "sat": ((2.4, -1.5), 3), "ran": ((3.5, -1.2), 3), "jumped": ((3.0, -2.4), 3),
}
CLUSTER_COLORS = [TEAL_C, GOLD_C, BLUE_C, PURPLE_B]
# Where the same words start before training (section 8): scattered, away from the axis lines
RANDOM_POS = {   # chosen (by a small search) so no two labels ever overlap while they move
    "cat": (3.3, -1.5), "dog": (5.1, 1.8), "kitten": (-2.9, -2.5), "mat": (-2.3, 1.1),
    "rug": (5.2, -0.4), "carpet": (2.2, -0.5), "one": (-5.4, -2.1), "two": (-1.2, 0.4),
    "three": (1.3, -2.8), "sat": (-5.8, 1.4), "ran": (-5.7, -2.7), "jumped": (4.8, -1.4),
}

CODE = """import numpy as np

vocab_size, d_model = 50257, 768
E = np.random.randn(vocab_size, d_model) * 0.02   # learned in training

ids = [464, 3797, 3332, 319, 262]   # "The cat sat on the"
x = E[ids]                          # one row per token
print(x.shape)                      # (5, 768)"""


# ---------------------------------------------------------------------------- helpers
def token(word, color=TOKEN_COLOR, font_size=28, font=None):
    label = Text(word, font_size=font_size, font=font) if font else Text(word, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.12, width=max(label.width + 0.35, 0.6), height=0.65,
                           stroke_color=color, fill_color=color, fill_opacity=0.3)
    return VGroup(box, label.move_to(box))


def fmt(v):
    """Two decimals with a real minus sign."""
    s = f"{v:.2f}"
    if s in ("-0.00", "0.00"):
        s = "0.00"
    return s.replace("-", "−")


def value_color(v):
    v = max(-1.0, min(1.0, v))
    return interpolate_color(GREY_E, POS_COLOR if v >= 0 else NEG_COLOR, abs(v))


def small_gauss(rng, sigma=0.45):
    """Illustrative matrix entry: a Gaussian value redrawn until it fits in (-0.95, 0.95)."""
    while True:
        v = rng.gauss(0, sigma)
        if abs(v) < 0.95:
            return v


def caption(text):
    return Text(text, font_size=20, color=GREY_B).to_corner(DR, buff=0.3)


def column_vector(values, font_size=30):
    entries = VGroup(*[Text(fmt(v), font=MONO, font_size=font_size) for v in values]).arrange(DOWN, buff=0.25)
    for e in entries:
        e.align_to(entries, RIGHT)
    top, bot = entries.get_top()[1] + 0.15, entries.get_bottom()[1] - 0.15
    xl, xr = entries.get_left()[0] - 0.18, entries.get_right()[0] + 0.18
    left = VMobject().set_points_as_corners([[xl + 0.12, top, 0], [xl, top, 0], [xl, bot, 0], [xl + 0.12, bot, 0]])
    right = VMobject().set_points_as_corners([[xr - 0.12, top, 0], [xr, top, 0], [xr, bot, 0], [xr - 0.12, bot, 0]])
    return VGroup(left.set_stroke(WHITE, 3), entries, right.set_stroke(WHITE, 3))


def strip(values, width, height, segments=16, fade_tail=0.0):
    """A long vector drawn as a thin strip of coloured cells (split into segments for a left-to-right reveal)."""
    n = len(values)
    rgba = np.zeros((1, n, 4), dtype=np.uint8)
    for i, v in enumerate(values):
        rgba[0, i, :3] = value_color(v).to_int_rgb()
        alpha = 1.0
        if fade_tail and i > n * (1 - fade_tail):
            alpha = max(0.0, 1 - (i - n * (1 - fade_tail)) / (n * fade_tail))
        rgba[0, i, 3] = int(255 * alpha)
    parts = Group()
    bounds = np.linspace(0, n, segments + 1).astype(int)
    for a, b in zip(bounds[:-1], bounds[1:]):
        img = ImageMobject(rgba[:, a:b, :])
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        img.stretch_to_fit_width(width * (b - a) / n).stretch_to_fit_height(height)
        parts.add(img)
    parts.arrange(RIGHT, buff=0)
    return parts


def map_axes():
    return Axes(x_range=[-6, 6, 1], y_range=[-3.2, 3.2, 1], x_length=12, y_length=6.4,
                axis_config={"include_ticks": False, "stroke_color": GREY_D, "stroke_width": 2,
                             "tip_width": 0.18, "tip_height": 0.18}).move_to([0, -0.15, 0])


def map_point(word, pos=None):
    p, cl = MAP[word]
    pos = p if pos is None else pos
    dot = Dot([pos[0], pos[1], 0], radius=0.09, color=CLUSTER_COLORS[cl])
    label = Text(word, font_size=24).next_to(dot, RIGHT, buff=0.12)
    return VGroup(dot, label)


def cluster_ring(points, cl):
    box = VGroup(*points)
    return Ellipse(width=box.width + 0.7, height=box.height + 0.7, stroke_color=CLUSTER_COLORS[cl],
                   stroke_width=2, stroke_opacity=0.6, fill_color=CLUSTER_COLORS[cl],
                   fill_opacity=0.07).move_to(box)


class EmbeddingsVideo(VoicedScene):
    VIDEO = "v03"

    def construct(self):
        play_token_intro(self, TITLE, 3, TAGLINE)

        # 1. IDs are just labels ------------------------------------------------------------
        self.section(1)
        toks = VGroup(*[token(w) for w, _ in SENTENCE]).arrange(RIGHT, buff=0.3).move_to([0, 2.1, 0])
        ids = VGroup(*[Text(str(i), font=MONO, font_size=26, color=GREY_B).next_to(t, DOWN, buff=0.3)
                       for t, (_, i) in zip(toks, SENTENCE)])
        ids_label = Text("token IDs", font_size=22, color=GREY_B).next_to(ids, LEFT, buff=0.6)
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in toks], lag_ratio=0.15, run_time=1.0))
        self.at("token")
        self.play(LaggedStart(*[FadeIn(i, shift=0.2 * DOWN) for i in ids], lag_ratio=0.1),
                  FadeIn(ids_label), run_time=0.8)
        self.at("label")
        self.play(Indicate(ids, color=WHITE, scale_factor=1.1), run_time=0.7)
        self.at("cat")
        self.play(toks[1][0].animate.set_stroke(YELLOW).set_fill(YELLOW, 0.3), ids[1].animate.set_color(YELLOW),
                  run_time=0.5)

        nb_rows = VGroup()
        for k, (num, word) in enumerate([(3796, '"rief"'), (3797, '" cat"'), (3798, '"esc"')]):
            y = -0.35 - 0.85 * k
            num_t = Text(str(num), font=MONO, font_size=30).move_to([-0.55, y, 0], aligned_edge=RIGHT)
            arrow_t = Text("→", font_size=28, color=GREY_B).move_to([-0.15, y, 0])
            tok = token(word, font=MONO, font_size=26,
                        color=YELLOW if num == 3797 else TOKEN_COLOR).move_to([0.25, y, 0], aligned_edge=LEFT)
            nb_rows.add(VGroup(num_t, arrow_t, tok))
        self.at("neighbors")
        self.play(FadeIn(nb_rows[1], shift=0.2 * DOWN), run_time=0.5)
        self.at("reef")
        self.play(FadeIn(nb_rows[0], shift=0.2 * DOWN), run_time=0.4)
        self.at("esque")
        self.play(FadeIn(nb_rows[2], shift=0.2 * UP), run_time=0.4)
        self.at("nothing")
        strikes = VGroup(*[Line(r[0].get_left() + 0.1 * LEFT, r[0].get_right() + 0.1 * RIGHT,
                                color=RED, stroke_width=4) for r in nb_rows])
        self.play(*[r[0].animate.set_color(GREY_D) for r in nb_rows], ids.animate.set_color(GREY_D),
                  LaggedStart(*[Create(s) for s in strikes], lag_ratio=0.2), run_time=0.8)
        meaning = Text("meaning?", font_size=36, color=YELLOW).move_to([3.6, -1.2, 0])
        self.at("meaning")
        self.play(FadeIn(meaning, scale=0.8), run_time=0.5)
        self.end_section()

        # 2. A vector per token, i.e. a point ---------------------------------------------------
        self.section(2)
        cat_tok, cat_id = toks[1], ids[1]
        self.play(FadeOut(VGroup(toks[0], *toks[2:], ids[0], *ids[2:], ids_label, nb_rows, strikes, meaning)),
                  cat_tok.animate.move_to([-5.3, 0.55, 0]), cat_id.animate.move_to([-5.3, -0.15, 0]),
                  run_time=0.8)
        self.play(cat_tok[0].animate.set_stroke(TOKEN_COLOR).set_fill(TOKEN_COLOR, 0.3),
                  cat_id.animate.set_color(GREY_B), run_time=0.3)
        vec = column_vector(CAT_VEC).move_to([-2.5, 0.3, 0])
        vec_arrow = Arrow(cat_tok.get_right(), vec.get_left(), buff=0.2, color=GREY_B)
        vec_note = Text("illustrative values", font_size=20, color=GREY_B).next_to(vec, DOWN, buff=0.3)
        self.at("vector")
        self.play(GrowArrow(vec_arrow), FadeIn(vec, shift=0.3 * RIGHT), FadeIn(vec_note), run_time=0.9)
        axes2 = Axes(x_range=[-3, 3, 1], y_range=[-2.5, 2.5, 1], x_length=5.4, y_length=4.6,
                     axis_config={"include_ticks": False, "stroke_color": GREY_C,
                                  "tip_width": 0.18, "tip_height": 0.18}).move_to([3.7, 0.3, 0])
        axes2_note = Text("illustrative 2-D picture", font_size=20, color=GREY_B).next_to(axes2, DOWN, buff=0.25)
        self.at("coordinates")
        self.play(Create(axes2), FadeIn(axes2_note), run_time=0.9)
        cat_dot = Dot(axes2.c2p(1.6, 1.3), radius=0.1, color=YELLOW)
        cat_dot_label = Text("cat", font_size=26).next_to(cat_dot, RIGHT, buff=0.12)
        self.at("point")
        self.play(TransformFromCopy(vec, cat_dot), run_time=0.7)
        self.play(FadeIn(cat_dot_label, shift=0.1 * RIGHT), run_time=0.3)
        self.end_section()

        # 3. The embedding matrix: a lookup table ------------------------------------------------
        self.section(3)
        row_ids = ["0", "1", "2", "⋮", "3796", "3797", "3798", "⋮", "50256"]
        rng = random.Random(3)
        cell_w, cell_h, x0, y0 = 1.3, 0.48, -2.6, 2.55
        table = VGroup()
        for r, rid in enumerate(row_ids):
            y = y0 - r * cell_h
            gap = rid == "⋮"
            vals = CAT_VEC if rid == "3797" else [small_gauss(rng) for _ in range(4)]
            if gap:
                label = Text("⋮", font_size=24, color=GREY_B)
            else:
                label = Text(rid, font=MONO, font_size=22, color=GREY_B)
            label.move_to([x0 - 0.2, y, 0], aligned_edge=RIGHT)
            cells = VGroup()
            for c in range(4):
                rect = Rectangle(width=cell_w, height=cell_h, stroke_color=GREY_D, stroke_width=1.5)
                rect.move_to([x0 + (c + 0.5) * cell_w, y, 0])
                txt = Text("⋮", font_size=24, color=GREY_B) if gap else Text(fmt(vals[c]), font=MONO, font_size=22)
                cells.add(VGroup(rect, txt.move_to(rect)))
            table.add(VGroup(label, cells))
        cells_all = VGroup(*[row[1] for row in table])
        title3 = Text("embedding matrix E", font_size=30).next_to(cells_all, UP, buff=0.3)
        note3 = caption("illustrative values")
        self.play(FadeOut(VGroup(vec, vec_arrow, vec_note, axes2, axes2_note, cat_dot, cat_dot_label)),
                  run_time=0.6)
        row3797 = table[5]
        self.play(cat_tok.animate.move_to([-5.4, row3797.get_y() + 0.7, 0]),
                  cat_id.animate.move_to([-5.4, row3797.get_y(), 0]), run_time=0.6)
        self.at("table")
        self.play(LaggedStart(*[FadeIn(row, shift=0.1 * DOWN) for row in table], lag_ratio=0.08),
                  FadeIn(note3), run_time=1.2)
        self.at("matrix")
        self.play(Write(title3), run_time=0.6)
        brace = Brace(cells_all, direction=RIGHT, color=GREY_B)
        brace_label = VGroup(Text("50,257 rows", font_size=26),
                             Text("(one per token)", font_size=24, color=GREY_B)).arrange(DOWN, buff=0.12)
        brace_label.next_to(brace, RIGHT, buff=0.2)
        self.at("vocabulary")
        self.play(GrowFromCenter(brace), FadeIn(brace_label, shift=0.2 * LEFT), run_time=0.8)
        look = Arrow(cat_id.get_right(), row3797[0].get_left(), buff=0.15, color=YELLOW)
        self.at("lookup")
        self.play(GrowArrow(look), cat_id.animate.set_color(YELLOW), row3797[0].animate.set_color(YELLOW),
                  run_time=0.6)
        grab = SurroundingRectangle(row3797, color=YELLOW, buff=0.06)
        self.at("grabs")
        self.play(Create(grab), run_time=0.4)
        cat_row = row3797[1].copy()
        self.play(cat_row.animate.move_to([0.9, -2.6, 0]), run_time=0.8)
        row_arrow = Arrow([-2.9, -2.6, 0], [0.9 - 2 * cell_w, -2.6, 0], buff=0.15, color=GREY_B)
        self.at("3797")
        self.play(cat_tok.animate.move_to([-3.4, -2.6, 0]), GrowArrow(row_arrow), run_time=0.6)
        self.end_section()

        # 4. Real sizes -------------------------------------------------------------------------------
        self.section(4)
        toy = VGroup(cat_tok, row_arrow, cat_row)
        toy.generate_target()
        toy.target[0].move_to([-6.3, 2.7, 0], aligned_edge=LEFT)
        ax0 = toy.target[0].get_right()[0] + 0.05
        toy.target[1].put_start_and_end_on([ax0, 2.7, 0], [ax0 + 0.6, 2.7, 0])
        toy.target[2].move_to([ax0 + 0.7, 2.7, 0], aligned_edge=LEFT)
        for cell, v in zip(toy.target[2], CAT_VEC):
            cell[0].set_fill(value_color(v), 0.6)
        toy_label = Text("toy vector: 4 numbers", font_size=24, color=GREY_B)
        toy_label.next_to(toy.target[2], RIGHT, buff=0.35)
        self.play(FadeOut(VGroup(table, title3, note3, brace, brace_label, look, grab, cat_id)),
                  MoveToTarget(toy), run_time=0.8)
        self.play(FadeIn(toy_label), run_time=0.4)
        vrng = np.random.default_rng(5)
        s768 = strip(np.clip(vrng.normal(0, 0.45, 768), -1, 1), 6.5, 0.35).move_to([-6.3, 1.2, 0], aligned_edge=LEFT)
        s768_label = Text("768 numbers (GPT-2 small)", font_size=26).next_to(s768, RIGHT, buff=0.3)
        self.at("768")
        self.play(LaggedStart(*[FadeIn(p) for p in s768], lag_ratio=0.12, run_time=1.2),
                  FadeIn(s768_label, run_time=0.8))
        small_w = 2.2
        n_long = int(768 * 12.6 / small_w)
        s_long = strip(np.clip(vrng.normal(0, 0.45, n_long), -1, 1), 12.6, 0.35, segments=24, fade_tail=0.25)
        s_long.move_to([-6.3, -0.4, 0], aligned_edge=LEFT)
        long_label = Text("largest models: many thousands", font_size=26)
        long_label.next_to(s_long, DOWN, buff=0.3).align_to(s_long, LEFT)
        s768_label.generate_target()
        s768_label.target.next_to([-6.3 + small_w, 1.2, 0], RIGHT, buff=0.3)
        self.at("thousands")
        self.play(s768.animate.stretch_to_fit_width(small_w, about_edge=LEFT), MoveToTarget(s768_label),
                  run_time=0.7)
        self.play(LaggedStart(*[FadeIn(p) for p in s_long], lag_ratio=0.08, run_time=1.2),
                  FadeIn(long_label, run_time=0.8))
        axes = map_axes()
        map_note = caption("illustrative 2-D picture")
        pts = {w: map_point(w) for w in MAP}
        self.at("two")
        self.play(FadeOut(Group(s768, s_long, s768_label, long_label, toy_label, cat_tok, row_arrow)),
                  cat_row.animate.scale(0.1).move_to(pts["cat"][0]).set_opacity(0),
                  FadeIn(pts["cat"][0], scale=0.3), Create(axes), FadeIn(map_note), run_time=1.0)
        self.remove(cat_row)
        self.add(pts["cat"])
        self.play(FadeIn(pts["cat"][1]), run_time=0.3)
        self.end_section()

        # 5. Meaning as position ------------------------------------------------------------------
        self.section(5)
        heading5 = Text("used in similar ways → close together", font_size=26).move_to([0, 3.5, 0])
        clusters = [["cat", "dog", "kitten"], ["mat", "rug", "carpet"], ["one", "two", "three"],
                    ["sat", "ran", "jumped"]]
        rings = [cluster_ring([pts[w] for w in words], k) for k, words in enumerate(clusters)]
        self.at("similar")
        self.play(FadeIn(heading5, shift=0.2 * DOWN), run_time=0.6)
        self.at("cat")
        self.play(Indicate(pts["cat"], color=YELLOW), run_time=0.5)
        for w in ["dog", "kitten"]:
            self.at(w)
            self.play(FadeIn(pts[w], scale=0.6), run_time=0.35)
        self.play(Create(rings[0]), run_time=0.4)
        for w in ["mat", "rug", "carpet"]:
            self.at(w)
            self.play(FadeIn(pts[w], scale=0.6), run_time=0.35)
        self.play(Create(rings[1]), run_time=0.4)
        self.at("numbers")
        self.play(LaggedStart(*[FadeIn(pts[w], scale=0.6) for w in clusters[2]], lag_ratio=0.25),
                  Create(rings[2]), run_time=0.8)
        self.at("verbs")
        self.play(LaggedStart(*[FadeIn(pts[w], scale=0.6) for w in clusters[3]], lag_ratio=0.25),
                  Create(rings[3]), run_time=0.8)
        self.end_section()

        # 6. Dot product and cosine similarity -----------------------------------------------------
        self.section(6)
        unit, origin = 0.85, np.array([-3.6, -1.2, 0])
        plane = NumberPlane(x_range=[-3.5, 3.5, 1], y_range=[-2.5, 3, 1], x_length=7 * unit, y_length=5.5 * unit,
                            background_line_style={"stroke_color": GREY_D, "stroke_width": 1, "stroke_opacity": 0.6},
                            axis_config={"stroke_color": GREY_C, "stroke_width": 2})
        plane.shift(origin - plane.c2p(0, 0))
        a_vec = np.array([3.0, 1.0])
        blen = np.sqrt(8.0)
        a_ang, b_ang0 = np.arctan2(1, 3), np.pi / 4
        theta = ValueTracker(b_ang0)

        def b_coords():
            return blen * np.cos(theta.get_value()), blen * np.sin(theta.get_value())

        def b_tip():
            return plane.c2p(*b_coords())

        a_arrow = Arrow(origin, plane.c2p(*a_vec), buff=0, color=BLUE_C, stroke_width=6)
        b_arrow = always_redraw(lambda: Arrow(origin, b_tip(), buff=0, color=TEAL, stroke_width=6))
        a_label = Text("a = (3, 1)", font_size=26, color=BLUE_C).next_to(a_arrow.get_end(), RIGHT, buff=0.15)
        b_label = Text("b = (2, 2)", font_size=26, color=TEAL).next_to(b_tip(), UP, buff=0.15)
        def b_label_pos():
            d = (b_tip() - origin) / np.linalg.norm(b_tip() - origin)
            return b_tip() + 0.1 * d + 0.32 * np.array([-d[1], d[0], 0])

        b_live_label = always_redraw(lambda: Text("b", font_size=26, color=TEAL).move_to(b_label_pos()))

        f1 = Text("a · b = a₁b₁ + a₂b₂", font_size=34).move_to([1.2, 2.9, 0], aligned_edge=LEFT)
        f2 = Text("= 3·2 + 1·2", font_size=34).move_to([0, 2.1, 0]).align_to(f1[3], LEFT)
        f3 = Text("= 8", font_size=34, color=YELLOW).next_to(f2, RIGHT, buff=0.25)

        def dot_value():
            bx, by = b_coords()
            return a_vec[0] * bx + a_vec[1] * by

        live = always_redraw(lambda: Text(f"a · b = {fmt(dot_value())}", font_size=34, color=YELLOW)
                             .move_to([1.2, 0.9, 0], aligned_edge=LEFT))
        verdicts = {k: Text(v, font_size=26, color=GREY_B).move_to([1.2, 0.3, 0], aligned_edge=LEFT)
                    for k, v in [("same", "same way → high"), ("perp", "perpendicular → 0"),
                                 ("opp", "opposite → negative")]}
        cos1 = Text("cos θ = a · b / (‖a‖ ‖b‖)", font_size=32).move_to([1.2, -1.0, 0], aligned_edge=LEFT)
        cos2 = Text("= 8 / (3.16 · 2.83)", font_size=32).move_to([0, -1.8, 0]).align_to(cos1[4], LEFT)
        cos3 = Text("≈ 0.89", font_size=32, color=YELLOW).move_to([0, -2.6, 0]).align_to(cos1[4], LEFT)

        self.play(FadeOut(VGroup(axes, map_note, heading5, *pts.values(), *rings)), run_time=0.5)
        self.play(Create(plane), GrowArrow(a_arrow), FadeIn(b_arrow), FadeIn(a_label), FadeIn(b_label),
                  run_time=0.9)
        self.at("dot")
        self.play(Write(f1), run_time=0.8)
        self.at("multiply")
        self.play(FadeIn(f2, shift=0.2 * DOWN), run_time=0.7)
        self.at("add")
        self.play(FadeIn(f3, scale=1.3), run_time=0.5)
        self.at("vectors")
        self.play(FadeIn(live), FadeOut(b_label), FadeIn(b_live_label), run_time=0.5)
        self.at("same")
        self.play(theta.animate.set_value(a_ang), FadeIn(verdicts["same"]), run_time=0.8)
        self.at("unrelated")
        self.play(theta.animate.set_value(a_ang + PI / 2), FadeOut(verdicts["same"]),
                  FadeIn(verdicts["perp"]), run_time=1.0)
        self.at("opposite")
        self.play(theta.animate.set_value(a_ang + PI), FadeOut(verdicts["perp"]),
                  FadeIn(verdicts["opp"]), run_time=0.9)
        self.at("divide")
        self.play(theta.animate.set_value(b_ang0), FadeOut(verdicts["opp"]), Write(cos1), run_time=1.0)
        self.at("cosine")
        self.play(FadeIn(cos2, shift=0.2 * DOWN), run_time=0.6)
        self.play(FadeIn(cos3, shift=0.2 * DOWN), run_time=0.5)
        angle = Angle(Line(origin, plane.c2p(*a_vec)), Line(origin, b_tip()), radius=1.0, color=YELLOW,
                      stroke_width=4)
        mid = (a_ang + b_ang0) / 2
        theta_label = Text("θ", font_size=30, color=YELLOW).move_to(origin + 1.35 * np.array([np.cos(mid), np.sin(mid), 0]))
        self.at("angle")
        self.play(Create(angle), FadeIn(theta_label), run_time=0.6)
        self.end_section()

        # 7. Directions carry meaning ------------------------------------------------------------------
        self.section(7)
        P = {"man": np.array([-4.4, -1.5, 0]), "woman": np.array([-2.9, 0.8, 0]),
             "king": np.array([1.2, -1.3, 0]), "queen": np.array([2.9, 0.75, 0])}
        step = P["woman"] - P["man"]
        result = P["king"] + step
        wdots = {}
        for w, side in [("man", DOWN), ("woman", UP), ("king", DOWN), ("queen", RIGHT)]:
            d = Dot(P[w], radius=0.1, color=WHITE)
            wdots[w] = VGroup(d, Text(w, font_size=28).next_to(d, side, buff=0.15))
        title7 = Text("classic word embeddings (word2vec)", font_size=26, color=GREY_B).move_to([0, 3.4, 0])
        note7 = caption("illustrative 2-D picture")
        mw = Arrow(P["man"], P["woman"], buff=0.14, color=TEAL, stroke_width=6)
        kq = Arrow(P["king"], P["queen"], buff=0.14, color=TEAL, stroke_width=6)
        eq = VGroup(Text("king", font_size=36), Text("− man", font_size=36), Text("+ woman", font_size=36),
                    Text("≈ queen", font_size=36, color=YELLOW)).arrange(RIGHT, buff=0.3).move_to([0, -3.0, 0])
        for m in (b_arrow, b_live_label, live):
            m.clear_updaters()
        self.play(FadeOut(VGroup(plane, a_arrow, b_arrow, a_label, b_live_label, f1, f2, f3, live,
                                 cos1, cos2, cos3, angle, theta_label)), run_time=0.6)
        self.remove(b_arrow, b_live_label, live)
        self.play(LaggedStart(*[FadeIn(g, scale=0.7) for g in wdots.values()], lag_ratio=0.2), run_time=1.0)
        self.at("classic")
        self.play(FadeIn(title7, shift=0.2 * DOWN), FadeIn(note7), run_time=0.6)
        self.at("man")
        self.play(Indicate(wdots["man"], color=YELLOW), run_time=0.4)
        self.at("woman")
        self.play(GrowArrow(mw), run_time=0.6)
        self.at("king")
        self.play(Indicate(wdots["king"], color=YELLOW), run_time=0.4)
        self.at("queen")
        self.play(GrowArrow(kq), run_time=0.6)
        ring_k = Circle(radius=0.2, color=YELLOW, stroke_width=3).move_to(P["king"])
        self.at("king")
        self.play(FadeIn(eq[0]), Create(ring_k), kq.animate.set_opacity(0.35), run_time=0.35)
        self.at("minus")
        self.play(FadeIn(eq[1]), run_time=0.3)
        moved = mw.copy().set_color(YELLOW)
        self.at("man")
        self.play(Indicate(wdots["man"], color=YELLOW), moved.animate.set_stroke(width=8), run_time=0.4)
        self.at("plus")
        self.play(FadeIn(eq[2]), run_time=0.3)
        self.at("woman")
        self.play(moved.animate.shift(P["king"] - P["man"]), run_time=0.5)
        ring_r = Circle(radius=0.14, color=YELLOW, stroke_width=4).move_to(result)
        self.at("lands")
        self.play(Create(ring_r), run_time=0.3)
        self.at("queen")
        self.play(FadeIn(eq[3], shift=0.2 * LEFT), Indicate(wdots["queen"], color=YELLOW), run_time=0.6)
        self.end_section()

        # 8. Learned, not designed ---------------------------------------------------------------------
        self.section(8)
        axes8 = map_axes()
        note8 = caption("illustrative 2-D picture")
        pts8 = {w: map_point(w, RANDOM_POS[w]) for w in MAP}
        steps = ValueTracker(0)
        counter = always_redraw(lambda: Text(f"training step {int(round(steps.get_value())):,}", font_size=24,
                                             color=GREY_B).to_corner(UL, buff=0.4))
        self.play(FadeOut(VGroup(*wdots.values(), title7, note7, mw, kq, moved, ring_k, ring_r, eq)), run_time=0.6)
        self.play(Create(axes8), FadeIn(note8), run_time=0.8)
        self.at("random")
        self.play(LaggedStart(*[FadeIn(g, scale=0.6) for g in pts8.values()], lag_ratio=0.06), FadeIn(counter),
                  run_time=0.8)
        jr = random.Random(11)
        fracs = [0.15, 0.3, 0.45, 0.6, 0.72, 0.84, 0.93, 1.0]
        self.at("nudges")
        for k, f in enumerate(fracs):
            jitter = 0.35 * (1 - f)
            anims = []
            for w, g in pts8.items():
                start, goal = np.array([*RANDOM_POS[w], 0.0]), np.array([*MAP[w][0], 0.0])
                target = start + (goal - start) * f + jitter * np.array([jr.uniform(-1, 1), jr.uniform(-1, 1), 0])
                anims.append(g.animate.shift(target - g[0].get_center()))
            self.play(*anims, steps.animate.set_value(250 * (k + 1)), run_time=0.45)
        rings8 = [cluster_ring([pts8[w] for w in words], k) for k, words in enumerate(clusters)]
        self.at("words")
        self.play(*[Create(r) for r in rings8], run_time=0.6)
        self.end_section()

        # 9. Code ------------------------------------------------------------------------------------
        self.section(9)
        code = Code(code_string=CODE, language="python", formatter_style="monokai",
                    background="window", paragraph_config={"font_size": 20})
        code.scale_to_fit_width(12.8).move_to([0, 0.2, 0])
        row_h = abs(code.line_numbers[0].get_y() - code.line_numbers[1].get_y())
        hl = Rectangle(width=code.code_lines.width + 0.25, height=row_h, color=YELLOW, stroke_width=2)
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[6])

        def move_hl(i):
            return hl.animate.match_y(code.line_numbers[i])

        counter.clear_updaters()
        self.play(FadeOut(VGroup(axes8, note8, counter, *pts8.values(), *rings8)), run_time=0.5)
        self.remove(counter)
        self.at("code")
        self.play(FadeIn(code, shift=0.3 * UP), run_time=0.7)
        self.at("line")
        self.play(Create(hl), run_time=0.5)
        self.at("matrix")
        self.play(move_hl(3), run_time=0.5)
        self.at("index")
        self.play(move_hl(5), run_time=0.5)
        self.at("get")
        self.play(move_hl(6), run_time=0.5)
        self.at("768")
        self.play(move_hl(7), run_time=0.5)
        self.end_section()

        # 10. Outro: same tokens, same vectors -> next up: position ---------------------------------------
        self.section(10)
        recap = VGroup(*[map_point(w) for w in MAP])
        recap_rings = VGroup(*[cluster_ring([recap[list(MAP).index(w)] for w in words], k)
                               for k, words in enumerate(clusters)])
        self.play(FadeOut(VGroup(code, hl)), run_time=0.5)
        self.at("every")
        self.play(FadeIn(recap), FadeIn(recap_rings), run_time=0.8)
        self.at("catch")
        self.play(FadeOut(VGroup(recap, recap_rings)), run_time=0.6)

        words1 = "The cat sat on the mat".split()
        words2 = "The mat sat on the cat".split()
        row1 = VGroup(*[token(w) for w in words1]).arrange(RIGHT, buff=0.15).move_to([-0.6, 2.9, 0])
        row2 = VGroup(*[token(w) for w in words2]).arrange(RIGHT, buff=0.15).move_to([-0.6, 2.0, 0])
        bag_order = sorted(words1)          # ['The', 'cat', 'mat', 'on', 'sat', 'the']
        self.at("the")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in row1], lag_ratio=0.15), run_time=0.9)
        self.at("and")
        self.at("the")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in row2], lag_ratio=0.15), run_time=0.9)

        def bag_slots(cx, cy=-0.3):
            return [np.array([cx + (i % 3 - 1) * 1.15, cy + 0.4 - (i // 3) * 0.8, 0]) for i in range(6)]

        bags, bag_tokens = VGroup(), VGroup()
        anims = []
        for row, words, cx in [(row1, words1, -3.3), (row2, words2, 3.3)]:
            bag = RoundedRectangle(corner_radius=0.25, width=3.9, height=2.0, stroke_color=GREY_B,
                                   stroke_width=2).move_to([cx, -0.3, 0])
            bags.add(bag)
            slots = bag_slots(cx)
            used = set()
            for t, w in zip(row, words):
                idx = next(i for i, bw in enumerate(bag_order) if bw == w and i not in used)
                used.add(idx)
                c = t.copy()
                bag_tokens.add(c)
                anims.append(c.animate.move_to(slots[idx]))
        eq_bags = Text("=", font_size=48).move_to([0, -0.3, 0])
        self.at("same")
        self.play(Create(bags), *anims, FadeIn(eq_bags), run_time=1.2)

        vrng10 = random.Random(21)
        tok_vals = {w: [vrng10.uniform(-1, 1) for _ in range(4)] for w in bag_order}

        def vec_columns(cx):
            cols = VGroup()
            for i, w in enumerate(bag_order):
                col = VGroup(*[Rectangle(width=0.36, height=0.28, stroke_color=GREY_D, stroke_width=1,
                                         fill_color=value_color(v), fill_opacity=0.9) for v in tok_vals[w]])
                cols.add(col.arrange(DOWN, buff=0))
            return cols.arrange(RIGHT, buff=0.14).move_to([cx, -2.6, 0])

        vcols = VGroup(vec_columns(-3.3), vec_columns(3.3))
        eq_vecs = Text("=", font_size=48).move_to([0, -2.6, 0])
        note10 = caption("illustrative values")
        self.at("vectors")
        self.play(*[LaggedStart(*[FadeIn(c, shift=0.2 * DOWN) for c in cols], lag_ratio=0.1) for cols in vcols],
                  FadeIn(eq_vecs), FadeIn(note10), run_time=0.8)
        qmark = Text("?", font_size=90, color=YELLOW).move_to([4.2, 2.45, 0])
        self.at("first")
        self.play(FadeIn(qmark, scale=0.5), run_time=0.5)

        next_label = Text("Next up", font_size=30, color=GREY_B)
        next_title = Text("Where Am I? Position", font_size=44)
        card = VGroup(next_label, next_title).arrange(DOWN, buff=0.35)
        self.at("next")
        self.play(FadeOut(VGroup(row1, row2, bags, bag_tokens, eq_bags, vcols, eq_vecs, note10)),
                  qmark.animate.move_to(ORIGIN), run_time=0.7)
        self.at("position")
        self.play(ReplacementTransform(qmark, card), run_time=0.9)
        late = self.renderer.time - (self.sec_start + self.sec["dur"] + self.SECTION_GAP)
        if late > 0.05:  # VoicedScene only reports overruns when the next section starts
            print(f"section 10 overran its narration by {late:.2f}s")
        self.end_section()
        self.wait(1.0)   # hold the Next up card
        self.play(FadeOut(card))
        self.wait(0.5)
