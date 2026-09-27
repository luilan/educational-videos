"""Video 5 — Attention I: Tokens Talking to Each Other.

Render from the repo root:  ./render.sh how-llms-work v05
"""
import numpy as np
from manim import *

from common import MONO, finish, next_up_card, token, token_row
from intro import play_token_intro
from v05_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

Q_COLOR, K_COLOR, V_COLOR = YELLOW, TEAL, ORANGE
RIVER_HUE, BANK_HUE, OTHER_HUE = GREEN_C, BLUE_C, GREY_B
SENT = ["I", "sat", "on", "the", "river", "bank"]
WEIGHTS = [0.03, 0.05, 0.04, 0.04, 0.72, 0.12]          # bank's attention over SENT (illustrative)
assert abs(sum(WEIGHTS) - 1.0) < 1e-9

# Section 8: illustrative 3-number value vectors, one per token of SENT
VALUES = np.array([[0.10, 0.40, -0.20],
                   [0.30, -0.10, 0.20],
                   [-0.20, 0.10, 0.30],
                   [0.00, 0.20, 0.10],
                   [0.80, -0.50, 0.60],
                   [0.20, 0.35, -0.40]])
BLEND = np.round(np.array(WEIGHTS) @ VALUES, 2)          # weighted mix, shown rounded
BANK_X = np.array([0.30, 0.50, -0.30])                   # bank's own vector
UPDATED = np.round(BANK_X + BLEND, 2)

# Section 7: illustrative 2-D query / keys
QUERY = (1.6, 1.2)
KEY_THE = (-0.4, 1.0)
KEY_SAT = (0.6, -1.3)
RIVER_LEN = 1.5
RIVER_START_DEG = 150.0
RIVER_END_DEG = float(np.degrees(np.arctan2(0.9, 1.2)))  # same direction as the query -> (1.2, 0.9)


# ---------------------------------------------------------------------------- helpers
def fmt(v, nd=2, width=None):
    s = f"{v:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    if width:
        s = s.rjust(width)
    return s.replace("-", "−")


def label(text, color=GREY_B, font_size=22, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def shade(hue, v):
    return interpolate_color(BLACK, hue, 0.2 + 0.8 * float(v))


def cells(vals, hue, w=0.32, h=0.22, buff=0.05, direction=DOWN):
    """A small vector drawn as a strip of coloured cells (brightness = value in [0, 1])."""
    col = VGroup(*[Rectangle(width=w, height=h, stroke_color=hue, stroke_width=1.2,
                             fill_color=shade(hue, v), fill_opacity=1) for v in vals])
    return col.arrange(direction, buff=buff)


def num_vec(entries, color=WHITE, font_size=22, buff=0.14, bracket_color=GREY_B):
    """A column vector of numbers: VGroup(brackets, rows)."""
    rows = VGroup(*[Text(e, font=MONO, font_size=font_size, color=color) for e in entries])
    rows.arrange(DOWN, buff=buff)
    for r in rows:
        r.align_to(rows, RIGHT)
    h, w = rows.height + 0.3, rows.width + 0.36

    def bracket(side):
        x = side * w / 2
        d = -side * 0.12
        pts = [[x + d, h / 2, 0], [x, h / 2, 0], [x, -h / 2, 0], [x + d, -h / 2, 0]]
        return VMobject(stroke_color=bracket_color, stroke_width=2.5).set_points_as_corners(pts)

    brackets = VGroup(bracket(-1), bracket(1)).move_to(rows)
    return VGroup(brackets, rows)


def wsub(sub, color=WHITE, size=32):
    """'W' with a subscript letter, e.g. W_Q, built from two Text objects."""
    w = Text("W", font_size=size, color=color)
    s = Text(sub, font_size=max(20, int(size * 0.66)), color=color)
    s.next_to(w, RIGHT, buff=0.03).align_to(w, DOWN).shift(0.1 * DOWN)
    return VGroup(w, s)


def box_to(tok, color, opacity=0.3):
    """Recolour a token's box only (never the whole box+label group)."""
    return tok[0].animate.set_stroke(color=color).set_fill(color=color, opacity=opacity)


def pulse(mob, scale=1.18):
    return mob.animate(rate_func=there_and_back).scale(scale)


def arc_up(p, q, k=0.38, base=0.25):
    """A Bezier arc from p to q that bulges upward."""
    h = k * abs(q[0] - p[0]) + base
    return CubicBezier(p, p + h * UP, q + h * UP, q)


def dashed_box(mob, buff=0.15, color=GREY_B):
    rect = SurroundingRectangle(mob, buff=buff, corner_radius=0.12, color=color, stroke_width=2)
    return DashedVMobject(rect, num_dashes=36)


class AttentionIntroVideo(VoicedScene):
    VIDEO = "v05"

    def clear_anims(self):
        anims = []
        for m in list(self.mobjects):
            if isinstance(m, ValueTracker):
                m.clear_updaters()
                self.remove(m)
            else:
                anims.append(FadeOut(m))
        return anims

    def clear(self, run_time=0.45):
        anims = self.clear_anims()
        if anims:
            self.play(*anims, run_time=run_time)

    def construct(self):
        play_token_intro(self, TITLE, 5, TAGLINE)
        self.s1_recap()
        self.s2_context()
        self.s3_idea()
        self.s4_weights()
        self.s5_qkv()
        self.s6_library()
        self.s7_dot()
        self.s8_update()
        self.s9_causal()
        self.s10_learned()
        self.s11_outro()
        finish(self)

    # 1. Recap: every vector was made on its own ---------------------------------------
    def s1_recap(self):
        self.section(1)
        toks = token_row(["The", "cat", "sat", "on", "the"], buff=0.8).move_to([0, 1.35, 0])
        rng = np.random.default_rng(1)
        vecs = VGroup(*[cells(rng.uniform(0.15, 1.0, 4), BLUE_C).next_to(t, DOWN, buff=0.45) for t in toks])
        what = Text("what it is", font_size=28, color=GREY_B)
        plus = Text("+", font_size=28, color=GREY_B)
        where = Text("where it is", font_size=28, color=GREY_B)
        VGroup(what, plus, where).arrange(RIGHT, buff=0.25).move_to([0, -1.45, 0])
        self.at("token")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in toks], lag_ratio=0.12), run_time=0.8)
        self.at("vector")
        self.play(LaggedStart(*[GrowFromEdge(v, UP) for v in vecs], lag_ratio=0.1), run_time=0.6)
        self.at("what")
        self.play(FadeIn(what, shift=0.2 * UP), run_time=0.4)
        self.at("where")
        self.play(FadeIn(plus), FadeIn(where, shift=0.2 * UP), run_time=0.4)

        boxes = VGroup(*[dashed_box(VGroup(t, v)) for t, v in zip(toks, vecs)])
        alone = label("each made on its own", font_size=24).move_to([0, -2.2, 0])
        self.at("own")
        self.play(LaggedStart(*[Create(b) for b in boxes], lag_ratio=0.1), FadeIn(alone), run_time=0.8)

        q = Text("?", font_size=80, color=YELLOW).move_to([0, 2.85, 0])
        self.at("words")
        self.play(FadeIn(q, scale=1.5), run_time=0.5)
        self.end_section()

    # 2. Context changes meaning ---------------------------------------------------------
    def s2_context(self):
        self.section(2)
        self.clear()
        R, VX, CX0 = -0.15, 0.85, 1.8   # rows' right edge, vector column, caption left edge
        r1 = token_row(["I", "sat", "on", "the", "river", "bank"], buff=0.08, font_size=24)
        r2 = token_row(["I", "paid", "money", "into", "the", "bank"], buff=0.08, font_size=24)
        r1.move_to([0, 1.45, 0]).align_to(np.array([R, 0, 0]), RIGHT)
        r2.move_to([0, -1.45, 0]).align_to(np.array([R, 0, 0]), RIGHT)
        c1 = Text("edge of a river", font_size=26, color=GREY_A).next_to([CX0, r1.get_y(), 0], RIGHT, buff=0)
        c2 = Text("a place that keeps money", font_size=26, color=GREY_A).next_to([CX0, r2.get_y(), 0], RIGHT,
                                                                                  buff=0)
        self.at("bank")
        self.play(FadeIn(r1[5], shift=0.2 * DOWN), run_time=0.5)
        self.at("i")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in r1[:5]], lag_ratio=0.1), run_time=0.65)
        self.at("riverbank")
        self.play(box_to(r1[5], YELLOW), run_time=0.4)
        self.at("edge")
        self.play(FadeIn(c1, shift=0.2 * LEFT), run_time=0.5)
        self.at("river")
        self.play(pulse(r1[4]), run_time=0.5)
        self.at("i")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in r2], lag_ratio=0.1), run_time=0.7)
        self.at("bank")
        self.play(box_to(r2[5], YELLOW), run_time=0.4)
        self.at("keeps")
        self.play(FadeIn(c2, shift=0.2 * LEFT), run_time=0.5)
        self.at("money")
        self.play(pulse(r2[2]), run_time=0.5)

        vals = [0.9, 0.35, 0.7, 0.2]
        v1 = cells(vals, BANK_HUE).move_to([VX, r1.get_y(), 0])
        v2 = cells(vals, BANK_HUE).move_to([VX, r2.get_y(), 0])
        a1 = Arrow([R + 0.08, r1.get_y(), 0], [VX - 0.25, r1.get_y(), 0], buff=0, color=GREY_B, stroke_width=3,
                   max_tip_length_to_length_ratio=0.3)
        a2 = Arrow([R + 0.08, r2.get_y(), 0], [VX - 0.25, r2.get_y(), 0], buff=0, color=GREY_B, stroke_width=3,
                   max_tip_length_to_length_ratio=0.3)
        eq = Text("=", font_size=52, color=YELLOW).move_to([VX, 0.2, 0])
        same = Text("same vector", font_size=20, color=YELLOW).move_to([VX, -0.35, 0])
        self.at("same")
        self.play(GrowArrow(a1), GrowArrow(a2), GrowFromEdge(v1, LEFT), GrowFromEdge(v2, LEFT),
                  FadeIn(eq, scale=1.4), FadeIn(same), run_time=0.8)

        cx = 3.35
        neq = Text("≠", font_size=56, color=RED).move_to([cx, 0.2, 0])
        diff = Text("different meanings", font_size=20, color=RED_B).move_to([cx, -0.35, 0])
        self.at("different")
        self.play(FadeIn(neq, scale=1.4), FadeIn(diff), run_time=0.6)
        self.end_section()

    # 3. The idea: look around and borrow -------------------------------------------------
    def s3_idea(self):
        self.section(3)
        self.clear(0.4)
        title = Text("attention", font_size=40).move_to([0, 3.2, 0])
        row = token_row(SENT, buff=0.45).move_to([0, -0.45, 0])
        hues = [OTHER_HUE] * 4 + [RIVER_HUE, BANK_HUE]
        rng = np.random.default_rng(3)
        vals = [rng.uniform(0.2, 1.0, 4) for _ in SENT]
        vecs = VGroup(*[cells(v, h).next_to(t, DOWN, buff=0.35) for v, h, t in zip(vals, hues, row)])
        self.play(FadeIn(title, shift=0.2 * DOWN), FadeIn(row), FadeIn(vecs), run_time=0.6)

        b = row[5]
        arcs = VGroup(*[arc_up(b.get_top(), row[i].get_top()).set_stroke(GREY_B, 2, 0.9) for i in range(5)])
        inflow = VGroup(*[arc_up(row[i].get_top(), b.get_top()).set_stroke(YELLOW, 5) for i in range(5)])
        self.at("look")
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.15), run_time=0.9)
        self.at("pull")
        self.play(LaggedStart(*[ShowPassingFlash(a.copy(), time_width=0.5) for a in inflow], lag_ratio=0.1),
                  run_time=1.0)
        self.at("around")
        self.play(LaggedStart(*[ShowPassingFlash(a.copy().set_stroke(WHITE, 4), time_width=0.5)
                                for a in arcs], lag_ratio=0.12), pulse(b, 1.12), run_time=0.8)
        self.at("notices")
        self.play(arcs[4].animate.set_stroke(YELLOW, 6, 1), *[a.animate.set_stroke(opacity=0.25) for a in arcs[:4]],
                  pulse(row[4]), run_time=0.6)

        packet = vecs[4].copy()
        self.at("updates")
        self.play(packet.animate(path_arc=PI / 2).move_to(vecs[5]).set_opacity(0.2), run_time=0.9)
        self.remove(packet)

        tag = Text("bank (river)", font_size=26, t2c={"(river)": RIVER_HUE}).next_to(vecs[5], DOWN, buff=0.3)
        self.at("riverbank")
        self.play(*[c.animate.set_fill(interpolate_color(c.get_fill_color(), r.get_fill_color(), 0.6))
                      .set_stroke(interpolate_color(ManimColor(BANK_HUE), ManimColor(RIVER_HUE), 0.6))
                    for c, r in zip(vecs[5], vecs[4])],
                  FadeIn(tag, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # 4. Attention weights -----------------------------------------------------------------
    def s4_weights(self):
        self.section(4)
        self.clear()
        xs = [-4.9 + 1.5 * i for i in range(6)]
        base, S = -1.9, 4.4
        head = Text("how much does bank listen to each token?", font_size=30, t2c={"bank": BLUE_C})
        head.move_to([0, 3.1, 0])
        axis = Line([xs[0] - 0.75, base, 0], [xs[-1] + 0.75, base, 0], color=GREY_B, stroke_width=2)
        names = VGroup(*[token(w, font_size=24).move_to([x, base - 0.5, 0]) for w, x in zip(SENT, xs)])
        self.play(FadeIn(head, shift=0.2 * DOWN), Create(axis), FadeIn(names), run_time=0.6)

        bars = VGroup(*[Rectangle(width=0.8, height=w * S, stroke_width=0, fill_color=BLUE_C, fill_opacity=0.85)
                        .move_to([x, base, 0], aligned_edge=DOWN) for w, x in zip(WEIGHTS, xs)])
        vals = VGroup(*[Text(f"{w:.2f}", font=MONO, font_size=22).next_to(bar, UP, buff=0.12)
                        for w, bar in zip(WEIGHTS, bars)])
        note = label("illustrative values", font_size=20).move_to([(xs[0] + xs[-1]) / 2, -3.35, 0])
        self.at("listen")
        self.play(LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.12), run_time=1.0)
        self.at("weights")
        self.play(LaggedStart(*[FadeIn(v, shift=0.1 * UP) for v in vals], lag_ratio=0.08), FadeIn(note),
                  run_time=0.6)

        BX, BH = 5.4, 3.0
        segs = VGroup()
        y = base
        for w in WEIGHTS:
            seg = Rectangle(width=0.8, height=w * BH, stroke_color=BLACK, stroke_width=2, fill_color=BLUE_C,
                            fill_opacity=0.85).move_to([BX, y, 0], aligned_edge=DOWN)
            segs.add(seg)
            y += w * BH
        total = Text("sum = 1.00", font_size=28).next_to(segs, UP, buff=0.25)
        budget = label("budget", font_size=24).move_to([BX, base - 0.5, 0])
        self.at("one")
        self.play(*[TransformFromCopy(bar, seg) for bar, seg in zip(bars, segs)], FadeIn(total), run_time=0.7)
        self.at("budget")
        self.play(FadeIn(budget, shift=0.2 * UP), run_time=0.4)

        self.at("river")
        self.play(bars[4].animate.set_fill(YELLOW), vals[4].animate.set_color(YELLOW),
                  segs[4].animate.set_fill(YELLOW), box_to(names[4], YELLOW), run_time=0.5)
        for cue, k in [("the", 3), ("on", 2)]:
            self.at(cue)
            self.play(Indicate(bars[k], color=YELLOW, scale_factor=1.6),
                      Indicate(vals[k], color=YELLOW, scale_factor=1.4), run_time=0.5)
        self.end_section()

    # 5. Query, key, value ------------------------------------------------------------------
    def s5_qkv(self):
        self.section(5)
        self.clear()
        head = Text("how does a token decide?", font_size=30, color=GREY_B).move_to([0, 3.3, 0])
        bank = token("bank").move_to([-5.2, 0.9, 0])
        bvec = cells([0.9, 0.35, 0.7, 0.2], BANK_HUE).next_to(bank, DOWN, buff=0.3)
        self.play(FadeIn(head, shift=0.2 * DOWN), FadeIn(bank), FadeIn(bvec), run_time=0.6)

        ys = [2.1, 0.0, -2.1]
        colors = [Q_COLOR, K_COLOR, V_COLOR]
        vals = [[0.8, 0.3, 0.6, 0.9], [0.4, 0.9, 0.2, 0.7], [0.6, 0.5, 0.95, 0.3]]
        outs = VGroup(*[cells(v, c).move_to([-2.4, y, 0]) for v, c, y in zip(vals, colors, ys)])
        arrows = VGroup(*[Arrow(bvec.get_right(), o.get_left(), buff=0.2, color=c, stroke_width=4,
                                max_tip_length_to_length_ratio=0.1) for o, c in zip(outs, colors)])
        self.at("produces")
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(o, shift=0.2 * RIGHT))
                                for a, o in zip(arrows, outs)], lag_ratio=0.2), run_time=0.9)

        texts = [("query: what am I looking for?", "query:"), ("key: what do I contain?", "key:"),
                 ("value: what will I share?", "value:")]
        for (txt, bold), c, o, cue in zip(texts, colors, outs, ["query", "key", "value"]):
            t = Text(txt, font_size=30, color=c, t2w={bold: BOLD}).next_to(o, RIGHT, buff=0.5)
            self.at(cue)
            self.play(FadeIn(t, shift=0.2 * LEFT), pulse(o), run_time=0.5)
        self.end_section()

    # 6. Library analogy ---------------------------------------------------------------------
    def s6_library(self):
        self.section(6)
        self.clear()
        titles = ["Rivers & Lakes", "Banking 101", "Cooking", "Poetry"]
        fills = ["#3E2C23", "#23303E", "#2E3B26", "#3B2438"]
        widths = [3.2, 3.0, 2.8, 3.1]
        jitter = [0.1, -0.1, 0.05, -0.05]
        ys = [1.55, 0.8, 0.05, -0.7]
        BXC = 4.0
        books = VGroup()
        for t, f, w, j, y in zip(titles, fills, widths, jitter, ys):
            body = RoundedRectangle(corner_radius=0.06, width=w, height=0.62, stroke_color=GREY_B, stroke_width=2,
                                    fill_color=f, fill_opacity=1).move_to([BXC + j, y, 0])
            books.add(VGroup(body, Text(t, font_size=26, color=GREY_B).move_to(body)))
        plank = Rectangle(width=4.0, height=0.12, stroke_width=0, fill_color=GREY_BROWN, fill_opacity=1)
        plank.move_to([BXC, ys[-1] - 0.31 - 0.08, 0])
        self.at("library")
        self.play(LaggedStart(*[FadeIn(b, shift=0.3 * DOWN) for b in reversed(books)], lag_ratio=0.12),
                  FadeIn(plank), run_time=0.8)

        card_box = RoundedRectangle(corner_radius=0.15, width=4.1, height=1.2, stroke_color=Q_COLOR, stroke_width=3,
                                    fill_color=Q_COLOR, fill_opacity=0.12)
        card = VGroup(card_box, Text("which kind of bank?", font_size=26).move_to(card_box)).move_to([-4.4, 0.4, 0])
        qlab = Text("query", font_size=26, color=Q_COLOR).next_to(card, UP, buff=0.25)
        self.at("query")
        self.play(FadeIn(card, shift=0.3 * RIGHT), FadeIn(qlab), run_time=0.6)

        klab = Text("keys: the titles on the spines", font_size=24, color=K_COLOR).move_to([BXC, 2.45, 0])
        self.at("key")
        self.play(*[b[1].animate.set_color(K_COLOR) for b in books], FadeIn(klab, shift=0.2 * DOWN), run_time=0.6)

        scores = [0.9, 0.6, 0.1, 0.2]
        start = card.get_right() + 0.1 * RIGHT
        lines = VGroup(*[Line(start, [1.55, y, 0], color=GREY_B, stroke_width=2) for y in ys])
        stexts = VGroup(*[Text(f"{s:.1f}", font=MONO, font_size=24).move_to([1.95, y, 0])
                          for s, y in zip(scores, ys)])
        snote = label("illustrative match scores", font_size=20).move_to([BXC, -1.45, 0])
        self.at("compare")
        self.play(LaggedStart(*[AnimationGroup(Create(l), FadeIn(s)) for l, s in zip(lines, stexts)],
                              lag_ratio=0.2), FadeIn(snote), run_time=1.1)

        best = books[0]
        self.at("match")
        self.play(best[0].animate.set_stroke(K_COLOR, 5).set_fill(K_COLOR, 0.3), lines[0].animate.set_stroke(YELLOW, 4),
                  stexts[0].animate.set_color(YELLOW), *[b.animate.set_opacity(0.45) for b in books[1:]],
                  *[l.animate.set_stroke(opacity=0.3) for l in lines[1:]], run_time=0.6)

        # open book with orange contents
        OB = np.array([0.4, -2.45, 0])
        pages = VGroup(*[Rectangle(width=1.6, height=1.1, stroke_color=GREY_B, stroke_width=2, fill_color=GREY_E,
                                   fill_opacity=1) for _ in range(2)]).arrange(RIGHT, buff=0).move_to(OB)
        text_lines = VGroup()
        for p in pages:
            for k, frac in enumerate([0.8, 0.65, 0.75, 0.5]):
                y = p.get_top()[1] - 0.25 - 0.2 * k
                x0 = p.get_left()[0] + 0.18
                text_lines.add(Line([x0, y, 0], [x0 + frac * 1.25, y, 0], color=V_COLOR, stroke_width=3))
        open_book = VGroup(pages, text_lines)
        self.at("read")
        self.play(TransformFromCopy(best, open_book), run_time=0.7)

        info = Text("land along a river's edge", font_size=24, color=V_COLOR).next_to(card, DOWN, buff=0.35)
        self.at("contents")
        self.play(TransformFromCopy(text_lines, info), run_time=0.8)
        vlab = Text("value", font_size=30, color=V_COLOR).next_to(pages, RIGHT, buff=0.35)
        self.at("value")
        self.play(FadeIn(vlab, shift=0.2 * LEFT), pulse(info, 1.1), run_time=0.45)
        self.end_section()

    # 7. Matching with dot products ------------------------------------------------------------
    def s7_dot(self):
        self.section(7)
        self.clear()
        O = np.array([-3.6, -0.9, 0])
        S = 1.25

        def P(v):
            return O + S * np.array([v[0], v[1], 0])

        axes = VGroup(Line(O + [-2.4, 0, 0], O + [2.7, 0, 0]), Line(O + [0, -2.35, 0], O + [0, 2.3, 0]))
        axes.set_stroke(GREY_D, 2)

        def arrow(v, color):
            return Arrow(O, P(v), buff=0, color=color, stroke_width=6, max_tip_length_to_length_ratio=0.12)

        def unit(v):
            v = np.array([v[0], v[1], 0.0])
            return v / np.linalg.norm(v)

        q_arr = arrow(QUERY, Q_COLOR)
        q_lab = Text("bank's query", font_size=24, color=Q_COLOR).next_to(P(QUERY), UP, buff=0.2)
        the_arr = arrow(KEY_THE, K_COLOR)
        the_lab = Text("the", font_size=24, color=K_COLOR).move_to(P(KEY_THE) + 0.35 * unit(KEY_THE))
        sat_arr = arrow(KEY_SAT, K_COLOR)
        sat_lab = Text("sat", font_size=24, color=K_COLOR).move_to(P(KEY_SAT) + 0.35 * unit(KEY_SAT))
        th = ValueTracker(RIVER_START_DEG)

        def river_key():
            a = th.get_value() * DEGREES
            return (round(RIVER_LEN * np.cos(a), 2), round(RIVER_LEN * np.sin(a), 2))

        def river_arrow():
            a = th.get_value() * DEGREES
            v = (RIVER_LEN * np.cos(a), RIVER_LEN * np.sin(a))
            u = unit(v)
            perp = np.array([u[1], -u[0], 0])
            arr = arrow(v, K_COLOR)
            lab = Text("river", font_size=24, color=K_COLOR).move_to(P(v) + 0.1 * u + 0.45 * perp)
            return VGroup(arr, lab)

        r_arr = always_redraw(river_arrow)
        self.at("measured")
        self.play(Create(axes), GrowArrow(q_arr), FadeIn(q_lab), GrowArrow(the_arr), FadeIn(the_lab),
                  GrowArrow(sat_arr), FadeIn(sat_lab), FadeIn(r_arr), run_time=0.7)

        formula = Text("score = query · key", font_size=34, t2c={"query": Q_COLOR, "key": K_COLOR})
        formula.move_to([3.5, 2.75, 0])
        self.at("dot")
        self.play(FadeIn(formula, shift=0.2 * DOWN), run_time=0.5)

        NX, CX, AX, SX = 0.3, 1.6, 4.62, 6.2   # name left, coords left, arrow centre, score right

        def coords(v):
            return f"({fmt(v[0], width=5)}, {fmt(v[1], width=5)})"

        def dot(k):
            return round(QUERY[0] * k[0] + QUERY[1] * k[1], 2)

        def row(name, k, y, color=K_COLOR):
            n = Text(name, font_size=26, color=color).move_to([NX, y, 0], aligned_edge=LEFT)
            c = Text(coords(k), font=MONO, font_size=24, color=color).move_to([CX, y, 0], aligned_edge=LEFT)
            return VGroup(n, c)

        ROW_Y = [1.7, 0.85, 0.1, -0.65]
        q_row = row("query", QUERY, ROW_Y[0], Q_COLOR)
        s_head = label("score", font_size=24).move_to([SX, ROW_Y[0], 0], aligned_edge=RIGHT)
        rows = VGroup()
        for name, k, y in [("the", KEY_THE, ROW_Y[1]), ("sat", KEY_SAT, ROW_Y[2])]:
            r = row(name, k, y)
            r.add(Text("→", font_size=26, color=GREY_B).move_to([AX, y, 0]),
                  Text(fmt(dot(k)), font=MONO, font_size=26).move_to([SX, y, 0], aligned_edge=RIGHT))
            rows.add(r)

        def river_row():
            k = river_key()
            r = row("river", k, ROW_Y[3])
            r.add(Text("→", font_size=26, color=GREY_B).move_to([AX, ROW_Y[3], 0]),
                  Text(fmt(dot(k)), font=MONO, font_size=26).move_to([SX, ROW_Y[3], 0], aligned_edge=RIGHT))
            return r

        r_row = always_redraw(river_row)
        note = label("illustrative 2-D values", font_size=20).move_to([3.5, -3.3, 0])
        self.at("embeddings")
        self.play(LaggedStart(FadeIn(VGroup(q_row, s_head)), FadeIn(rows[0]), FadeIn(rows[1]), FadeIn(r_row),
                              lag_ratio=0.25), FadeIn(note), run_time=1.0)

        self.at("query")
        self.play(Indicate(q_arr, color=Q_COLOR, scale_factor=1.08), Indicate(q_row, color=Q_COLOR, scale_factor=1.05),
                  run_time=0.6)
        self.at("direction")
        self.play(th.animate.set_value(RIVER_END_DEG), run_time=1.3)
        r_arr.clear_updaters()
        r_row.clear_updaters()

        self.at("high")
        self.play(r_row[3].animate.set_color(YELLOW), Flash(r_row[3], color=YELLOW, flash_radius=0.55),
                  run_time=0.6)
        hl = SurroundingRectangle(r_row, color=YELLOW, buff=0.12, corner_radius=0.08)
        self.at("river")
        self.play(Create(hl), pulse(r_arr[1], 1.3), run_time=0.5)
        self.end_section()

    # 8. Updating the vector --------------------------------------------------------------------
    def s8_update(self):
        self.section(8)
        self.clear()
        xs = [-3.625 + 1.45 * i for i in range(6)]
        toks = VGroup(*[token(w, font_size=24).move_to([x, 3.0, 0]) for w, x in zip(SENT, xs)])
        vcols = VGroup(*[num_vec([fmt(v) for v in VALUES[i]], color=V_COLOR, font_size=20).move_to([x, 1.75, 0])
                         for i, x in enumerate(xs)])
        vname = Text("values", font_size=24, color=V_COLOR).move_to([-5.15, 1.75, 0])
        self.at("collects")
        self.play(LaggedStart(*[FadeIn(VGroup(t, c), shift=0.2 * DOWN) for t, c in zip(toks, vcols)],
                              lag_ratio=0.08), FadeIn(vname), run_time=0.8)

        wts = VGroup(*[Text(f"×{w:.2f}", font_size=24).move_to([x, 0.6, 0]) for w, x in zip(WEIGHTS, xs)])
        wname = label("weights", font_size=24).move_to([-5.15, 0.6, 0])
        self.at("mix")
        self.play(LaggedStart(*[FadeIn(w, shift=0.1 * DOWN) for w in wts], lag_ratio=0.06), FadeIn(wname),
                  run_time=0.6)

        EY = -1.95
        EX = [-3.6, -2.5, -1.4, -0.3, 0.8]
        blend = num_vec([fmt(v) for v in BLEND], color=V_COLOR).move_to([EX[2], EY, 0])
        h_blend = Text("weighted mix", font_size=22, color=V_COLOR).next_to(blend, UP, buff=0.3)
        note = label("illustrative values", font_size=20).move_to([0, -3.35, 0])
        movers = [c.copy() for c in vcols]
        self.at("values")
        self.play(LaggedStart(*[m.animate.move_to(blend).scale(0.6).set_opacity(0) for m in movers],
                              lag_ratio=0.08), run_time=0.75)
        for m in movers:
            self.remove(m)
        self.play(FadeIn(blend, scale=0.8), FadeIn(h_blend), FadeIn(note), run_time=0.35)

        pick = SurroundingRectangle(VGroup(vcols[4], wts[4]), color=YELLOW, buff=0.12, corner_radius=0.08)
        self.at("rivers")
        self.play(Create(pick), wts[4].animate.set_color(YELLOW), run_time=0.5)

        bank_v = num_vec([fmt(v) for v in BANK_X], color=BANK_HUE).move_to([EX[0], EY, 0])
        h_bank = Text("bank", font_size=22, color=BANK_HUE).next_to(bank_v, UP, buff=0.3)
        plus = Text("+", font_size=40).move_to([EX[1], EY, 0])
        eq = Text("=", font_size=40).move_to([EX[3], EY, 0])
        new_v = num_vec([fmt(v) for v in UPDATED]).move_to([EX[4], EY, 0])
        h_new = Text("new bank", font_size=22).next_to(new_v, UP, buff=0.3)
        heads = VGroup(h_bank, h_blend, h_new)
        h_bank.match_y(h_blend)
        h_new.match_y(h_blend)
        self.at("adds")
        self.play(FadeIn(bank_v, shift=0.2 * RIGHT), FadeIn(h_bank), FadeIn(plus), run_time=0.45)
        merge = [bank_v[1].copy(), blend[1].copy()]
        self.play(FadeIn(eq), *[m.animate.move_to(new_v[1]).scale(0.7).set_opacity(0) for m in merge],
                  run_time=0.45)
        for m in merge:
            self.remove(m)
        self.play(FadeIn(new_v, scale=0.8), FadeIn(h_new), run_time=0.35)

        self.at("now")
        self.play(Indicate(new_v, color=WHITE, scale_factor=1.1), run_time=0.6)
        final = Text("bank, next to a river", font_size=30, t2c={"river": RIVER_HUE}).next_to(new_v, RIGHT, buff=0.6)
        self.at("river")
        self.play(FadeIn(final, shift=0.2 * LEFT), run_time=0.5)
        self.end_section()

    # 9. Causal: only look backwards ---------------------------------------------------------------
    def s9_causal(self):
        self.section(9)
        self.clear(0.3)
        head = Text("causal attention", font_size=36).move_to([0, 3.1, 0])
        self.at("rule")
        self.play(FadeIn(head, shift=0.2 * DOWN), run_time=0.5)
        row = token_row(["The", "cat", "sat", "on", "the"], buff=0.75, font_size=34).move_to([0, -0.5, 0])
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in row], lag_ratio=0.1), run_time=0.7)
        sub = label("a token can only look backwards", font_size=26).move_to([0, 2.4, 0])
        self.at("only")
        self.play(FadeIn(sub), run_time=0.4)

        cat = row[1]
        to_the = CurvedArrow(cat.get_top() + 0.3 * LEFT, row[0].get_top(), angle=0.8 * PI, color=GREEN,
                             stroke_width=6, tip_length=0.24)
        to_self = CurvedArrow(cat.get_top() + 0.25 * RIGHT, cat.get_top() + 0.15 * LEFT,
                              angle=1.5 * PI, color=GREEN, stroke_width=6, tip_length=0.2)
        self.at("backwards")
        self.play(box_to(cat, YELLOW), Create(to_the), Create(to_self), run_time=0.7)

        fut = row[2:]
        fut_lab = label("the future: not written yet", font_size=26).next_to(fut, DOWN, buff=0.35)
        self.at("future")
        self.play(*[t[0].animate.set_stroke(GREY_D).set_fill(GREY_D, 0.15) for t in fut],
                  *[t[1].animate.set_color(GREY_D) for t in fut], FadeIn(fut_lab), run_time=0.6)

        tick = Text("✓", font_size=48, color=GREEN).next_to(to_the, UP, buff=0.12)
        self.at("they")
        self.play(ShowPassingFlash(to_the.copy().set_stroke(GREEN_A, 10), time_width=0.6),
                  FadeIn(tick, scale=1.4), run_time=0.45)

        arc = ArcBetweenPoints(cat.get_top() + 0.45 * RIGHT, row[2].get_top(), angle=-0.8 * PI, color=RED,
                               stroke_width=6)
        arc.add_tip(tip_length=0.24)
        tips = arc.pop_tips()
        bad = VGroup(DashedVMobject(arc, num_dashes=10), tips)
        cross = Text("✗", font_size=48, color=RED).next_to(arc, UP, buff=0.12)
        self.at("sat")
        self.play(Create(bad), FadeIn(cross, scale=1.4), run_time=0.5)
        self.end_section()

    # 10. The projections are learned ------------------------------------------------------------------
    def s10_learned(self):
        self.section(10)
        self.clear(0.35)
        CXS = [-3.9, 0.3, 4.5]
        colors = [Q_COLOR, K_COLOR, V_COLOR]
        names = ["q", "k", "v"]
        OY = -1.35
        rng = np.random.default_rng(10)
        outs = VGroup()
        for cx, c, n in zip(CXS, colors, names):
            strip = cells(rng.uniform(0.25, 1.0, 4), c, w=0.3, h=0.3, buff=0.05, direction=RIGHT).move_to([cx, OY, 0])
            outs.add(VGroup(strip, Text(n, font_size=32, color=c).next_to(strip, LEFT, buff=0.3)))
        for k, cue in enumerate(["queries", "keys", "values"]):
            self.at(cue)
            self.play(FadeIn(outs[k], shift=0.2 * UP), run_time=0.4)

        x_strip = cells([0.9, 0.35, 0.7, 0.2], BANK_HUE, w=0.3, h=0.3, buff=0.05, direction=RIGHT).move_to([CXS[1], 2.85, 0])
        x_lab = VGroup(Text("x", font_size=32, color=BANK_HUE), label("token vector", font_size=22))
        x_lab.arrange(RIGHT, buff=0.25).next_to(x_strip, LEFT, buff=0.35)
        self.at("multiplying")
        self.play(FadeIn(x_strip), FadeIn(x_lab), run_time=0.5)

        MY = 0.55

        def matrix():
            g = VGroup(*[Square(0.24, stroke_width=0, fill_color=interpolate_color(BLUE_E, RED_E, rng.uniform()),
                                fill_opacity=1) for _ in range(16)])
            g.arrange_in_grid(4, 4, buff=0.04)
            frame = SurroundingRectangle(g, buff=0.06, color=GREY_B, stroke_width=2)
            return VGroup(frame, g)

        mats = VGroup(*[matrix().move_to([cx, MY, 0]) for cx in CXS])
        wlabs = VGroup(*[wsub(s, color=c).next_to(m, LEFT, buff=0.35).shift(0.2 * UP)
                         for s, c, m in zip("QKV", colors, mats)])
        in_arrows = VGroup(*[Arrow(x_strip.get_bottom(), m.get_top(), buff=0.12, color=GREY_B, stroke_width=3,
                                   max_tip_length_to_length_ratio=0.08) for m in mats])
        out_arrows = VGroup(*[Arrow(m.get_bottom(), o[0].get_top(), buff=0.1, color=c, stroke_width=3,
                                    max_tip_length_to_length_ratio=0.25) for m, o, c in zip(mats, outs, colors)])
        self.at("matrix")
        self.play(LaggedStart(*[GrowArrow(a) for a in in_arrows], lag_ratio=0.1),
                  LaggedStart(*[FadeIn(m, scale=0.8) for m in mats], lag_ratio=0.1), FadeIn(wlabs),
                  LaggedStart(*[GrowArrow(a) for a in out_arrows], lag_ratio=0.1), run_time=0.8)

        forms = VGroup()
        for cx, c, n, s in zip(CXS, colors, names, "QKV"):
            f = VGroup(Text(f"{n} = x ·", font_size=26, color=c), wsub(s, color=c, size=26)).arrange(RIGHT, buff=0.1)
            f[1].shift((f[0][2].get_bottom()[1] - f[1][0].get_bottom()[1]) * UP)
            forms.add(f.move_to([cx, OY - 0.75, 0]))
        self.at("those")
        self.play(FadeIn(forms, shift=0.1 * UP), run_time=0.5)

        learned = VGroup(*[label("learned", font_size=20).next_to(w, DOWN, buff=0.15) for w in wlabs])
        self.at("learned")
        self.play(FadeIn(learned), run_time=0.4)

        self.at("training")
        for _ in range(3):
            self.play(*[sq.animate.set_fill(interpolate_color(BLUE_E, RED_E, rng.uniform()))
                        for m in mats for sq in m[1]], run_time=0.3)

        rule = Text("rule: “river explains bank”", font_size=26, color=GREY_B, slant=ITALIC).move_to([0, -3.15, 0])
        self.at("nobody")
        self.play(FadeIn(rule), run_time=0.45)
        strike = Line(rule.get_left() + 0.1 * LEFT, rule.get_right() + 0.1 * RIGHT, color=RED, stroke_width=4)
        cross = Text("✗", font_size=34, color=RED).next_to(rule, RIGHT, buff=0.3)
        self.at("bank")
        self.play(Create(strike), FadeIn(cross), run_time=0.45)

        cap = Text("learned from data, not hand-written", font_size=28, color=YELLOW).move_to([0, -3.15, 0])
        self.at("data")
        self.play(FadeOut(VGroup(rule, strike, cross)), FadeIn(cap, shift=0.2 * UP), run_time=0.55)
        self.end_section()

    # 11. Outro -----------------------------------------------------------------------------------------
    def s11_outro(self):
        self.section(11)
        self.clear(0.4)
        row = token_row(SENT, buff=0.45).move_to([0, -0.6, 0])
        b = row[5]
        arcs = VGroup(*[arc_up(b.get_top(), row[i].get_top()).set_stroke(GREY_B, 2, 0.6) for i in range(4)])
        key_arc = arc_up(b.get_top(), row[4].get_top()).set_stroke(YELLOW, 6)
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in row], lag_ratio=0.08), run_time=0.6)
        title = Text("attention", font_size=40).move_to([0, 3.0, 0])
        self.at("attention")
        self.play(FadeIn(title, shift=0.2 * DOWN), LaggedStart(*[Create(a) for a in [*arcs, key_arc]], lag_ratio=0.12),
                  run_time=0.9)
        ask = Text("?", font_size=48, color=Q_COLOR).next_to(b, RIGHT, buff=0.3)
        self.at("asking")
        self.play(FadeIn(ask, scale=1.5), run_time=0.4)
        inflow = arc_up(row[4].get_top(), b.get_top()).set_stroke(WHITE, 8)
        borrow = label("tokens borrow meaning from each other", font_size=26).move_to([0, -1.8, 0])
        self.at("borrowing")
        self.play(ShowPassingFlash(inflow, time_width=0.6), box_to(b, RIVER_HUE), FadeIn(borrow, shift=0.2 * UP),
                  run_time=0.8)
        self.at("next")
        self.play(*self.clear_anims(), run_time=0.6)
        card = next_up_card(NEXT)
        self.at("math")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.8)
        self.end_section()
        t_end = self.renderer.time - (self.sec_start + self.sec["dur"] + self.SECTION_GAP)
        if t_end > 0.05:
            print(f"section 11 overran its narration by {t_end:.2f}s")
