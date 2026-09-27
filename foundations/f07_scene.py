"""Foundations F7 — Softmax, Properly.

Render from the repo root:  ./render.sh foundations f07
"""
import numpy as np
from manim import *

from common import code_panel, finish, highlight, next_up_card, token, token_row, used_in_card
from f07_script import LABEL, NEXT, TAGLINE, TITLE, USED_IN
from intro import play_token_intro
from voiced_scene import VoicedScene

MONO = "DejaVu Sans Mono"
SCORE_COLOR, PROB_COLOR, HARD_COLOR = BLUE, TEAL, BLUE_D


def softmax(x, T=1.0):
    z = np.asarray(x, dtype=float) / T
    e = np.exp(z - z.max())
    return e / e.sum()


# ------------------------------------------------------------------ sections 1-3: illustrative logits
LOGITS = np.array([2.1, -0.8, 0.4, 1.3])
EXPS = np.exp(LOGITS)                   # 8.17, 0.45, 1.49, 3.67
TOTAL = EXPS.sum()                      # 13.78
PROBS = EXPS / TOTAL                    # 59.3, 3.3, 10.8, 26.6 %
assert [f"{v:.2f}" for v in EXPS] == ["8.17", "0.45", "1.49", "3.67"] and f"{TOTAL:.2f}" == "13.78"
assert f"{sum(round(v, 2) for v in EXPS):.2f}" == "13.78"          # the shown values add up to the shown total
assert [f"{100 * p:.1f}" for p in PROBS] == ["59.3", "3.3", "10.8", "26.6"]
assert round(sum(round(100 * p, 1) for p in PROBS), 1) == 100.0     # the shown percentages add up to 100

# ------------------------------------------------------------------ sections 4-9: "The cat sat on the" → ?
WORDS = ["mat", "floor", "sofa"]
SCORES = np.array([3.0, 2.0, 0.0])
E3 = np.exp(SCORES)
P3 = softmax(SCORES)
assert [f"{v:.1f}" for v in E3] == ["20.1", "7.4", "1.0"] and f"{E3.sum():.1f}" == "28.5"
assert [f"{100 * p:.1f}" for p in P3] == ["70.5", "25.9", "3.5"]
assert np.allclose(softmax(SCORES + 10), P3)
assert [f"{100 * p:.1f}" for p in softmax(SCORES, 0.5)] == ["87.9", "11.9", "0.2"]
assert [f"{100 * p:.1f}" for p in softmax(SCORES, 2.0)] == ["54.7", "33.1", "12.2"]
SHIFTED = np.exp(SCORES - SCORES.max())
assert [f"{v:.2f}" for v in SHIFTED] == ["1.00", "0.37", "0.05"] and f"{SHIFTED.sum():.2f}" == "1.42"
assert np.allclose(SHIFTED / SHIFTED.sum(), P3)
with np.errstate(over="ignore"):
    assert np.isinf(np.exp(np.float64(1000.0))) and f"{np.exp(709.0):.1e}" == "8.2e+307"
HARD = [1.0, 0.0, 0.0]

CODE = """def softmax(scores, temperature=1.0):
    z = np.array(scores) / temperature
    e = np.exp(z - z.max())         # no overflow
    return e / e.sum()

softmax([3, 2, 0])                  # [0.705, 0.259, 0.035]
softmax([3, 2, 0], 0.5)             # [0.879, 0.119, 0.002]"""

# ------------------------------------------------------------------ probability-panel layout (sections 4-7)
ROWS = [0.7, -0.3, -1.3]    # y of the mat / floor / sofa rows
HEAD_Y = 1.75               # column headers
PW, BAR_H = 3.2, 0.5        # bar length for 100 %, bar thickness
SOFT_X0 = 1.7               # bar start of the softmax panel in sections 4-5
S6_SHIFT = 1.75             # the panel slides left by this much in section 6
SCORE_X = -2.2              # score column in sections 6-7


# ------------------------------------------------------------------ helpers
def label(text, color=WHITE, font_size=30, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def num(v, nd=1):
    s = f"{v:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−")


def short(v):
    """One decimal, without a trailing '.0' (3 → "3", 1.5 → "1.5")."""
    s = num(v, 1)
    return s[:-2] if s.endswith(".0") else s


def pct(p):
    return f"{100 * p:.1f} %"


def rich(text, colors=(), font_size=34, color=WHITE, **kw):
    """Text with coloured substrings (Text has no glyphs for spaces, so index the space-free string)."""
    t = Text(text, font_size=font_size, color=color, **kw)
    flat = text.replace(" ", "")
    for sub, col in colors:
        sub = sub.replace(" ", "")
        i = flat.find(sub)
        t[i:i + len(sub)].set_color(col)
    return t


def prob_bar(p, y, x0, color=PROB_COLOR):
    return Rectangle(width=max(p * PW, 0.02), height=BAR_H, stroke_width=0, fill_color=color,
                     fill_opacity=0.85).move_to([x0, y, 0], aligned_edge=LEFT)


def prob_val(text, bar):
    return label(text, GREY_A, 26).next_to(bar, RIGHT, buff=0.15)


def word_col(right_x, font_size=28):
    return VGroup(*[label(w, WHITE, font_size).move_to([right_x, y, 0], aligned_edge=RIGHT)
                    for w, y in zip(WORDS, ROWS)])


def base_line(x0):
    return Line([x0, ROWS[0] + 0.45, 0], [x0, ROWS[-1] - 0.45, 0], color=GREY_B, stroke_width=2)


def gap_braces(x):
    """Braces between neighbouring score rows (their tips point left)."""
    return VGroup(*[Brace(Line([x, ROWS[k], 0], [x, ROWS[k + 1], 0]), LEFT, buff=0.05, color=YELLOW)
                    for k in range(2)])


def gap_labels(braces, T=1.0):
    gaps = [SCORES[0] - SCORES[1], SCORES[1] - SCORES[2]]
    return VGroup(*[label(short(g / T), YELLOW, 30).next_to(b, LEFT, buff=0.12) for g, b in zip(gaps, braces)])


def score_vals(values, x=SCORE_X):
    return VGroup(*[label(short(v), SCORE_COLOR, 34).move_to([x, y, 0]) for v, y in zip(values, ROWS)])


def box(text, color, width, height=0.62, font_size=26):
    rect = RoundedRectangle(corner_radius=0.15, width=width, height=height, stroke_color=color,
                            fill_color=color, fill_opacity=0.25)
    return VGroup(rect, label(text, WHITE if color != YELLOW else YELLOW, font_size).move_to(rect))


class SoftmaxVideo(VoicedScene):
    VIDEO = "f07"

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
        self.s1_logits()
        self.s2_exponentiate()
        self.s3_divide()
        self.s4_example()
        self.s5_soft_vs_hard()
        self.s6_differences()
        self.s7_temperature()
        self.s8_overflow()
        self.s9_code()
        self.s10_outro()
        finish(self)

    # 1. Logits → probabilities --------------------------------------------------------------------
    def s1_logits(self):
        self.section(1)
        cx, y0, u = -4.2, -1.15, 0.9             # left panel centre, zero line, scene units per score unit
        net = box("neural network", PURPLE_B, 3.0).move_to([cx, 2.65, 0])
        self.at("network")
        self.play(FadeIn(net, shift=0.2 * DOWN), run_time=0.5)
        out = Arrow([cx, 2.32, 0], [cx, 1.9, 0], buff=0, color=GREY_B, stroke_width=4,
                    max_tip_length_to_length_ratio=0.4)
        self.at("outputs")
        self.play(GrowArrow(out), run_time=0.35)

        head = label("raw scores = logits", WHITE, 30).move_to([cx, 1.65, 0])
        head[9:].set_color(YELLOW)
        raw, logit_word = head[:9], head[9:]
        self.at("raw")
        self.play(FadeIn(raw), run_time=0.4)

        zero = Line([cx - 2.0, y0, 0], [cx + 2.0, y0, 0], color=GREY_B, stroke_width=2)
        zero_lab = label("0", GREY_B, 22).next_to(zero, LEFT, buff=0.15)
        bars, vals = VGroup(), VGroup()
        for i, s in enumerate(LOGITS):
            x = cx + (i - 1.5) * 0.95
            bar = Rectangle(width=0.6, height=abs(s) * u, stroke_width=0, fill_color=SCORE_COLOR, fill_opacity=0.85)
            bar.move_to([x, y0, 0], aligned_edge=DOWN if s >= 0 else UP)
            v = label(num(s), WHITE, 26).next_to(bar, UP if s >= 0 else DOWN, buff=0.12)
            bars.add(bar)
            vals.add(v)
        note = label("illustrative", GREY_B, 20).move_to([cx, -2.75, 0])
        self.at("logits")
        self.play(FadeIn(logit_word), Create(zero), FadeIn(zero_lab), FadeIn(note),
                  LaggedStart(*[AnimationGroup(GrowFromEdge(b, DOWN if s >= 0 else UP), FadeIn(v))
                                for b, v, s in zip(bars, vals, LOGITS)], lag_ratio=0.15), run_time=0.9)
        self.at("large")
        self.play(Indicate(VGroup(bars[0], vals[0]), scale_factor=1.12), run_time=0.45)
        self.at("small")
        self.play(Indicate(VGroup(bars[2], vals[2]), scale_factor=1.12), run_time=0.45)
        self.at("negative")
        self.play(bars[1].animate.set_fill(RED, 0.85), vals[1].animate.set_color(RED), run_time=0.3)
        self.play(Indicate(VGroup(bars[1], vals[1]), color=RED, scale_factor=1.15), run_time=0.45)

        rx = 4.2
        p_head = label("probabilities", WHITE, 30).move_to([rx, 1.65, 0])
        target = DashedVMobject(RoundedRectangle(corner_radius=0.2, width=3.4, height=2.3), num_dashes=44)
        target.set_stroke(GREY_B, 2).move_to([rx, -0.2, 0])
        rule1 = label("all ≥ 0", WHITE, 34).move_to([rx, 0.25, 0])
        rule2 = label("sum = 1", WHITE, 34).move_to([rx, -0.65, 0])
        self.at("probabilities")
        self.play(FadeIn(p_head), Create(target), run_time=0.6)
        self.at("positive")
        self.play(FadeIn(rule1, shift=0.15 * UP), run_time=0.4)
        self.at("one")
        self.play(FadeIn(rule2, shift=0.15 * UP), run_time=0.4)

        sm = box("softmax", YELLOW, 2.2, height=0.9, font_size=34).move_to([0, -0.2, 0])
        a1 = Arrow([-2.15, -0.2, 0], sm.get_left(), buff=0.1, color=GREY_B)
        a2 = Arrow(sm.get_right(), target.get_left(), buff=0.1, color=GREY_B)
        self.at("softmax")
        self.play(GrowArrow(a1), FadeIn(sm, scale=0.8), GrowArrow(a2), run_time=0.7)
        self.end_section()

    # 2. Step 1: exponentiate ------------------------------------------------------------------------
    def s2_exponentiate(self):
        self.section(2)
        self.head = label("Step 1:  exponentiate", WHITE, 34).move_to([0, 3.3, 0])
        self.play(*self.clear_anims(), FadeIn(self.head), run_time=0.5)

        ax = Axes(x_range=[-1.5, 2.5, 1], y_range=[0, 9, 1], x_length=5.2, y_length=4.5, tips=False,
                  axis_config={"stroke_color": GREY_B, "stroke_width": 2, "include_ticks": False})
        ax.shift(np.array([-3.6, -1.9, 0]) - ax.c2p(0, 0))
        curve = ax.plot(np.exp, x_range=[-1.5, np.log(9)], color=YELLOW, stroke_width=4)
        c_lab = label("eˣ", YELLOW, 34).next_to(curve.get_end(), RIGHT, buff=0.15)
        x_name = label("score", GREY_B, 22).next_to(ax.c2p(2.5, 0), RIGHT, buff=0.15)

        ys = [1.35, 0.45, -0.45, -1.35]
        sx, ax_x, ex = 1.6, 2.75, 3.9                          # table columns: score, arrow, eˣ
        sc_head = label("score", WHITE, 30).move_to([sx, 2.25, 0])
        ex_head = label("eˣ", WHITE, 34).move_to([ex, 2.25, 0])
        t_scores = VGroup(*[label(num(s), RED if s < 0 else SCORE_COLOR, 32).move_to([sx, y, 0])
                            for s, y in zip(LOGITS, ys)])
        t_arrows = VGroup(*[label("→", GREY_B, 30).move_to([ax_x, y, 0]) for y in ys])
        t_exps = VGroup(*[label(num(e, 2), WHITE, 32).move_to([ex, y, 0]) for e, y in zip(EXPS, ys)])

        dots, s_labs, segs, tops = VGroup(), VGroup(), VGroup(), VGroup()
        for s in LOGITS:
            col = RED if s < 0 else SCORE_COLOR
            dots.add(Dot(ax.c2p(s, 0), radius=0.08, color=col))
            s_labs.add(label(num(s), col, 24).next_to(ax.c2p(s, 0), DOWN, buff=0.2))
            segs.add(Line(ax.c2p(s, 0), ax.c2p(s, np.exp(s)), color=WHITE, stroke_width=5))
            tops.add(Dot(ax.c2p(s, np.exp(s)), radius=0.08, color=WHITE))
        segs.set_z_index(1)
        tops.set_z_index(2)
        dots.set_z_index(2)

        self.at("exponentiate")
        self.play(Create(ax), Create(curve), FadeIn(c_lab), FadeIn(x_name), FadeIn(sc_head), FadeIn(ex_head),
                  run_time=0.6)
        self.at("score")
        self.play(FadeIn(dots, scale=0.5), FadeIn(s_labs), FadeIn(t_scores), run_time=0.5)
        self.at("e")
        self.play(Indicate(c_lab, scale_factor=1.3), run_time=0.4)
        self.at("every")
        self.play(*[Create(sg) for sg in segs], FadeIn(tops), FadeIn(t_arrows), FadeIn(t_exps), run_time=0.6)

        check = label("every result > 0  ✓", GREEN, 30).move_to([ax_x, -2.3, 0])
        self.at("positive")
        self.play(segs.animate.set_color(GREEN), tops.animate.set_color(GREEN), t_exps.animate.set_color(GREEN),
                  FadeIn(check, shift=0.15 * UP), run_time=0.5)
        ring = Circle(radius=0.28, color=YELLOW, stroke_width=4).move_to(tops[1])
        row = SurroundingRectangle(VGroup(t_scores[1], t_exps[1]), color=YELLOW, buff=0.15, corner_radius=0.1)
        self.at("negative")
        self.play(Create(ring), Create(row), run_time=0.5)
        self.t_exp_col = VGroup(ex_head, t_exps)
        self.end_section()

    # 3. Step 2: divide by the total -----------------------------------------------------------------
    def s3_divide(self):
        self.section(3)
        head2 = label("Step 2:  divide by the total", WHITE, 34).move_to(self.head)
        col = self.t_exp_col
        ex = -3.2
        self.play(*self.clear_anims(keep=[col, self.head]), FadeTransform(self.head, head2),
                  col.animate.shift((ex - col.get_x()) * RIGHT), run_time=0.6)
        self.head = head2

        ys = [1.35, 0.45, -0.45, -1.35]
        arr_x, x0, pw = -1.95, -1.0, 5.5
        arrows = VGroup(*[label("→", GREY_B, 30).move_to([arr_x, y, 0]) for y in ys])
        div = label("÷ total", YELLOW, 30).move_to([arr_x, 2.25, 0])
        self.at("divide")
        self.play(FadeIn(div), FadeIn(arrows), run_time=0.5)

        rule = Line([ex - 0.75, -1.8, 0], [ex + 0.75, -1.8, 0], color=GREY_B, stroke_width=2)
        total = label(f"total = {TOTAL:.2f}", YELLOW, 30).move_to([ex, -2.3, 0])
        bars = VGroup(*[Rectangle(width=p * pw, height=0.5, stroke_width=0, fill_color=PROB_COLOR, fill_opacity=0.85)
                        .move_to([x0, y, 0], aligned_edge=LEFT) for p, y in zip(PROBS, ys)])
        vals = VGroup(*[prob_val(pct(p), b) for p, b in zip(PROBS, bars)])
        p_head = label("probability", WHITE, 30).move_to([x0 + 2.2, 2.25, 0])
        self.at("total")
        self.play(Create(rule), FadeIn(total), run_time=0.4)
        self.play(FadeIn(p_head), LaggedStart(*[AnimationGroup(GrowFromEdge(b, LEFT), FadeIn(v))
                                                for b, v in zip(bars, vals)], lag_ratio=0.12), run_time=0.7)
        sum1 = label("sum = 1  ✓", GREEN, 32).move_to([x0 + 2.2, -2.3, 0])
        self.at("one")
        self.play(FadeIn(sum1, shift=0.15 * UP), run_time=0.4)
        recipe = rich("softmax  =  exponentiate,  then  ÷ total", [("softmax", YELLOW)], font_size=34)
        recipe.move_to(self.head)
        self.at("recipe")
        self.play(FadeTransform(self.head, recipe), run_time=0.5)
        self.head = recipe
        self.end_section()

    # 4. Worked example: mat / floor / sofa ----------------------------------------------------------
    def s4_example(self):
        self.section(4)
        context = token_row(["The", "cat", "sat", "on", "the"])
        slot = VGroup(DashedVMobject(RoundedRectangle(corner_radius=0.12, width=0.8, height=0.65), num_dashes=24)
                      .set_stroke(YELLOW, 2), label("?", YELLOW, 28))
        slot[1].move_to(slot[0])
        ctx = VGroup(context, slot).arrange(RIGHT, buff=0.12).move_to([0, 3.05, 0])
        cands = VGroup(*[token(w).move_to([-5.3, y, 0]) for w, y in zip(WORDS, ROWS)])
        sx, ex, dx = -3.3, -1.3, 0.4
        h_score = label("score", WHITE, 30).move_to([sx, HEAD_Y, 0])
        self.play(*self.clear_anims(), FadeIn(ctx), FadeIn(cands), FadeIn(h_score), run_time=0.5)

        scores = VGroup(*[label(short(s), SCORE_COLOR, 36).move_to([sx, y, 0]) for s, y in zip(SCORES, ROWS)])
        for k, cue in enumerate(["3", "2", "0"]):
            self.at(cue)
            self.play(FadeIn(scores[k], scale=0.6), run_time=0.35)

        h_exp = label("eˣ", WHITE, 34).move_to([ex, HEAD_Y, 0])
        arr1 = VGroup(*[label("→", GREY_B, 30).move_to([(sx + ex) / 2, y, 0]) for y in ROWS])
        self.at("exponentiated")
        self.play(FadeIn(h_exp), FadeIn(arr1), run_time=0.5)
        exps = VGroup(*[label(num(e, 1), WHITE, 36).move_to([ex, y, 0]) for e, y in zip(E3, ROWS)])
        for k, cue in enumerate(["20", "7", "1"]):
            self.at(cue)
            self.play(FadeIn(exps[k], scale=0.6), run_time=0.35)

        rule = Line([ex - 0.7, -1.85, 0], [ex + 0.7, -1.85, 0], color=GREY_B, stroke_width=2)
        total = label(f"total ≈ {E3.sum():.1f}", YELLOW, 30).move_to([ex, -2.35, 0])
        self.at("total")
        self.play(Create(rule), run_time=0.3)
        self.at("28")
        self.play(FadeIn(total, shift=0.15 * UP), run_time=0.4)

        h_div = label(f"÷ {E3.sum():.1f}", YELLOW, 30).move_to([dx, HEAD_Y, 0])
        arr2 = VGroup(*[label("→", GREY_B, 30).move_to([dx, y, 0]) for y in ROWS])
        h_prob = label("probability", WHITE, 30).move_to([SOFT_X0 + 1.6, HEAD_Y, 0])
        self.at("divided")
        self.play(FadeIn(h_div), FadeIn(arr2), FadeIn(h_prob), run_time=0.5)
        bars = VGroup(*[prob_bar(p, y, SOFT_X0) for p, y in zip(P3, ROWS)])
        vals = VGroup(*[prob_val(pct(p), b) for p, b in zip(P3, bars)])
        for k, cue in enumerate(["70", "26", "3"]):
            self.at(cue)
            self.play(GrowFromEdge(bars[k], LEFT), FadeIn(vals[k]), run_time=0.5)
        self.end_section()

    # 5. Hard max vs softmax -----------------------------------------------------------------------
    def s5_soft_vs_hard(self):
        self.section(5)
        head = rich("Why the name “softmax”?", [("“softmax”", YELLOW)], font_size=34).move_to([0, 3.0, 0])
        self.play(*self.clear_anims(), FadeIn(head), run_time=0.5)

        hx0 = -4.5
        h_head = label("hard max", WHITE, 30).move_to([hx0 + 1.6, HEAD_Y, 0])
        h_words = word_col(hx0 - 0.25)
        h_axis = base_line(hx0)
        h_bars = VGroup(*[prob_bar(p, y, hx0, HARD_COLOR) for p, y in zip(HARD, ROWS)])
        h_vals = VGroup(*[prob_val(t, b) for t, b in zip(["100 %", "0 %", "0 %"], h_bars)])
        self.at("hard")
        self.play(FadeIn(h_head), FadeIn(h_words), Create(h_axis), GrowFromEdge(h_bars[0], LEFT), FadeIn(h_vals),
                  run_time=0.7)
        self.add(h_bars[1], h_bars[2])
        self.at("top")
        self.play(Indicate(VGroup(h_words[0], h_bars[0], h_vals[0]), scale_factor=1.08), run_time=0.5)

        s_head = label("softmax", YELLOW, 30).move_to([SOFT_X0 + 1.6, HEAD_Y, 0])
        s_words = word_col(SOFT_X0 - 0.25)
        s_axis = base_line(SOFT_X0)
        s_bars = VGroup(*[prob_bar(p, y, SOFT_X0) for p, y in zip(P3, ROWS)])
        s_vals = VGroup(*[prob_val(pct(p), b) for p, b in zip(P3, s_bars)])
        self.at("soft")
        self.play(FadeIn(s_head), FadeIn(s_words), Create(s_axis),
                  LaggedStart(*[AnimationGroup(GrowFromEdge(b, LEFT), FadeIn(v)) for b, v in zip(s_bars, s_vals)],
                              lag_ratio=0.15), run_time=0.8)
        self.at("toward")
        self.play(Indicate(VGroup(s_words[0], s_bars[0], s_vals[0]), scale_factor=1.08), run_time=0.5)
        self.at("others")
        self.play(Indicate(VGroup(s_words[1:], s_bars[1:], s_vals[1:]), scale_factor=1.08), run_time=0.6)
        caption = rich("a soft version of the max", [("soft", YELLOW)], font_size=34).move_to([0, -2.6, 0])
        self.at("soft")
        self.play(FadeIn(caption, shift=0.15 * UP), run_time=0.5)
        self.panel = dict(head=s_head, words=s_words, axis=s_axis, bars=s_bars, vals=s_vals)
        self.end_section()

    # 6. Only differences matter -------------------------------------------------------------------
    def s6_differences(self):
        self.section(6)
        p = self.panel
        panel = VGroup(p["head"], p["words"], p["axis"], p["bars"], p["vals"])
        self.play(*self.clear_anims(keep=[panel]), panel.animate.shift(S6_SHIFT * LEFT), run_time=0.45)

        head = label("only the differences matter", WHITE, 34).move_to([0, 3.0, 0])
        s_head = label("score", WHITE, 30).move_to([SCORE_X, HEAD_Y, 0])
        s_vals = score_vals(SCORES)
        braces = gap_braces(SCORE_X - 0.35)
        gaps = gap_labels(braces)
        self.at("differences")
        self.play(FadeIn(head), FadeIn(s_head), FadeIn(s_vals), run_time=0.4)
        self.play(FadeIn(braces), FadeIn(gaps), run_time=0.4)

        s_head2 = rich("score + 10", [("+ 10", YELLOW)], font_size=30).move_to(s_head)
        s_vals2 = score_vals(SCORES + 10)
        self.at("10")
        self.play(FadeTransform(s_head, s_head2), *[FadeTransform(a, b) for a, b in zip(s_vals, s_vals2)],
                  run_time=0.6)
        self.at("score")
        self.play(Indicate(gaps, color=YELLOW, scale_factor=1.3), run_time=0.45)
        self.at("probabilities")
        self.play(Indicate(p["vals"], color=TEAL_A, scale_factor=1.1), run_time=0.5)
        same = label("unchanged  ✓", GREEN, 30).move_to([SOFT_X0 - S6_SHIFT + 1.6, -2.3, 0])
        self.at("change")
        self.play(FadeIn(same, shift=0.15 * UP), run_time=0.4)
        self.s6 = dict(head=head, s_head=s_head2, s_vals=s_vals2, braces=braces, gaps=gaps, same=same)
        self.end_section()

    # 7. Temperature --------------------------------------------------------------------------------
    def s7_temperature(self):
        self.section(7)
        p, s6 = self.panel, self.s6
        x0 = SOFT_X0 - S6_SHIFT
        temp = ValueTracker(1.0)

        def live_scores():
            return score_vals(SCORES / temp.get_value())

        def live_gaps():
            return gap_labels(s6["braces"], temp.get_value())

        def live_bars():
            probs = softmax(SCORES, temp.get_value())
            bars = VGroup(*[prob_bar(q, y, x0) for q, y in zip(probs, ROWS)])
            return VGroup(bars, VGroup(*[prob_val(pct(q), b) for q, b in zip(probs, bars)]))

        l_scores, l_gaps, l_bars = always_redraw(live_scores), always_redraw(live_gaps), always_redraw(live_bars)
        # identical at T = 1: swap the static bars and gap labels for live ones without a visible change
        self.remove(p["bars"], p["vals"], s6["gaps"])
        self.add(l_bars, l_gaps)
        s_head = label("score", WHITE, 30).move_to(s6["s_head"])
        self.play(FadeOut(s6["head"]), FadeOut(s6["same"]), FadeTransform(s6["s_head"], s_head),
                  FadeOut(s6["s_vals"]), FadeIn(l_scores), run_time=0.5)

        s_head2 = label("score / T", WHITE, 30).move_to(s_head)
        formula = Text("softmax(scores / T)", font=MONO, font_size=30).move_to([-2.4, 3.0, 0])
        self.at("divide")
        self.play(FadeTransform(s_head, s_head2), FadeIn(formula), run_time=0.5)
        readout = always_redraw(lambda: label(f"T = {temp.get_value():.1f}", YELLOW, 40)
                                .move_to([2.3, 3.0, 0], aligned_edge=LEFT))
        self.at("temperature")
        self.play(FadeIn(readout), run_time=0.4)

        cap_pos = [0.4, -2.65, 0]
        low = label("T < 1:  gaps grow → sharper", GREY_A, 30).move_to(cap_pos)
        high = label("T > 1:  gaps shrink → flatter", GREY_A, 30).move_to(cap_pos)
        self.at("below")
        self.play(temp.animate.set_value(0.5), run_time=1.0)
        self.at("grow")
        self.play(FadeIn(low, shift=0.15 * UP), run_time=0.4)
        top_bar = prob_bar(softmax(SCORES, 0.5)[0], ROWS[0], x0)
        top_row = SurroundingRectangle(VGroup(p["words"][0], top_bar, prob_val(pct(softmax(SCORES, 0.5)[0]), top_bar)),
                                       color=YELLOW, buff=0.12, corner_radius=0.08)
        self.at("dominates")
        self.play(Create(top_row), run_time=0.5)
        self.at("above")
        self.play(FadeOut(low), FadeOut(top_row), temp.animate.set_value(2.0), run_time=1.0)
        self.at("flattens")
        self.play(FadeIn(high, shift=0.15 * UP), run_time=0.4)
        self.end_section()

    # 8. The subtract-the-max trick -----------------------------------------------------------------
    def s8_overflow(self):
        self.section(8)
        head = rich("A practical trick:  subtract the max", [("subtract the max", YELLOW)], font_size=34)
        head.move_to([0, 3.2, 0])
        self.play(*self.clear_anims(), run_time=0.5)
        self.at("trick")
        self.play(FadeIn(head), run_time=0.4)

        lx = -4.85
        big = label("e¹⁰⁰⁰ = ∞   (overflow)", RED, 40).move_to([lx, 2.05, 0], aligned_edge=LEFT)
        note = label("largest float64 ≈ e⁷⁰⁹ ≈ 8.2 × 10³⁰⁷", GREY_B, 24).move_to([lx, 1.35, 0], aligned_edge=LEFT)
        self.at("thousand")
        self.play(FadeIn(big, shift=0.15 * UP), run_time=0.5)
        self.at("computer")
        self.play(FadeIn(note), run_time=0.4)

        sub = rich("[3, 2, 0] − 3 = [0, −1, −3]", [("− 3", YELLOW)], font_size=36)
        sub.move_to([lx, 0.25, 0], aligned_edge=LEFT)
        sub_note = label("− the largest score", YELLOW, 24).next_to(sub, RIGHT, buff=0.5)
        self.at("subtracts")
        self.play(FadeIn(sub, shift=0.15 * UP), run_time=0.5)
        self.play(FadeIn(sub_note), run_time=0.3)
        exps = rich(f"eˣ ≈ [{', '.join(f'{v:.2f}' for v in SHIFTED)}]     total ≈ {SHIFTED.sum():.2f}",
                    [("eˣ", GREEN)], font_size=36)
        exps.move_to([lx, -0.75, 0], aligned_edge=LEFT)
        self.at("first")
        self.play(FadeIn(exps, shift=0.15 * UP), run_time=0.5)
        self.at("differences")
        self.play(Indicate(sub, scale_factor=1.06), run_time=0.6)

        probs = rich(f"÷ total → [{', '.join(pct(q) for q in P3)}]   identical ✓", [("identical ✓", GREEN)],
                     font_size=36)
        probs.move_to([lx, -1.75, 0], aligned_edge=LEFT)
        self.at("identical")
        self.play(FadeIn(probs, shift=0.15 * UP), run_time=0.5)
        safe = label("✓  no overflow", GREEN, 38).move_to([0, -2.9, 0])
        self.at("overflows")
        self.play(FadeIn(safe, scale=0.8), run_time=0.5)
        self.end_section()

    # 9. Code ---------------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, 26)
        code.move_to([0, 0.2, 0])
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[1])
        self.play(*self.clear_anims(), FadeIn(code, shift=0.3 * UP), run_time=0.7)
        self.at("temperature")
        self.play(Create(hl), run_time=0.4)
        self.at("exponentiate")
        self.play(highlight(hl, code, 2), run_time=0.4)
        self.at("normalize")
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.end_section()

    # 10. Outro -------------------------------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        card = used_in_card(USED_IN).move_to(ORIGIN)
        rows = card[1]
        self.at("attention")
        self.play(*self.clear_anims(), FadeIn(card, shift=0.2 * UP), run_time=0.6)
        for k, cue in enumerate(["5", "6", "10"]):
            self.at(cue)
            self.play(Indicate(rows[k], scale_factor=1.08), run_time=0.4)
        nxt = next_up_card(NEXT)
        self.at("spread")
        self.play(FadeOut(card), run_time=0.3)
        self.play(FadeIn(nxt, shift=0.2 * UP), run_time=0.5)
        self.end_section()
