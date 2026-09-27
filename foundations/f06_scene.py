"""Foundations F6 — Probability and Sampling.

Render from the repo root:  ./render.sh foundations f06
"""
import numpy as np
from manim import *

from common import code_panel, finish, highlight, next_up_card, token, token_row, used_in_card
from f06_script import LABEL, NEXT, TAGLINE, TITLE, USED_IN
from intro import play_token_intro
from voiced_scene import VoicedScene

# ---------------------------------------------------------------------------- the running example
WORDS = ["mat", "floor", "sofa", "bed", "roof"]
P = np.array([0.4, 0.25, 0.15, 0.1, 0.1])          # illustrative next-word distribution
COLORS = [BLUE_C, TEAL_E, PURPLE_B, MAROON_C, GOLD_E]
WORD_COLOR = dict(zip(WORDS, COLORS))
assert abs(P.sum() - 1.0) < 1e-12

# 10,000 spins of the spinner (section 5), simulated for real
SAMPLES = np.random.default_rng(0).choice(WORDS, size=10000, p=P)
CUM = np.vstack([np.zeros(5), np.cumsum(np.stack([SAMPLES == w for w in WORDS], axis=1), axis=0)])
FREQ = CUM[-1] / 10000
assert [f"{f:.3f}" for f in FREQ] == ["0.400", "0.247", "0.152", "0.104", "0.098"]

# the seeded generator of sections 7-8
_rng = np.random.default_rng(seed=7)
SEED7_ONE = str(_rng.choice(WORDS, p=P))
SEED7_TEN = [str(w) for w in _rng.choice(WORDS, size=10, p=P)]
assert SEED7_ONE == "floor"
assert SEED7_TEN == ["bed", "sofa", "mat", "mat", "bed", "mat", "bed", "sofa", "floor", "mat"]
SEED7_STREAM = [SEED7_ONE] + SEED7_TEN

CODE = """words = ["mat", "floor", "sofa", "bed", "roof"]
p = [0.4, 0.25, 0.15, 0.1, 0.1]           # adds up to 1
rng = np.random.default_rng(seed=7)
rng.choice(words, p=p)                    # 'floor'
rng.choice(words, size=10, p=p)           # ten samples"""


# ---------------------------------------------------------------------------- helpers
def label(text, color=GREY_B, font_size=26, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def caption(text, font_size=20):
    return Text(text, font_size=font_size, color=GREY_B).to_corner(DR, buff=0.3)


def word_token(w, font_size=28):
    return token(w, color=WORD_COLOR[w], font_size=font_size)


def bubble(text, color=BLUE_C, font_size=26, tail_left=True):
    """A chat bubble: rounded box + small tail, with its text."""
    t = Text(text, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.25, width=t.width + 0.6, height=t.height + 0.6,
                           stroke_color=color, fill_color=color, fill_opacity=0.15)
    y = box.get_bottom()[1]
    if tail_left:
        x = box.get_left()[0] + 0.5
        pts = [[x, y, 0], [x - 0.15, y - 0.4, 0], [x + 0.4, y, 0]]
    else:
        x = box.get_right()[0] - 0.5
        pts = [[x, y, 0], [x + 0.15, y - 0.4, 0], [x - 0.4, y, 0]]
    tail = Polygon(*pts, stroke_color=color, fill_color=color, fill_opacity=0.15, stroke_width=2)
    return VGroup(box, tail, t.move_to(box))


class ProbabilityVideo(VoicedScene):
    VIDEO = "f06"

    # ------------------------------------------------------------------ stage helpers
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

    def clear(self, run_time=0.4, keep=()):
        anims = self.clear_anims(keep)
        if anims:
            self.play(*anims, run_time=run_time)

    def construct(self):
        play_token_intro(self, TITLE, 0, TAGLINE, label=LABEL)
        self.s1_probabilities()
        self.s2_number()
        self.s3_distribution()
        self.s4_spinner()
        self.s5_many_spins()
        self.s6_variety()
        self.s7_seed()
        self.s8_code()
        self.s9_outro()
        self.end_section()
        finish(self)

    # 1. An LLM produces probabilities ------------------------------------------------------------
    def s1_probabilities(self):
        self.section(1)
        row = token_row(["The", "cat", "sat", "on", "the"])
        q = token("?", color=YELLOW)
        VGroup(*row, q).arrange(RIGHT, buff=0.12).move_to([-1.0, 2.3, 0])
        self.at("llm")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * UP) for t in row], lag_ratio=0.12), run_time=0.8)
        self.at("word")
        self.play(FadeIn(q, scale=0.6), run_time=0.5)

        base_y, scale, pitch = -2.1, 6.0, 1.0
        xs = [q.get_x() + (i - 2) * pitch for i in range(5)]
        bars = VGroup(*[Rectangle(width=0.6, height=p * scale, stroke_width=0, fill_color=c, fill_opacity=0.85)
                        .move_to([x, base_y, 0], aligned_edge=DOWN) for p, c, x in zip(P, COLORS, xs)])
        names = VGroup(*[label(w, WHITE, 24).move_to([x, base_y - 0.35, 0]) for w, x in zip(WORDS, xs)])
        axis = Line([xs[0] - 0.55, base_y, 0], [xs[-1] + 0.55, base_y, 0], color=GREY_B, stroke_width=2)
        arrow = Arrow(q.get_bottom() + 0.05 * DOWN, [q.get_x(), bars[0].get_top()[1] + 0.25, 0], buff=0.05,
                      color=GREY_B, stroke_width=4)
        what = VGroup(label("probabilities", WHITE, 30), label("for the next word", GREY_B, 28)).arrange(DOWN, buff=0.15)
        what.move_to([axis.get_left()[0] - 0.5, -0.9, 0], aligned_edge=RIGHT)
        self.at("probabilities")
        self.play(GrowArrow(arrow), Create(axis), FadeIn(names), FadeIn(what, shift=0.2 * RIGHT),
                  LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.15), run_time=1.0)
        self.end_section()

    # 2. A number between 0 and 1 ----------------------------------------------------------------
    def s2_number(self):
        self.section(2)
        self.clear()
        y0 = -0.4

        def X(v):
            return -5.0 + 10.0 * v

        line = Line([X(0) - 0.2, y0, 0], [X(1) + 0.2, y0, 0], color=GREY_B, stroke_width=3)
        minor = VGroup(*[Line([X(k / 10), y0 - 0.08, 0], [X(k / 10), y0 + 0.08, 0], color=GREY_C, stroke_width=2)
                         for k in range(1, 10) if k != 5])
        major = VGroup(*[Line([X(v), y0 - 0.18, 0], [X(v), y0 + 0.18, 0], color=WHITE, stroke_width=3)
                         for v in (0, 0.5, 1)])
        n0 = label("0", WHITE, 32).move_to([X(0), y0 - 0.55, 0])
        n1 = label("1", WHITE, 32).move_to([X(1), y0 - 0.55, 0])
        title = Text("a probability is a number between 0 and 1", font_size=36).move_to([0, 2.7, 0])
        self.at("number")
        self.play(FadeIn(title, shift=0.2 * DOWN), Create(line), FadeIn(minor), FadeIn(major[0]), FadeIn(major[2]),
                  FadeIn(n0), FadeIn(n1), run_time=0.8)
        sub = label("how likely something is", GREY_B, 30).next_to(title, DOWN, buff=0.3)
        self.at("likely")
        self.play(FadeIn(sub, shift=0.2 * DOWN), run_time=0.5)

        def mark(v, text, color):
            dot = Dot([X(v), y0, 0], radius=0.12, color=color)
            txt = label(text, color, 30).move_to([X(v), y0 + 0.75, 0])
            return dot, txt

        d0, t0 = mark(0, "impossible", RED)
        self.at("zero")
        self.play(FadeIn(d0, scale=0.5), FadeIn(t0, shift=0.2 * UP), run_time=0.5)
        d1, t1 = mark(1, "certain", GREEN)
        self.at("certain")
        self.play(FadeIn(d1, scale=0.5), FadeIn(t1, shift=0.2 * UP), run_time=0.5)
        dh, th = mark(0.5, "about half the time", YELLOW)
        nh = label("0.5", WHITE, 32).move_to([X(0.5), y0 - 0.55, 0])
        self.at("half")
        self.play(FadeIn(major[1]), FadeIn(nh), FadeIn(dh, scale=0.5), FadeIn(th, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # 3. A distribution adds up to 1 --------------------------------------------------------------
    def s3_distribution(self):
        self.section(3)
        self.clear()
        scale, x0, y0, row_h, bar_h = 10.0, -4.1, 1.8, 0.6, 0.42
        ys = [y0 - i * row_h for i in range(5)]
        names = VGroup(*[label(w, WHITE, 28).move_to([x0 - 0.25, y, 0], aligned_edge=RIGHT)
                         for w, y in zip(WORDS, ys)])
        bars = VGroup(*[Rectangle(width=p * scale, height=bar_h, stroke_width=0, fill_color=c, fill_opacity=0.85)
                        .move_to([x0, y, 0], aligned_edge=LEFT) for p, c, y in zip(P, COLORS, ys)])
        vals = VGroup(*[label(f"{p:.2f}", GREY_A, 28).next_to(b, RIGHT, buff=0.2) for p, b in zip(P, bars)])
        title = Text("a probability distribution", font_size=36).move_to([0, 3.1, 0])
        self.at("distribution")
        self.play(FadeIn(title, shift=0.2 * DOWN), FadeIn(names),
                  LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.12), run_time=0.9)
        sub = label("one whole unit, spread across all the options", GREY_B, 26).next_to(title, DOWN, buff=0.2)
        self.at("unit")
        self.play(FadeIn(sub, shift=0.2 * DOWN), run_time=0.5)

        row = token_row(["The", "cat", "sat", "on", "the"])
        q = token("?", color=YELLOW)
        ctx = VGroup(*row, q).arrange(RIGHT, buff=0.12).move_to([0, 2.85, 0])
        self.at("after")
        self.play(FadeOut(title), FadeOut(sub), LaggedStart(*[FadeIn(t) for t in ctx], lag_ratio=0.08),
                  run_time=0.7)

        self.at("mat")
        self.play(Indicate(names[0], color=YELLOW), run_time=0.5)
        self.at("4")
        self.play(FadeIn(vals[0], shift=0.2 * LEFT), run_time=0.4)
        self.at("floor")
        self.play(Indicate(names[1], color=YELLOW), run_time=0.5)
        self.at("25")
        self.play(FadeIn(vals[1], shift=0.2 * LEFT), run_time=0.4)
        note = caption("illustrative")
        self.at("together")
        self.play(*[FadeIn(v, shift=0.2 * LEFT) for v in vals[2:]], FadeIn(note), run_time=0.6)

        sy = -2.25
        slot = DashedVMobject(Rectangle(width=scale, height=bar_h + 0.1, color=GREY_B, stroke_width=2),
                              num_dashes=60).move_to([0, sy, 0])
        e0 = label("0", GREY_B, 24).next_to(slot, DOWN, buff=0.15).align_to(slot, LEFT)
        e1 = label("1", GREY_B, 24).next_to(slot, DOWN, buff=0.15).align_to(slot, RIGHT)
        self.at("add")
        self.play(Create(slot), FadeIn(e0), FadeIn(e1), run_time=0.6)

        names.set_z_index(3)                           # the moving bars pass behind the row text
        vals.set_z_index(3)
        stack = bars.copy()
        left = -scale / 2
        targets = []
        for b, p in zip(stack, P):
            targets.append(b.animate.move_to([left, sy, 0], aligned_edge=LEFT))
            left += p * scale
        total = Text("sum = 1.00", font_size=34, color=YELLOW).move_to([0, sy - 0.85, 0])
        self.at("one")
        self.play(*targets, FadeIn(total, shift=0.2 * UP), run_time=0.8)
        self.end_section()

    # 4. Sampling = a spinner ---------------------------------------------------------------------
    def s4_spinner(self):
        self.section(4)
        C, R = np.array([-3.0, -0.3, 0.0]), 2.4
        start = 90.0
        slices, labels = VGroup(), VGroup()
        mids = []
        acc = 0.0
        for w, p, c in zip(WORDS, P, COLORS):
            a = 360.0 * p
            s = Sector(radius=R, angle=a * DEGREES, start_angle=(start - acc - a) * DEGREES, arc_center=C)
            s.set_fill(c, 0.6).set_stroke(BLACK, 3)
            slices.add(s)
            m = start - acc - a / 2
            mids.append(m)
            d = np.array([np.cos(m * DEGREES), np.sin(m * DEGREES), 0])
            lab = VGroup(label(w, WHITE, 26), label(f"{p:.2f}", WHITE, 22)).arrange(DOWN, buff=0.08)
            labels.add(lab.move_to(C + 0.64 * R * d))
            acc += a
        theta = ValueTracker(start)

        def needle():
            d = np.array([np.cos(theta.get_value() * DEGREES), np.sin(theta.get_value() * DEGREES), 0])
            return Arrow(C, C + 2.05 * d, buff=0, color=WHITE, stroke_width=8, max_tip_length_to_length_ratio=0.14)

        hand = always_redraw(needle)
        hub = Dot(C, radius=0.13, color=WHITE)
        head = Text("sampling", font_size=40, color=YELLOW).move_to([3.4, 2.4, 0])
        desc = VGroup(label("pick one option at random,", WHITE, 28),
                      label("according to the probabilities", WHITE, 28)).arrange(DOWN, buff=0.15)
        desc.next_to(head, DOWN, buff=0.35)
        self.clear(run_time=0.35)
        self.play(FadeIn(slices), FadeIn(labels), FadeIn(hand), FadeIn(hub), FadeIn(head, shift=0.2 * DOWN),
                  FadeIn(desc), run_time=0.7)

        pick = 1                                       # the needle lands on "floor", clear of its label
        self.at("random")
        self.play(theta.animate.set_value(mids[pick] + 25 - 720), run_time=2.5,
                  rate_func=rate_functions.ease_out_cubic)
        hand.clear_updaters()
        landed = VGroup(label("landed on", GREY_B, 28), word_token(WORDS[pick])).arrange(RIGHT, buff=0.3)
        landed.move_to([3.4, -0.2, 0])
        self.play(FadeIn(landed, shift=0.2 * UP), run_time=0.4)

        ring = Sector(radius=R, angle=360 * P[pick] * DEGREES,
                      start_angle=(mids[pick] - 180 * P[pick]) * DEGREES, arc_center=C)
        ring.set_fill(opacity=0).set_stroke(YELLOW, 6)
        size = label("floor:  0.25 × 360° = 90°", WHITE, 28).move_to([3.4, -1.7, 0])
        size_sub = label("slice size = probability", GREY_B, 26).next_to(size, DOWN, buff=0.25)
        self.at("slice")
        self.play(Create(ring), *[s.animate.set_fill(opacity=0.25) for k, s in enumerate(slices) if k != pick],
                  slices[pick].animate.set_fill(opacity=0.9), FadeIn(size), FadeIn(size_sub), run_time=0.6)
        self.end_section()

    # 5. Many spins → frequencies ≈ probabilities -------------------------------------------------
    def s5_many_spins(self):
        self.section(5)
        base_y, scale, bw = -2.3, 8.0, 1.1
        xs = [-4.0 + 2.0 * i for i in range(5)]
        logn = ValueTracker(1.0)                       # log10 of the number of spins

        def n_now():
            return int(round(10 ** logn.get_value()))

        axis = Line([-5.3, base_y, 0], [5.3, base_y, 0], color=GREY_B, stroke_width=2)
        names = VGroup(*[label(w, WHITE, 28).move_to([x, base_y - 0.4, 0]) for w, x in zip(WORDS, xs)])
        outlines = VGroup(*[Rectangle(width=bw, height=p * scale, stroke_color=WHITE, stroke_width=3, fill_opacity=0)
                            .move_to([x, base_y, 0], aligned_edge=DOWN) for p, x in zip(P, xs)])
        outlines.set_z_index(2)

        def hist():
            f = CUM[n_now()] / n_now()
            return VGroup(*[Rectangle(width=bw, height=max(v * scale, 0.01), stroke_width=0, fill_color=c,
                                      fill_opacity=0.85).move_to([x, base_y, 0], aligned_edge=DOWN)
                            for v, c, x in zip(f, COLORS, xs)])

        def pcts():
            f = CUM[n_now()] / n_now()
            return VGroup(*[label(f"{100 * v:.1f}%", WHITE, 26)
                            .move_to([x, base_y + max(v, p) * scale + 0.3, 0]) for v, p, x in zip(f, P, xs)])

        def counter():
            return Text(f"{n_now():,} spins", font_size=36).move_to([0, 2.8, 0])

        bars, vals, count = always_redraw(hist), always_redraw(pcts), always_redraw(counter)
        legend = VGroup(
            VGroup(Square(0.3, stroke_width=0, fill_color=GREY_B, fill_opacity=0.85), label("how often it came up")),
            VGroup(Square(0.3, stroke_color=WHITE, stroke_width=3, fill_opacity=0), label("its probability")),
        )
        for item in legend:
            item.arrange(RIGHT, buff=0.2)
        legend.arrange(RIGHT, buff=0.8).move_to([0, 2.05, 0])
        note = caption("simulated with numpy · illustrative probabilities")
        self.clear(run_time=0.25)
        self.play(FadeIn(axis), FadeIn(names), FadeIn(outlines), FadeIn(bars), FadeIn(vals),
                  FadeIn(count), FadeIn(legend), FadeIn(note), run_time=0.25)
        self.at("many")
        self.play(logn.animate.set_value(4.0), run_time=5.2, rate_func=linear)
        for m in (bars, vals, count):
            m.clear_updaters()

        box = SurroundingRectangle(VGroup(outlines[0], vals[0], names[0]), color=YELLOW, buff=0.12)
        self.at("40")
        self.play(Create(box), vals[0].animate.set_color(YELLOW), run_time=0.5)
        box2 = SurroundingRectangle(VGroup(outlines[1], vals[1], names[1]), color=YELLOW, buff=0.12)
        self.at("quarter")
        self.play(Transform(box, box2), vals[0].animate.set_color(WHITE), vals[1].animate.set_color(YELLOW),
                  run_time=0.5)
        self.end_section()

    # 6. Greedy vs sampling -----------------------------------------------------------------------
    def s6_variety(self):
        self.section(6)
        self.clear()
        lx, tx = -2.3, -1.95
        g_lab = label("always the biggest:", GREY_B, 28).move_to([lx, 2.5, 0], aligned_edge=RIGHT)
        g_row = VGroup(*[token("mat", color=GREY_D, font_size=24) for _ in range(4)]).arrange(RIGHT, buff=0.12)
        g_row.move_to([tx, 2.5, 0], aligned_edge=LEFT)
        for t in g_row:
            t[1].set_color(GREY_B)
        self.at("biggest")
        self.play(FadeIn(g_lab), LaggedStart(*[FadeIn(t, shift=0.2 * RIGHT) for t in g_row], lag_ratio=0.2),
                  run_time=0.8)
        g_tag = label("predictable", GREY_B, 28).next_to(g_row, RIGHT, buff=0.4)
        self.at("predictable")
        self.play(FadeIn(g_tag), run_time=0.4)

        s_words = ["mat", "floor", "mat", "sofa", "mat", "bed"]
        s_lab = label("sampling:", WHITE, 28).move_to([lx, 1.2, 0], aligned_edge=RIGHT)
        s_row = VGroup(*[word_token(w, font_size=24) for w in s_words]).arrange(RIGHT, buff=0.12)
        s_row.move_to([tx, 1.2, 0], aligned_edge=LEFT)
        s_tag = label("variety", GREEN, 28).next_to(s_row, RIGHT, buff=0.4)
        self.at("variety")
        self.play(FadeIn(s_lab), LaggedStart(*[FadeIn(t, shift=0.2 * RIGHT) for t in s_row], lag_ratio=0.15),
                  FadeIn(s_tag), run_time=0.9)

        prompt = VGroup(label("same prompt:", GREY_B, 28), Text("“The cat sat on the …”", font_size=28))
        prompt.arrange(RIGHT, buff=0.3).move_to([0, -0.3, 0])
        self.at("prompt")
        self.play(FadeIn(prompt, shift=0.2 * UP), run_time=0.5)
        b1 = bubble("The cat sat on the mat.", BLUE_C).move_to([-3.2, -1.9, 0])
        b2 = bubble("The cat sat on the sofa.", PURPLE_B, tail_left=False).move_to([3.2, -1.9, 0])
        r1 = label("run 1", GREY_B, 24).next_to(b1, DOWN, buff=0.35)
        r2 = label("run 2", GREY_B, 24).next_to(b2, DOWN, buff=0.35)
        self.at("different")
        self.play(FadeIn(b1, shift=0.2 * UP), FadeIn(b2, shift=0.2 * UP), FadeIn(r1), FadeIn(r2), run_time=0.7)
        self.end_section()

    # 7. Pseudo-random generators and seeds -------------------------------------------------------
    def s7_seed(self):
        self.section(7)
        self.clear()

        def generator(y):
            box = RoundedRectangle(corner_radius=0.15, width=3.2, height=1.7, stroke_color=BLUE_C,
                                   fill_color=BLUE_C, fill_opacity=0.15)
            name = VGroup(label("pseudo-random", WHITE, 26), label("generator", WHITE, 26)).arrange(DOWN, buff=0.1)
            seed = label("seed = 7", YELLOW, 28)
            inner = VGroup(name, seed).arrange(DOWN, buff=0.2)
            box.move_to([-4.4, y, 0])
            inner.move_to(box)
            return VGroup(box, name, seed)

        def outputs(gen):
            toks = VGroup(*[word_token(w) for w in SEED7_STREAM[:4]], token("…", color=GREY_D))
            toks.arrange(RIGHT, buff=0.12).next_to(gen, RIGHT, buff=0.9)
            arr = Arrow(gen.get_right(), toks.get_left(), buff=0.12, color=GREY_B, stroke_width=4)
            return arr, toks

        g1 = generator(1.5)
        a1, o1 = outputs(g1)
        self.at("pseudo")
        self.play(FadeIn(g1, shift=0.2 * RIGHT), run_time=0.5)
        self.play(GrowArrow(a1), LaggedStart(*[FadeIn(t, shift=0.4 * RIGHT) for t in o1], lag_ratio=0.25),
                  run_time=1.2)
        self.at("start")
        self.play(Indicate(g1[2], scale_factor=1.2), run_time=0.5)

        g2 = generator(-0.8)
        a2, o2 = outputs(g2)
        self.at("same")
        self.play(FadeIn(g2, shift=0.2 * RIGHT), GrowArrow(a2),
                  LaggedStart(*[FadeIn(t, shift=0.4 * RIGHT) for t in o2], lag_ratio=0.2), run_time=1.1)
        check = VGroup(Text("✓", font_size=48, color=GREEN), label("identical", GREEN, 28)).arrange(RIGHT, buff=0.2)
        check.next_to(o2, RIGHT, buff=0.35)
        self.at("same")
        self.play(FadeIn(check, scale=0.7), run_time=0.5)
        self.at("every")
        self.play(Indicate(o1, scale_factor=1.05), Indicate(o2, scale_factor=1.05), run_time=0.6)
        cap = Text("same seed → same results", font_size=36, color=YELLOW).move_to([0, -2.8, 0])
        self.at("reproducible")
        self.play(FadeIn(cap, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # 8. The NumPy code ---------------------------------------------------------------------------
    def s8_code(self):
        self.section(8)
        code, hl = code_panel(CODE, font_size=26)
        code.move_to([0, 1.3, 0])
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[2])
        self.clear(run_time=0.35)
        self.play(FadeIn(code, shift=0.3 * UP), run_time=0.5)
        self.at("seed")
        self.play(Create(hl), run_time=0.4)

        lx = -5.8
        one_lab = label("one:", GREY_B, 24).move_to([lx, -1.0, 0], aligned_edge=LEFT)
        one = word_token(SEED7_ONE, font_size=24).next_to(one_lab, RIGHT, buff=0.3)
        self.at("one")
        self.play(highlight(hl, code, 3), FadeIn(one_lab), FadeIn(one, shift=0.2 * UP), run_time=0.5)
        ten_lab = label("ten:", GREY_B, 24).move_to([lx, -2.2, 0], aligned_edge=LEFT)
        ten = VGroup(*[word_token(w, font_size=24) for w in SEED7_TEN]).arrange(RIGHT, buff=0.1)
        ten.next_to(ten_lab, RIGHT, buff=0.3)
        self.at("many")
        self.play(highlight(hl, code, 4), FadeIn(ten_lab),
                  LaggedStart(*[FadeIn(t, shift=0.2 * UP) for t in ten], lag_ratio=0.1), run_time=1.0)
        self.end_section()

    # 9. Outro ------------------------------------------------------------------------------------
    def s9_outro(self):
        self.section(9)
        card = used_in_card(USED_IN).move_to(ORIGIN)
        rows = card[1]
        self.at("see")
        self.play(*self.clear_anims(), FadeIn(card, shift=0.2 * UP), run_time=0.7)
        for k, cue in enumerate(["1", "10", "14"]):
            self.at(cue)
            self.play(Indicate(rows[k], scale_factor=1.08), run_time=0.5)
        nxt = next_up_card(NEXT)
        self.at("probabilities")
        self.play(FadeOut(card), FadeIn(nxt, shift=0.2 * UP), run_time=0.7)
