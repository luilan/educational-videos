"""Video 8 — The MLP: Where Facts Live.

Render from the repo root:  ./render.sh how-llms-work v08
"""
import numpy as np
from manim import *

from common import code_panel, finish, highlight, next_up_card, token, token_row
from intro import play_token_intro
from v08_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

VEC_COLOR = BLUE_C      # token vectors
MLP_COLOR = GREEN_C     # MLP accent
NEG_COLOR = RED_C       # negative hidden values
MONO = "DejaVu Sans Mono"

CODE = """def gelu(x):
    c = np.sqrt(2 / np.pi)
    return 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x**3)))

def mlp(x, W1, b1, W2, b2):
    h = gelu(x @ W1 + b1)      # expand 768 -> 3072, then bend
    return h @ W2 + b2         # project back 3072 -> 768"""


# ---------------------------------------------------------------------------- helpers
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def caption(text, corner=DR, font_size=22):
    return Text(text, font_size=font_size, color=GREY_B).to_corner(corner, buff=0.3)


def vec_col(values, color=VEC_COLOR, cell=0.26, neg_color=NEG_COLOR):
    """A column vector drawn as stacked cells; fill strength ~ |value|."""
    g = VGroup(*[Square(cell, stroke_color=GREY_B, stroke_width=1.2,
                        fill_color=(color if v >= 0 else neg_color),
                        fill_opacity=0.08 + 0.8 * min(abs(v), 1.0)) for v in values])
    return g.arrange(DOWN, buff=0)


def mlp_box(width=1.8, height=0.8, font_size=34, text="MLP", color=MLP_COLOR):
    box = RoundedRectangle(corner_radius=0.12, width=width, height=height, stroke_color=color,
                           fill_color=color, fill_opacity=0.25)
    return VGroup(box, Text(text, font_size=font_size).move_to(box))


def mat_box(text, side=1.1, font_size=34, color=MLP_COLOR):
    box = RoundedRectangle(corner_radius=0.08, width=side, height=side, stroke_color=color,
                           fill_color=GREEN_E, fill_opacity=0.35)
    return VGroup(box, Text(text, font_size=font_size).move_to(box))


def gelu_icon(width=1.1, height=0.9, color=MLP_COLOR):
    frame = RoundedRectangle(corner_radius=0.1, width=width, height=height, stroke_color=color,
                             stroke_width=3, fill_color=BLACK, fill_opacity=1)
    xs = np.linspace(-3, 2, 60)
    curve = VMobject(stroke_color=WHITE, stroke_width=4).set_points_smoothly([[x, gelu(x), 0] for x in xs])
    base = Line([-3, 0, 0], [2, 0, 0], stroke_color=GREY_C, stroke_width=2)
    g = VGroup(base, curve)
    g.stretch_to_fit_width(width * 0.72).stretch_to_fit_height(height * 0.6).move_to(frame)
    return VGroup(frame, g)


def book_icon(color=GOLD_C):
    def page(side):
        pts = [[0, -0.24, 0], [side * 0.55, -0.14, 0], [side * 0.55, 0.36, 0], [0, 0.26, 0]]
        return Polygon(*pts, stroke_color=color, stroke_width=3, fill_color=color, fill_opacity=0.2)

    lines = VGroup()
    for side in (-1, 1):
        for k in range(3):
            y = 0.17 - 0.13 * k
            lines.add(Line([side * 0.1, y, 0], [side * 0.44, y + 0.07, 0], stroke_color=color, stroke_width=2))
    return VGroup(page(-1), page(1), lines)


def neuron_col(x, ys, radius=0.2, dots_y=None):
    circles = VGroup(*[Circle(radius=radius, stroke_color=MLP_COLOR, stroke_width=3, fill_color=GREEN_E,
                              fill_opacity=0.4).move_to([x, y, 0]) for y in ys])
    dots = Text("⋮", font_size=32, color=GREY_B).move_to([x, dots_y, 0]) if dots_y is not None else VGroup()
    return circles, dots


def arrow(start, end, color=GREY_B, width=4, ratio=0.25):
    return Arrow(start, end, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=ratio,
                 max_stroke_width_to_length_ratio=10)


def vec_arrow(start, end, color, width=6):
    return Arrow(start, end, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.12,
                 max_stroke_width_to_length_ratio=10)


def arc_over(p, q, color=GREY_B, width=2.5, angle=-0.6 * PI, opacity=0.9):
    return ArcBetweenPoints(p, q, angle=angle, stroke_color=color, stroke_width=width, stroke_opacity=opacity)


class MLPVideo(VoicedScene):
    VIDEO = "v08"

    def section(self, i):
        super().section(i)
        print(f"[v08] section {i} starts at {self.renderer.time:.2f}s")

    def clear_anims(self):
        return [FadeOut(m) for m in list(self.mobjects)]

    def construct(self):
        play_token_intro(self, TITLE, 8, TAGLINE)
        self.s1_recap()
        self.s2_three_steps()
        self.s3_sizes()
        self.s4_neurons()
        self.s5_gelu()
        self.s6_bend()
        self.s7_facts()
        self.s8_weights()
        self.s9_code()
        self.s10_outro()

    # 1. Recap: attention mixes, the MLP works on each token alone -------------------------------
    def s1_recap(self):
        self.section(1)
        words = ["The", "cat", "sat", "on", "the"]
        xs = [-4.4, -2.2, 0.0, 2.2, 4.4]
        toks = VGroup(*[token(w).move_to([x, 0.9, 0]) for w, x in zip(words, xs)])
        rng = np.random.default_rng(8)
        cols = VGroup(*[vec_col(rng.uniform(0.1, 1, 6), cell=0.3).next_to(t, DOWN, buff=0.25) for t in toks])
        vec_lab = Text("token vectors", font_size=24, color=GREY_B).move_to([0, -1.95, 0])
        self.play(LaggedStart(*[FadeIn(VGroup(t, c), shift=0.2 * DOWN) for t, c in zip(toks, cols)],
                              lag_ratio=0.12), FadeIn(vec_lab), run_time=0.9)

        pairs = [(0, 1), (1, 2), (0, 2), (2, 3), (1, 4), (3, 4)]
        arcs = VGroup(*[arc_over(toks[i].get_top() + 0.1 * UP, toks[j].get_top() + 0.1 * UP) for i, j in pairs])
        self.at("between")
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.15), run_time=0.9)

        big = mlp_box(2.0, 0.8, 36).move_to([0, 2.75, 0])
        self.at("mlp")
        self.play(FadeOut(arcs), FadeIn(big, shift=0.2 * DOWN), run_time=0.6)

        copies = VGroup(*[mlp_box(1.3, 0.55, 26).move_to([x, 1.95, 0]) for x in xs])
        self.at("itself")
        self.play(*[TransformFromCopy(big, c) for c in copies], FadeOut(big), run_time=0.8)

        # one at a time: each copy processes its own vector
        new_cols = VGroup(*[vec_col(rng.uniform(0.1, 1, 6), cell=0.3).move_to(c) for c in cols])
        self.at("one")
        self.play(LaggedStart(*[AnimationGroup(c[0].animate(rate_func=there_and_back).set_stroke(YELLOW, 6),
                                               Transform(col, nc))
                                for c, col, nc in zip(copies, cols, new_cols)], lag_ratio=0.5), run_time=1.3)

        eqs = VGroup(*[Text("=", font_size=34, color=MLP_COLOR).move_to([(xs[k] + xs[k + 1]) / 2, 1.95, 0])
                       for k in range(4)])
        self.at("same")
        self.play(FadeIn(eqs), *[c[0].animate(rate_func=there_and_back).set_fill(MLP_COLOR, 0.6) for c in copies],
                  run_time=0.6)
        same = Text("same weights at every position", font_size=30, color=YELLOW).move_to([0, -2.85, 0])
        self.at("position")
        self.play(FadeIn(same, shift=0.15 * UP), run_time=0.6)

        facts = VGroup(book_icon(), Text("facts?", font_size=30, color=GOLD_A)).arrange(RIGHT, buff=0.3)
        facts.move_to([0, 3.15, 0])
        self.at("knowledge")
        self.play(FadeIn(facts, shift=0.2 * DOWN), run_time=0.7)
        self.at("live")
        self.play(facts[0].animate(rate_func=there_and_back).scale(1.2), run_time=0.5)
        self.end_section()

    # 2. Three steps: expand, bend, project ------------------------------------------------------
    def s2_three_steps(self):
        self.section(2)
        rng = np.random.default_rng(21)
        xin, xh, xout = rng.uniform(0.1, 1, 4), rng.uniform(-1, 1, 16), rng.uniform(0.1, 1, 4)
        xh2 = gelu(2.5 * xh) / 2.5
        self.cin = cin = vec_col(xin).move_to([-5.4, 0, 0])
        self.ch = ch = vec_col(xh, color=MLP_COLOR).move_to([-2.2, 0, 0])
        self.icon = icon = gelu_icon().move_to([-0.3, 0, 0])
        self.ch2 = ch2 = vec_col(xh2, color=MLP_COLOR).move_to([1.6, 0, 0])
        self.cout = cout = vec_col(xout).move_to([5.2, 0, 0])
        a1 = arrow(cin.get_right() + 0.15 * RIGHT, ch.get_left() + 0.15 * LEFT)
        a2 = arrow(ch.get_right() + 0.12 * RIGHT, icon.get_left() + 0.08 * LEFT)
        a3 = arrow(icon.get_right() + 0.08 * RIGHT, ch2.get_left() + 0.12 * LEFT)
        a4 = arrow(ch2.get_right() + 0.15 * RIGHT, cout.get_left() + 0.15 * LEFT)

        self.play(*self.clear_anims(), run_time=0.5)
        self.play(FadeIn(cin, shift=0.2 * RIGHT), run_time=0.5)
        self.title2 = title = Text("MLP = multi-layer perceptron", font_size=36).move_to([0, 3.2, 0])
        title[:3].set_color(MLP_COLOR)
        self.at("multi")
        self.play(Write(title), run_time=0.8)

        steps = VGroup(Text("1. expand", font_size=26, color=GREY_B).move_to([a1.get_x(), -0.8, 0]),
                       Text("2. bend", font_size=26, color=GREY_B).move_to([icon.get_x(), -0.8, 0]),
                       Text("3. project", font_size=26, color=GREY_B).move_to([a4.get_x(), -0.8, 0]))
        self.at("three")
        self.play(LaggedStart(*[FadeIn(s, shift=0.1 * UP) for s in steps], lag_ratio=0.25), run_time=0.7)

        self.at("expand")
        self.play(steps[0].animate.set_color(YELLOW), GrowArrow(a1), TransformFromCopy(cin, ch), run_time=1.0)
        four = Text("4×", font_size=40, color=YELLOW).move_to([a1.get_x(), 1.0, 0])
        self.at("four")
        self.play(FadeIn(four, scale=1.3), run_time=0.4)

        self.at("non")
        self.play(steps[0].animate.set_color(GREY_B), steps[1].animate.set_color(YELLOW),
                  GrowArrow(a2), FadeIn(icon, scale=0.8), GrowArrow(a3), TransformFromCopy(ch, ch2), run_time=0.9)

        self.at("project")
        self.play(steps[1].animate.set_color(GREY_B), steps[2].animate.set_color(YELLOW),
                  GrowArrow(a4), TransformFromCopy(ch2, cout), run_time=0.7)
        self.end_section()

    # 3. Sizes in GPT-2 small -------------------------------------------------------------------
    def s3_sizes(self):
        self.section(3)
        gpt = Text("GPT-2 small", font_size=36).move_to([0, 3.2, 0])
        self.play(FadeTransform(self.title2, gpt), run_time=0.6)

        def size_label(text, col):
            return Text(text, font_size=30, color=YELLOW).next_to(col, UP, buff=0.25)

        l_in = size_label("768", self.cin)
        self.at("768")
        self.play(FadeIn(l_in, shift=0.15 * DOWN), run_time=0.5)
        l_h = VGroup(size_label("3,072", self.ch), size_label("3,072", self.ch2))
        self.at("072")
        self.play(FadeIn(l_h, shift=0.15 * DOWN), l_in.animate.set_color(WHITE), run_time=0.5)
        l_out = size_label("768", self.cout)
        self.at("768")
        self.play(FadeIn(l_out, shift=0.15 * DOWN), l_h.animate.set_color(WHITE), run_time=0.5)
        self.end_section()

    # 4. Neurons as detectors --------------------------------------------------------------------
    def s4_neurons(self):
        self.section(4)
        self.play(*self.clear_anims(), run_time=0.5)
        rng = np.random.default_rng(4)
        xcol = vec_col(rng.uniform(0.1, 1, 6), cell=0.3).move_to([-5.8, -0.2, 0])
        xlab = Text("x", font_size=30, color=VEC_COLOR).next_to(xcol, UP, buff=0.2)
        ys = [1.75, 1.19, 0.63, 0.07, -0.49, -1.61, -2.17]
        circles, dots = neuron_col(-3.6, ys, dots_y=-1.05)
        head = Text("3,072 neurons", font_size=26, color=GREY_B).move_to([-3.6, 2.5, 0])
        wires = VGroup()
        for c in circles:
            for cell in xcol:
                wires.add(Line(cell.get_right(), c.get_left(), stroke_color=GREY, stroke_width=1, stroke_opacity=0.35))
        self.play(FadeIn(xcol), FadeIn(xlab), FadeIn(head),
                  LaggedStart(*[FadeIn(c, scale=0.6) for c in circles], lag_ratio=0.08), FadeIn(dots),
                  Create(wires), run_time=1.0)

        k1, k2 = 1, 3
        mine = VGroup(*[w for i, w in enumerate(wires) if i // 6 == k1])
        self.at("neuron")
        self.play(circles[k1].animate.set_fill(YELLOW, 0.8).set_stroke(YELLOW),
                  mine.animate.set_stroke(YELLOW, 2, 0.8), run_time=0.5)

        # 2-D inset: the neuron's weight direction w versus the token vector x (illustrative)
        O = np.array([2.2, -1.5, 0])
        hline = Line(O + 1.2 * LEFT, O + 4.2 * RIGHT, stroke_color=GREY_D, stroke_width=2)
        vline = Line(O + 0.9 * DOWN, O + 3.6 * UP, stroke_color=GREY_D, stroke_width=2)
        wv, xv = np.array([3.4, 1.2, 0]), np.array([1.6, 2.8, 0])
        w_arr = vec_arrow(O, O + wv, MLP_COLOR)
        w_lab = Text("w", font_size=30, color=MLP_COLOR).next_to(O + wv, RIGHT, buff=0.15)
        eq = Text("neuron = w · x", font_size=32).move_to([3.6, 2.85, 0])
        eq[:6].set_color(YELLOW)
        eq[7].set_color(MLP_COLOR)
        eq[9].set_color(VEC_COLOR)
        self.at("dot")
        self.play(FadeIn(eq, shift=0.15 * DOWN), Create(hline), Create(vline), GrowArrow(w_arr), FadeIn(w_lab),
                  run_time=0.8)
        x_arr = vec_arrow(O, O + xv, VEC_COLOR)
        x_lab = Text("x", font_size=30, color=VEC_COLOR).next_to(O + xv, UL, buff=0.1)
        self.at("vector")
        self.play(GrowArrow(x_arr), FadeIn(x_lab), run_time=0.6)

        w_hat = wv / np.linalg.norm(wv)
        foot = O + np.dot(xv, w_hat) * w_hat
        drop = DashedLine(O + xv, foot, stroke_color=GREY_B, stroke_width=2.5, dash_length=0.1)
        proj = Line(O, foot, stroke_color=YELLOW, stroke_width=9)
        self.at("measures")
        self.play(Create(drop), Create(proj), run_time=0.8)
        how = Text("w · x: how far x points along w", font_size=24, color=GREY_A).move_to([3.6, -2.95, 0])
        self.at("direction")
        self.play(FadeIn(how, shift=0.1 * UP), run_time=0.6)

        q = Text("?", font_size=32, color=YELLOW).next_to(circles[k1], RIGHT, buff=0.25)
        self.at("question")
        self.play(FadeIn(q, scale=1.4), run_time=0.4)
        animal = Text("animal?", font_size=28, color=YELLOW).next_to(circles[k1], RIGHT, buff=0.25)
        illus = caption("illustrative", corner=DL)
        self.at("animals")
        self.play(FadeTransform(q, animal), FadeIn(illus), run_time=0.5)
        eos = Text("end of sentence?", font_size=28, color=YELLOW).next_to(circles[k2], RIGHT, buff=0.25)
        self.at("end")
        self.play(circles[k2].animate.set_fill(YELLOW, 0.8).set_stroke(YELLOW), FadeIn(eos, shift=0.1 * RIGHT),
                  run_time=0.5)
        self.end_section()

    # 5. GELU ------------------------------------------------------------------------------------
    def s5_gelu(self):
        self.section(5)
        self.play(*self.clear_anims(), run_time=0.5)
        axes = Axes(x_range=[-4, 4, 1], y_range=[-1, 4, 1], x_length=8, y_length=5, tips=False,
                    axis_config={"color": GREY_B, "stroke_width": 2, "tick_size": 0.06}).move_to([-0.8, -0.35, 0])
        ticks = VGroup(*[Text(s, font_size=20, color=GREY_B).move_to(axes.c2p(v, 0) + 0.3 * DOWN)
                         for s, v in [("−4", -4), ("−2", -2), ("2", 2), ("4", 4)]],
                       *[Text(s, font_size=20, color=GREY_B).move_to(axes.c2p(0, v) + 0.3 * LEFT)
                         for s, v in [("2", 2), ("4", 4)]])
        sub = Text("the nonlinear function", font_size=26, color=GREY_B).move_to([0, 2.7, 0])
        self.play(FadeIn(axes), FadeIn(ticks), FadeIn(sub), run_time=0.6)

        curve = axes.plot(gelu, x_range=[-4, 4], color=MLP_COLOR, stroke_width=6)
        relu = DashedVMobject(axes.plot(lambda x: max(x, 0.0), x_range=[-4, 4], use_smoothing=False),
                              num_dashes=40).set_stroke(GREY_C, 2, 0.7)
        legend = Text("dashed: ReLU, for comparison", font_size=20, color=GREY_B).to_corner(DR, buff=0.3)
        title = Text("GELU", font_size=44, color=MLP_COLOR).move_to([0, 3.3, 0])
        self.at("jell")
        self.play(FadeIn(title, shift=0.15 * DOWN), Create(relu), Create(curve), FadeIn(legend), run_time=1.0)

        pos = axes.plot(gelu, x_range=[1, 4], color=YELLOW, stroke_width=9)
        pos_lab = Text("positive:\npasses through, ≈ x", font_size=24, color=YELLOW, line_spacing=0.8)
        pos_lab.move_to([4.95, 0.9, 0])
        self.at("positive")
        self.play(Create(pos), FadeIn(pos_lab, shift=0.1 * LEFT), run_time=0.7)

        neg = axes.plot(gelu, x_range=[-4, -1], color=YELLOW, stroke_width=9)
        neg_lab = Text("negative: squashed to ≈ 0", font_size=24, color=YELLOW).move_to([-3.4, -0.5, 0])
        self.at("negative")
        self.play(FadeOut(pos), pos_lab.animate.set_color(GREY_A), Create(neg), FadeIn(neg_lab, shift=0.1 * DOWN),
                  run_time=0.7)

        dot_q = Dot(axes.c2p(-2, gelu(-2)), radius=0.1, color=YELLOW)
        quiet = Text("quiet: ≈ −0.05", font_size=24, color=YELLOW).move_to(axes.c2p(-2, 0) + 0.5 * UP)
        self.at("quiet")
        self.play(FadeOut(neg), neg_lab.animate.set_color(GREY_A), FadeIn(dot_q, scale=1.5), FadeIn(quiet),
                  run_time=0.5)
        dot_f = Dot(axes.c2p(2.5, gelu(2.5)), radius=0.1, color=YELLOW)
        fires = Text("fires: ≈ 2.5", font_size=24, color=YELLOW).next_to(dot_f, RIGHT, buff=0.3).shift(0.35 * DOWN)
        self.at("shows")
        self.play(FadeIn(dot_f, scale=1.5), FadeIn(fires), run_time=0.5)
        self.end_section()

    # 6. Why the bend matters --------------------------------------------------------------------
    def s6_bend(self):
        self.section(6)
        self.play(*self.clear_anims(), run_time=0.5)
        header = Text("without the bend (no GELU)", font_size=30, color=GREY_B).move_to([0, 3.2, 0])
        self.at("bend")
        self.play(FadeIn(header, shift=0.15 * DOWN), run_time=0.5)

        f1 = Text("W₂ (W₁ x)", font_size=36)
        f2 = Text("= (W₂ W₁) x", font_size=36)
        f3 = Text("= W x", font_size=36)
        VGroup(f1, f2, f3).arrange(RIGHT, buff=0.4).move_to([0, 2.2, 0])
        rng = np.random.default_rng(6)
        xcol = vec_col(rng.uniform(0.1, 1, 4)).move_to([-3.3, 0.3, 0])
        ocol = vec_col(rng.uniform(0.1, 1, 4)).move_to([3.3, 0.3, 0])
        w1 = mat_box("W₁").move_to([-1.2, 0.3, 0])
        w2 = mat_box("W₂").move_to([1.2, 0.3, 0])
        b1 = arrow(xcol.get_right() + 0.12 * RIGHT, w1.get_left() + 0.08 * LEFT)
        b2 = arrow(w1.get_right() + 0.08 * RIGHT, w2.get_left() + 0.08 * LEFT)
        b3 = arrow(w2.get_right() + 0.08 * RIGHT, ocol.get_left() + 0.12 * LEFT)
        self.at("two")
        self.play(FadeIn(f1), FadeIn(xcol), GrowArrow(b1), FadeIn(w1), GrowArrow(b2), FadeIn(w2), GrowArrow(b3),
                  FadeIn(ocol), run_time=0.8)
        self.at("collapse")
        self.play(FadeIn(f2, shift=0.15 * LEFT), run_time=0.6)

        w = mat_box("W", side=1.3).move_to([0, 0.3, 0])
        w_note = Text("W = W₂ W₁", font_size=24, color=GREY_B).next_to(w, DOWN, buff=0.2)
        c1 = arrow(xcol.get_right() + 0.12 * RIGHT, w.get_left() + 0.08 * LEFT)
        c3 = arrow(w.get_right() + 0.08 * RIGHT, ocol.get_left() + 0.12 * LEFT)
        self.at("single")
        self.play(ReplacementTransform(VGroup(w1, w2), w), FadeOut(b2), Transform(b1, c1), Transform(b3, c3),
                  FadeIn(f3, shift=0.15 * LEFT), FadeIn(w_note), run_time=0.8)

        stack = VGroup(*[mat_box(f"W{s}", side=0.8, font_size=28) for s in "₁₂₃₄"]).arrange(RIGHT, buff=0.55)
        stack.move_to([-1.3, -1.85, 0])
        s_arrows = VGroup(*[arrow(stack[k].get_right() + 0.06 * RIGHT, stack[k + 1].get_left() + 0.06 * LEFT,
                                  width=3, ratio=0.35) for k in range(3)])
        self.at("stacking")
        self.play(LaggedStart(*[FadeIn(m, shift=0.15 * UP) for m in stack], lag_ratio=0.2),
                  FadeIn(s_arrows), run_time=0.6)
        just = Text("= still just one matrix", font_size=26, color=GREY_B).next_to(stack, RIGHT, buff=0.4)
        self.at("nothing")
        self.play(*[m[0].animate.set_stroke(GREY_C).set_fill(GREY_D, 0.3) for m in stack],
                  *[m[1].animate.set_color(GREY_C) for m in stack], s_arrows.animate.set_color(GREY_D),
                  FadeIn(just, shift=0.1 * LEFT), run_time=0.7)

        # with vs without the bend: fitting curvy data
        def f(t):
            return 0.75 * np.sin(1.7 * t) + 0.25 * t

        ts = np.linspace(-2.0, 2.0, 13)
        noise = np.random.default_rng(5).normal(0, 0.07, len(ts))
        panels = VGroup()
        for cx in (-3.3, 3.3):
            c = np.array([cx, -0.3, 0])
            frame = RoundedRectangle(corner_radius=0.15, width=5.0, height=3.2, stroke_color=GREY_D, stroke_width=2)
            dots = VGroup(*[Dot(c + np.array([t, f(t) + n, 0]), radius=0.06, color=GREY_B) for t, n in zip(ts, noise)])
            panels.add(VGroup(frame.move_to(c), dots))
        t_left = Text("no bend", font_size=28).next_to(panels[0], UP, buff=0.25)
        t_right = Text("with the bend", font_size=28, color=MLP_COLOR).next_to(panels[1], UP, buff=0.25)
        self.at("nonlinearity")
        self.play(*self.clear_anims(), FadeIn(panels), FadeIn(t_left), FadeIn(t_right), run_time=0.8)

        c_r = np.array([3.3, -0.3, 0])
        bent = ParametricFunction(lambda t: c_r + np.array([t, f(t), 0]), t_range=[-2.15, 2.15],
                                  stroke_color=MLP_COLOR, stroke_width=5)
        more = Text("curves: more than straight lines", font_size=24, color=MLP_COLOR).next_to(panels[1], DOWN, buff=0.3)
        self.at("learn")
        self.play(Create(bent), FadeIn(more), run_time=0.7)

        c_l = np.array([-3.3, -0.3, 0])
        yv = np.array([f(t) + n for t, n in zip(ts, noise)])
        slope, icpt = np.polyfit(ts, yv, 1)
        straight = Line(c_l + np.array([-2.2, icpt - 2.2 * slope, 0]), c_l + np.array([2.2, icpt + 2.2 * slope, 0]),
                        stroke_color=RED_C, stroke_width=5)
        only = Text("only straight lines", font_size=24, color=RED_C).next_to(panels[0], DOWN, buff=0.3)
        self.at("straight")
        self.play(Create(straight), FadeIn(only), run_time=0.5)
        self.end_section()

    # 7. Recalling facts -------------------------------------------------------------------------
    def s7_facts(self):
        self.section(7)
        self.play(*self.clear_anims(), run_time=0.5)
        ys = [1.35, 0.73, 0.11, -0.51, -1.13, -1.75]
        circles, _ = neuron_col(-3.9, ys, radius=0.22)
        head = Text("hidden neurons", font_size=24, color=GREY_B).move_to([-3.9, 2.15, 0])
        down = arrow([-3.45, -0.2, 0], [-0.1, -0.2, 0], width=5, ratio=0.12)
        down_lab = Text("down projection", font_size=24, color=GREY_B).next_to(down, UP, buff=0.15)

        O = np.array([1.8, -1.8, 0])
        hline = Line(O + 1.0 * LEFT, O + 4.6 * RIGHT, stroke_color=GREY_D, stroke_width=2)
        vline = Line(O + 0.7 * DOWN, O + 4.2 * UP, stroke_color=GREY_D, stroke_width=2)
        tip = O + np.array([2.6, 0.9, 0])
        x_arr = vec_arrow(O, tip, VEC_COLOR)
        x_lab = Text("“Tower” vector", font_size=22, color=VEC_COLOR).move_to([3.4, O[1] - 0.4, 0])
        self.play(FadeIn(head), LaggedStart(*[FadeIn(c, scale=0.6) for c in circles], lag_ratio=0.08),
                  GrowArrow(down), FadeIn(down_lab), Create(hline), Create(vline), GrowArrow(x_arr), FadeIn(x_lab),
                  run_time=0.9)

        active = [circles[1], circles[4]]
        self.at("active")
        self.play(*[c.animate.set_fill(MLP_COLOR, 0.9) for c in active], run_time=0.4)
        pushes = VGroup(vec_arrow(tip, tip + np.array([0.9, 0.5, 0]), GREY_A, 4),
                        vec_arrow(tip, tip + np.array([-0.5, 0.9, 0]), GREY_A, 4))
        push_lab = Text("pushes", font_size=22, color=GREY_A).next_to(pushes, UP, buff=0.1)
        self.at("push")
        self.play(ShowPassingFlash(down.copy().set_color(YELLOW), time_width=0.6),
                  *[GrowArrow(p) for p in pushes], FadeIn(push_lab), run_time=0.7)

        simp = caption("simplified picture", corner=UR, font_size=24)
        self.at("simplified")
        self.play(FadeIn(simp), FadeOut(pushes), FadeOut(push_lab),
                  *[c.animate.set_fill(GREEN_E, 0.4) for c in active], run_time=0.6)

        eiffel_n = circles[2]
        eiffel = Text("Eiffel Tower", font_size=26, color=YELLOW).next_to(eiffel_n, LEFT, buff=0.25)
        self.at("eiffel")
        self.play(eiffel_n.animate.set_fill(YELLOW, 0.9).set_stroke(YELLOW), FadeIn(eiffel, shift=0.1 * RIGHT),
                  run_time=0.6)
        self.at("fires")
        self.play(Flash(eiffel_n, color=YELLOW, line_length=0.2, flash_radius=0.35),
                  ShowPassingFlash(down.copy().set_color(YELLOW), time_width=0.6), run_time=0.6)

        paris_end = tip + np.array([-0.4, 2.3, 0])
        push = vec_arrow(tip, paris_end, MLP_COLOR)
        push_lab2 = Text("+ Paris", font_size=28, color=MLP_COLOR).next_to(push.get_center(), RIGHT, buff=0.25)
        self.at("add")
        self.play(GrowArrow(push), FadeIn(push_lab2), run_time=0.7)
        result = vec_arrow(O, paris_end, YELLOW)
        res_lab = Text("“Tower” + Paris", font_size=24, color=YELLOW).next_to(paris_end, UP, buff=0.15)
        self.at("paris")
        self.play(GrowArrow(result), FadeIn(res_lab), run_time=0.7)

        found = Text("facts are largely recalled in MLP layers", font_size=28, t2c={"MLP layers": MLP_COLOR})
        found.move_to([0, -3.15, 0])
        self.at("researchers")
        self.play(FadeIn(found, shift=0.15 * UP), run_time=0.7)
        self.end_section()

    # 8. Where the weights are -------------------------------------------------------------------
    def s8_weights(self):
        self.section(8)
        self.play(*self.clear_anims(), run_time=0.5)

        pie_c = np.array([-2.3, -2.3, 0])
        pie = VGroup(Sector(radius=0.75, angle=TAU * 2 / 3, start_angle=PI / 2, color=MLP_COLOR, fill_opacity=0.85),
                     Sector(radius=0.75, angle=TAU / 3, start_angle=PI / 2 + TAU * 2 / 3, color=VEC_COLOR,
                            fill_opacity=0.85))
        pie_lab = Text("MLP ≈ 2/3 of each layer's weights", font_size=26)
        pie_lab[:3].set_color(MLP_COLOR)
        pie_grp = VGroup(pie, pie_lab)
        pie_lab.next_to(pie, RIGHT, buff=0.4)
        final = pie_grp.copy().move_to([pie_c[0] + (pie_lab.width + 0.4) / 2, pie_c[1], 0])
        pie_grp.scale(1.4).move_to(ORIGIN)
        self.at("most")
        self.play(FadeIn(pie, scale=0.8), FadeIn(pie_lab, shift=0.1 * LEFT), run_time=0.8)

        header = Text("GPT-2 small: weights per layer", font_size=32).move_to([0, 3.1, 0])
        self.at("gpt")
        self.play(FadeIn(header, shift=0.15 * DOWN), Transform(pie_grp, final), run_time=0.7)

        x0, unit = -4.0, 3.5 / 2.359296  # bar length per million weights
        y_mlp, y_att = 1.6, 0.1
        lab_mlp = Text("MLP", font_size=30, color=MLP_COLOR)
        lab_att = Text("attention", font_size=30, color=VEC_COLOR)
        lab_mlp.move_to([x0 - 0.3, y_mlp, 0], aligned_edge=RIGHT)
        lab_att.move_to([x0 - 0.3, y_att, 0], aligned_edge=RIGHT)
        self.at("mlp")
        self.play(FadeIn(lab_mlp), FadeIn(lab_att), run_time=0.4)

        def bar(millions, y, color):
            return Rectangle(width=millions * unit, height=0.55, stroke_width=0, fill_color=color,
                             fill_opacity=0.85).move_to([x0, y, 0], aligned_edge=LEFT)

        bar_mlp = bar(4.718592, y_mlp, MLP_COLOR)
        val_mlp = Text("≈ 4.7 M", font_size=30).next_to(bar_mlp, RIGHT, buff=0.25)
        calc_mlp = Text("2 × 768 × 3,072 = 4,718,592", font_size=22, color=GREY_B)
        calc_mlp.next_to(bar_mlp, DOWN, buff=0.15, aligned_edge=LEFT)
        self.at("4")
        self.play(GrowFromEdge(bar_mlp, LEFT), FadeIn(val_mlp), FadeIn(calc_mlp), run_time=0.8)

        bar_att = bar(2.359296, y_att, VEC_COLOR)
        val_att = Text("≈ 2.4 M", font_size=30).next_to(bar_att, RIGHT, buff=0.25)
        calc_att = Text("4 × 768 × 768 = 2,359,296", font_size=22, color=GREY_B)
        calc_att.next_to(bar_att, DOWN, buff=0.15, aligned_edge=LEFT)
        self.at("million")
        self.play(GrowFromEdge(bar_att, LEFT), FadeIn(val_att), FadeIn(calc_att), run_time=0.7)

        ghosts = VGroup(*[Rectangle(width=bar_att.width, height=0.55, stroke_color=YELLOW, stroke_width=3,
                                    fill_opacity=0) for _ in range(2)]).arrange(RIGHT, buff=0)
        ghosts.move_to([x0, y_mlp, 0], aligned_edge=LEFT)
        two = Text("2×", font_size=40, color=YELLOW).next_to(val_mlp, RIGHT, buff=0.35)
        self.at("twice")
        self.play(TransformFromCopy(VGroup(bar_att.copy().set_fill(opacity=0).set_stroke(YELLOW, 3),
                                           bar_att.copy().set_fill(opacity=0).set_stroke(YELLOW, 3)), ghosts),
                  FadeIn(two, scale=1.3), run_time=0.8)
        self.end_section()

    # 9. Code ------------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, font_size=26)
        code.move_to(ORIGIN)
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[0])
        self.play(*self.clear_anims(), FadeIn(code), run_time=0.7)
        self.at("jell")
        self.play(Create(hl), run_time=0.4)
        self.at("expand")
        self.play(highlight(hl, code, 5), run_time=0.4)
        self.at("project")
        self.play(highlight(hl, code, 6), run_time=0.4)
        self.end_section()

    # 10. Outro ----------------------------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        self.play(*self.clear_anims(), run_time=0.5)
        divider = Line([0, 3.3, 0], [0, -2.6, 0], color=GREY_D, stroke_width=2)
        self.at("halves")
        self.play(Create(divider), run_time=0.5)

        left_title = Text("attention: tokens talk", font_size=30).move_to([-3.5, 2.9, 0])
        lrow = token_row(["The", "cat", "sat", "on"], buff=0.3).move_to([-3.5, -0.9, 0])
        self.at("tension")
        self.play(FadeIn(left_title, shift=0.15 * DOWN), LaggedStart(*[FadeIn(t) for t in lrow], lag_ratio=0.1),
                  run_time=0.6)
        pairs = [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3)]
        arcs = VGroup(*[arc_over(lrow[i].get_top() + 0.08 * UP, lrow[j].get_top() + 0.08 * UP, color=VEC_COLOR,
                                 width=3) for i, j in pairs])
        self.at("talk")
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.2), run_time=0.8)

        right_title = Text("MLP: each token thinks", font_size=30).move_to([3.5, 2.9, 0])
        box = mlp_box(2.4, 0.9, 34).move_to([3.5, -0.1, 0])
        box.set_z_index(1)
        tin = token("sat").move_to([3.5, 1.5, 0])
        a_in = arrow(tin.get_bottom() + 0.08 * DOWN, box.get_top() + 0.08 * UP, ratio=0.35)
        self.at("mlp")
        self.play(FadeIn(right_title, shift=0.15 * DOWN), FadeIn(box, shift=0.15 * DOWN), FadeIn(tin),
                  GrowArrow(a_in), run_time=0.6)
        tout = token("sat", color=MLP_COLOR).move_to([3.5, -1.7, 0])
        a_out = arrow(box.get_bottom() + 0.08 * DOWN, tout.get_top() + 0.08 * UP, ratio=0.35)
        self.at("token")
        self.play(box[0].animate(rate_func=there_and_back).set_fill(MLP_COLOR, 0.6), GrowArrow(a_out),
                  TransformFromCopy(tin, tout), run_time=0.8)

        left = VGroup(lrow, arcs)
        right = VGroup(tin, a_in, box, a_out, tout)
        l_lab = Text("attention", font_size=26, color=VEC_COLOR)
        r_lab = Text("MLP", font_size=26, color=MLP_COLOR)
        self.at("together")
        self.play(FadeOut(divider), FadeOut(left_title), FadeOut(right_title),
                  left.animate.scale(0.85).move_to([-2.1, -0.4, 0]),
                  right.animate.scale(0.85).move_to([2.1, -0.3, 0]), run_time=0.8)
        l_lab.next_to(left, UP, buff=0.3).set_y(1.75)
        r_lab.next_to(right, UP, buff=0.3).set_y(1.75)
        outline = RoundedRectangle(corner_radius=0.25, width=8.4, height=4.6, stroke_color=WHITE,
                                   stroke_width=3).move_to([0, -0.05, 0])
        block = Text("transformer block", font_size=32).next_to(outline, UP, buff=0.25)
        self.at("block")
        self.play(Create(outline), FadeIn(block, shift=0.15 * DOWN), FadeIn(l_lab), FadeIn(r_lab), run_time=0.6)

        card = next_up_card(NEXT)
        self.at("stack")
        self.play(*self.clear_anims(), FadeIn(card), run_time=0.6)
        self.end_section()
        late = self.renderer.time - (self.sec_start + self.sec["dur"] + self.SECTION_GAP)
        print(f"[v08] last section ends at {self.renderer.time:.2f}s (late by {late:.2f}s)")
        finish(self)
