"""Video 7 — Multi-Head Attention.

Render from the repo root:  ./render.sh how-llms-work v07
"""
import numpy as np
from manim import *

from common import code_panel, finish, highlight, next_up_card, token, token_row
from intro import play_token_intro
from v07_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

HEAD_COLORS = [BLUE_C, TEAL_C, GREEN_C, GOLD_C, RED_C, PURPLE_B,
               YELLOW_C, PINK, "#6F7BF7", ORANGE, MAROON_C, LIGHT_BROWN]
SENT = ["The", "cat", "slowly", "sat", "on", "the", "mat"]
D, N_HEADS, HD = 768, 12, 64
MONO = "DejaVu Sans Mono"

CODE = """def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads):
    n, d = X.shape
    hd = d // n_heads  # 768 // 12 = 64
    def split(M):      # (n, d) -> (heads, n, hd)
        return M.reshape(n, n_heads, hd).transpose(1, 0, 2)
    Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(hd)    # every head at once
    scores += np.triu(np.full((n, n), -np.inf), k=1)   # causal mask
    w = np.exp(scores - scores.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)                      # softmax
    out = (w @ V).transpose(1, 0, 2).reshape(n, d)     # merge the heads
    return out @ Wo                                    # output projection"""


# ---------------------------------------------------------------------------- helpers
def caption(text, font_size=22):
    return Text(text, font_size=font_size, color=GREY_B).to_corner(DR, buff=0.3)


def rgb01(color):
    return np.array(ManimColor(color).to_rgb())


def strip_image(rgb, width, height):
    """A row of len(rgb) cells (rgb: (n, 3) floats in 0..1) drawn as a crisp stretched image."""
    arr = np.zeros((1, len(rgb), 4), dtype=np.uint8)
    arr[0, :, :3] = np.clip(np.asarray(rgb) * 255, 0, 255).astype(np.uint8)
    arr[0, :, 3] = 255
    img = ImageMobject(arr)
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img.stretch_to_fit_width(width).stretch_to_fit_height(height)
    return img


def tinted(bright, color):
    """Cell colours: brightness values (n,) times a head colour."""
    return bright[:, None] * rgb01(color)[None, :]


def head_segments(bright, colors, seg_w, height, gap=0.0):
    """12 strips of 64 cells, one per head, laid out left to right."""
    segs = Group(*[strip_image(tinted(bright[k * HD:(k + 1) * HD], colors[k]), seg_w, height)
                   for k in range(len(colors))])
    return segs.arrange(RIGHT, buff=gap)


def value_rgb(values):
    """Signed values -> blue (positive) / red (negative) cells, as in the embeddings video."""
    out = np.zeros((len(values), 3))
    grey, pos, neg = rgb01(GREY_E), rgb01(BLUE_C), rgb01(RED_C)
    for i, v in enumerate(np.clip(values, -1, 1)):
        out[i] = grey + abs(v) * ((pos if v >= 0 else neg) - grey)
    return out


def bubble(text, color, font_size=26):
    label = Text(text, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.25, width=label.width + 0.55, height=label.height + 0.5,
                           stroke_color=color, stroke_width=3, fill_color=BLACK, fill_opacity=1)
    return VGroup(box, label.move_to(box))


def attn_grid(weights, color, cell=0.4):
    """Lower-triangular attention heatmap: row = token asking, column = token looked at."""
    n = len(weights)
    grid = VGroup()
    for i in range(n):
        for j in range(n):
            sq = Square(cell, stroke_color=GREY_D, stroke_width=1)
            sq.move_to([(j - (n - 1) / 2) * cell, ((n - 1) / 2 - i) * cell, 0])
            if j > i:
                sq.set_fill(BLACK, 0)
            else:
                w = float(weights[i][j])
                sq.set_fill(interpolate_color(ManimColor(GREY_E), ManimColor(color), min(1.0, w)), 1)
            grid.add(sq)
    return grid


def pattern_previous(n=6):
    W = np.zeros((n, n))
    W[0, 0] = 1
    for i in range(1, n):
        W[i, :i + 1] = 0.2 / i
        W[i, i - 1] = 0.8
    return W


def pattern_verb_object(n=6):
    W = np.zeros((n, n))
    for i in range(n):
        W[i, :i + 1] = 1 / (i + 1)
    W[4, :5] = 0.05
    W[4, 1] = 0.8           # the object looks back at its verb
    return W


def pattern_induction(n=6):
    W = np.zeros((n, n))
    W[0, 0] = 1
    W[1, :2] = [0.7, 0.3]
    W[2, :3] = [0.6, 0.2, 0.2]
    for i in range(3, n):    # "A B C A B C": each repeat looks at what followed its first copy
        W[i, :i + 1] = 0.2 / i
        W[i, i - 2] = 0.8
    return W


def pattern_noisy(n=6, seed=11):
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, :i + 1] = rng.dirichlet(np.full(i + 1, 0.6))
    return W


def arc_arrow(start, end, angle, color, width=4, opacity=1.0):
    return CurvedArrow(start, end, angle=angle, color=color, stroke_width=width, tip_length=0.16,
                       stroke_opacity=opacity)


class MultiHeadVideo(VoicedScene):
    VIDEO = "v07"

    def section(self, i):
        super().section(i)
        print(f"[v07] section {i} starts at {self.renderer.time:.2f}s")

    def clear_anims(self):
        return [FadeOut(m) for m in list(self.mobjects)]

    def construct(self):
        play_token_intro(self, TITLE, 7, TAGLINE)

        # 1. One head is not enough ----------------------------------------------------------------
        self.section(1)
        row = token_row(SENT, buff=0.15).move_to([0, -1.4, 0])
        sat = row[3]
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in row], lag_ratio=0.1), run_time=1.0)
        tag = Text("one head = one kind of question", font_size=28, color=GREY_B).move_to([0, 3.1, 0])
        self.at("question")
        self.play(FadeIn(tag, shift=0.2 * DOWN), run_time=0.6)
        self.at("sat")
        self.play(sat[0].animate.set_stroke(YELLOW).set_fill(YELLOW, 0.35), run_time=0.5)
        tag2 = Text("one word, several questions", font_size=28, color=GREY_B).move_to(tag)
        self.at("several")
        self.play(FadeTransform(tag, tag2), run_time=0.6)
        specs = [("Who is sitting?", [-4.1, 0.5, 0]), ("Where?", [sat.get_x(), 1.6, 0]),
                 ("What came just before?", [4.0, 0.5, 0])]
        for k, (cue, (text, pos)) in enumerate(zip(["sitting", "where", "before"], specs)):
            b = bubble(text, HEAD_COLORS[k]).move_to(pos)
            tail = Line(b.get_bottom(), sat.get_top(), buff=0.08, color=HEAD_COLORS[k], stroke_width=2)
            self.at(cue)
            self.play(FadeIn(b, scale=0.85), Create(tail), run_time=0.5)
        self.end_section()

        # 2. Several heads ----------------------------------------------------------------------------
        self.section(2)
        self.play(*self.clear_anims(), run_time=0.5)
        big = VGroup(RoundedRectangle(corner_radius=0.2, width=7.0, height=2.8, stroke_color=WHITE,
                                      fill_color=GREY_E, fill_opacity=1),
                     Text("attention", font_size=40)).move_to([0, 0.2, 0])
        big[1].move_to(big[0])
        self.at("big")
        self.play(GrowFromCenter(big), run_time=0.6)

        boxes = VGroup(*[RoundedRectangle(corner_radius=0.15, width=2.65, height=2.8, stroke_color=c,
                                          fill_color=c, fill_opacity=0.12) for c in HEAD_COLORS[:4]])
        dots = Text("⋯", font_size=48, color=GREY_B)
        VGroup(*boxes, dots).arrange(RIGHT, buff=0.35).move_to([0, 0.2, 0])
        minis = VGroup(*[Text("attention", font_size=22).move_to(b.get_center() + 0.25 * UP) for b in boxes])
        self.at("several")
        self.play(*[TransformFromCopy(big[0], b) for b in boxes], FadeOut(big), run_time=0.8)
        self.play(FadeIn(minis), run_time=0.3)
        self.at("side")
        self.play(FadeIn(dots, shift=0.2 * RIGHT), run_time=0.4)
        labels = VGroup(*[Text(f"head {k + 1}", font_size=26, color=c).move_to(b.get_top() + 0.45 * DOWN)
                          for k, (b, c) in enumerate(zip(boxes, HEAD_COLORS))])
        self.at("ahead")
        self.play(LaggedStart(*[FadeIn(l, shift=0.15 * DOWN) for l in labels], lag_ratio=0.15), run_time=0.7)
        trios = []
        for b, c in zip(boxes, HEAD_COLORS):
            trio = VGroup()
            for name in ("W_Q", "W_K", "W_V"):
                sq = RoundedRectangle(corner_radius=0.06, width=0.8, height=0.55, stroke_color=c,
                                      stroke_width=2, fill_color=c, fill_opacity=0.25)
                trio.add(VGroup(sq, Text(name, font_size=20).move_to(sq)))
            trio.arrange(RIGHT, buff=0.06).move_to(b.get_bottom() + 0.62 * UP)
            trios.append(trio)
        for idx, cue in enumerate(["own", "key", "value"]):
            self.at(cue)
            self.play(LaggedStart(*[FadeIn(t[idx], shift=0.15 * UP) for t in trios], lag_ratio=0.1),
                      run_time=0.45)
        self.end_section()

        # 3. Splitting the vector ---------------------------------------------------------------------
        self.section(3)
        self.play(*self.clear_anims(), run_time=0.5)
        rng = np.random.default_rng(7)
        bright = rng.uniform(0.3, 1.0, D)
        L, h, gap = 12.0, 0.5, 0.1
        grey = head_segments(bright, [GREY_B] * N_HEADS, L / N_HEADS, h).move_to([0, 0.2, 0])
        outline = Rectangle(width=L, height=h, stroke_color=GREY_B, stroke_width=1.5).move_to(grey)
        vec_label = Text("one token's vector", font_size=28).next_to(outline, UP, buff=0.35)
        self.at("vector")
        self.play(LaggedStart(*[FadeIn(s) for s in grey], lag_ratio=0.08), Create(outline),
                  FadeIn(vec_label), run_time=0.9)
        title = Text("GPT-2 small", font_size=34).move_to([0, 2.9, 0])
        self.at("gpt")
        self.play(FadeIn(title, shift=0.2 * DOWN), run_time=0.5)
        colored = head_segments(bright, HEAD_COLORS, (L - (N_HEADS - 1) * gap) / N_HEADS, h, gap)
        colored.move_to(grey)
        nums = VGroup(*[Text(str(k + 1), font_size=20, color=c).next_to(s, DOWN, buff=0.15)
                        for k, (s, c) in enumerate(zip(colored, HEAD_COLORS))])
        title2 = Text("GPT-2 small: 12 heads", font_size=34).move_to(title)  # no spaces in submobjects
        title2_head = title2[:len("GPT-2small")]
        self.at("12")
        self.play(*[ReplacementTransform(g, c) for g, c in zip(grey, colored)], FadeOut(outline),
                  FadeIn(nums), title.animate.move_to(title2_head), FadeIn(title2[len("GPT-2small"):]),
                  run_time=0.9)
        top_line = Line(colored.get_corner(UL), colored.get_corner(UR))
        brace768 = Brace(top_line, UP, buff=0.12, color=GREY_B)
        lab768 = Text("768 numbers", font_size=28).next_to(brace768, UP, buff=0.15)
        self.at("768")
        self.play(FadeOut(vec_label), run_time=0.25)
        self.play(GrowFromCenter(brace768), FadeIn(lab768, shift=0.15 * DOWN), run_time=0.5)
        brace64 = Brace(Line(colored[0].get_corner(DL), colored[0].get_corner(DR)), DOWN,
                        buff=0.5, color=HEAD_COLORS[0])
        lab64 = Text("64 each", font_size=26, color=HEAD_COLORS[0]).next_to(brace64, DOWN, buff=0.12)
        lab64.align_to(brace64, LEFT)
        eq = Text("768 ÷ 12 = 64", font_size=34).move_to([0, -2.4, 0])
        self.at("64")
        self.play(GrowFromCenter(brace64), FadeIn(lab64), FadeIn(eq, shift=0.2 * UP), run_time=0.7)
        self.end_section()

        # 4. Different patterns -----------------------------------------------------------------------
        self.section(4)
        self.play(*self.clear_anims(), run_time=0.5)
        xs = [-4.65, -1.55, 1.55, 4.65]
        tags = VGroup(*[Text(f"head {k + 1}", font_size=28, color=HEAD_COLORS[k]).move_to([x, 2.55, 0])
                        for k, x in enumerate(xs)])
        self.at("own")
        self.play(LaggedStart(*[FadeIn(t, shift=0.15 * DOWN) for t in tags], lag_ratio=0.15), run_time=0.7)
        empty = [attn_grid(np.zeros((6, 6)), HEAD_COLORS[k]).move_to([x, 0.5, 0]) for k, x in enumerate(xs)]
        self.at("different")
        self.play(LaggedStart(*[FadeIn(g) for g in empty], lag_ratio=0.15), run_time=0.8)
        note = caption("illustrative patterns")
        self.at("researchers")
        self.play(FadeIn(note), run_time=0.5)
        pats = [pattern_previous(), pattern_verb_object(), pattern_induction(), pattern_noisy()]
        names = [["previous token"], ["verb → object"], ["repeated phrase", "(induction)"],
                 ["hard to interpret"]]
        for k, cue in enumerate(["previous", "verb", "repeated", "harder"]):
            filled = attn_grid(pats[k], HEAD_COLORS[k]).move_to(empty[k])
            name = VGroup(*[Text(s, font_size=24) for s in names[k]]).arrange(DOWN, buff=0.1)
            name.next_to(empty[k], DOWN, buff=0.35)
            self.at(cue)
            self.play(Transform(empty[k], filled), FadeIn(name, shift=0.15 * UP),
                      run_time=0.6 if cue == "harder" else 0.8)
        self.end_section()

        # 5. An illustration ------------------------------------------------------------------------
        self.section(5)
        self.play(*self.clear_anims(), run_time=0.4)
        row = token_row(SENT, buff=0.5).move_to([0, -0.5, 0])
        for t in row[4:]:
            t.fade(0.75)
        masked = Text("future: masked", font_size=20, color=GREY_B).next_to(VGroup(*row[4:]), DOWN, buff=0.3)
        note = caption("illustrative patterns")
        self.at("illustration")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in row], lag_ratio=0.08),
                  FadeIn(masked), FadeIn(note), run_time=0.9)
        sat = row[3]
        self.at("sat")
        self.play(sat[0].animate.set_stroke(YELLOW).set_fill(YELLOW, 0.35), run_time=0.4)

        legend_rows = [("head 1 → cat: who is sitting", HEAD_COLORS[0]),
                       ("head 2 → one step back", HEAD_COLORS[1]),
                       ("head 3 → spread widely", HEAD_COLORS[2])]
        legend = VGroup()
        for k, (text, c) in enumerate(legend_rows):
            sw = Line(ORIGIN, 0.5 * RIGHT, color=c, stroke_width=6)
            legend.add(VGroup(sw, Text(text, font_size=24, color=c).next_to(sw, RIGHT, buff=0.2)))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_corner(UL, buff=0.5)

        arc1 = arc_arrow(sat.get_top() + 0.12 * RIGHT, row[1].get_top(), PI * 0.65, HEAD_COLORS[0], 5)
        self.at("cat")
        self.play(Create(arc1), FadeIn(legend[0]), run_time=0.6)
        arc2 = arc_arrow(sat.get_top() + 0.2 * LEFT, row[2].get_top() + 0.2 * RIGHT, PI * 0.85,
                         HEAD_COLORS[1], 5)
        self.at("back")
        self.play(Create(arc2), FadeIn(legend[1]), run_time=0.6)
        weights = [0.3, 0.25, 0.25]
        arcs3 = VGroup(*[arc_arrow(sat.get_bottom() + (0.1 - 0.1 * j) * RIGHT, row[j].get_bottom(),
                                   -PI * 0.55, HEAD_COLORS[2], 1.5 + 8 * w)
                         for j, w in enumerate(weights)])
        self.at("widely")
        self.play(LaggedStart(*[Create(a) for a in arcs3], lag_ratio=0.2), FadeIn(legend[2]), run_time=0.9)
        self.end_section()

        # 6. Concatenate ----------------------------------------------------------------------------
        self.section(6)
        self.play(*self.clear_anims(), run_time=0.5)
        rng = np.random.default_rng(21)
        out_bright = rng.uniform(0.3, 1.0, D)
        outs = head_segments(out_bright, HEAD_COLORS, 1.0, 0.45)
        head_labels = VGroup()
        for k, s in enumerate(outs):
            r, c = divmod(k, 6)
            s.move_to([-4.75 + 1.9 * c, 1.5 - 1.5 * r, 0])
            head_labels.add(Text(f"head {k + 1}", font_size=20, color=HEAD_COLORS[k]).next_to(s, UP, buff=0.15))
        title6 = Text("each head: 64 numbers", font_size=32).move_to([0, 3.1, 0])
        self.at("64")
        self.play(LaggedStart(*[FadeIn(Group(s, l), shift=0.15 * DOWN) for s, l in zip(outs, head_labels)],
                              lag_ratio=0.08), FadeIn(title6), run_time=1.0)
        title6b = Text("concatenate", font_size=32).move_to(title6)
        self.at("glue")
        self.play(*[s.animate.move_to([-5.5 + k, -1.0, 0]) for k, s in enumerate(outs)],
                  FadeOut(head_labels), FadeTransform(title6, title6b), run_time=1.2)
        brace = Brace(Line([-6, -1.225, 0], [6, -1.225, 0]), DOWN, buff=0.12, color=GREY_B)
        lab = Text("12 × 64 = 768 numbers", font_size=28).next_to(brace, DOWN, buff=0.15)
        self.at("768")
        self.play(GrowFromCenter(brace), FadeIn(lab, shift=0.15 * UP), run_time=0.6)
        self.end_section()

        # 7. Output projection -----------------------------------------------------------------------
        self.section(7)
        self.play(FadeOut(VGroup(title6b, brace, lab)),
                  outs.animate.scale_to_fit_width(8).move_to([0, 2.5, 0]), run_time=0.8)
        heads_lab = Text("all heads", font_size=22, color=GREY_B).next_to(outs, RIGHT, buff=0.3)
        wo = VGroup(RoundedRectangle(corner_radius=0.12, width=2.0, height=0.85, stroke_color=WHITE,
                                     fill_color=GREY_E, fill_opacity=1),
                    Text("W_O", font_size=30)).move_to([0, 1.05, 0])
        wo[1].move_to(wo[0])
        a1 = Arrow(outs.get_bottom(), wo.get_top(), buff=0.08, color=GREY_B, stroke_width=4,
                   max_tip_length_to_length_ratio=0.3)
        self.at("matrix")
        self.play(FadeIn(heads_lab), GrowArrow(a1), FadeIn(wo, shift=0.15 * DOWN), run_time=0.5)
        wo_lab = Text("output projection", font_size=26).next_to(wo, RIGHT, buff=0.35)
        self.at("projection")
        self.play(FadeIn(wo_lab, shift=0.15 * LEFT), run_time=0.4)
        mix_rng = np.random.default_rng(5)
        cols = np.array([rgb01(c) for c in HEAD_COLORS])
        mix_rgb = mix_rng.dirichlet(np.full(N_HEADS, 0.35), D) @ cols
        mix_rgb *= mix_rng.uniform(0.5, 1.0, D)[:, None]
        mixed = Group(*[strip_image(mix_rgb[k * HD:(k + 1) * HD], 8 / N_HEADS, 0.45) for k in range(N_HEADS)])
        mixed.arrange(RIGHT, buff=0).move_to([0, -0.45, 0])
        a2 = Arrow(wo.get_bottom(), mixed.get_top(), buff=0.08, color=GREY_B, stroke_width=4,
                   max_tip_length_to_length_ratio=0.3)
        mixed_lab = Text("mixed", font_size=22, color=GREY_B).next_to(mixed, RIGHT, buff=0.3)
        self.at("mixes")
        self.play(GrowArrow(a2), *[TransformFromCopy(s, m) for s, m in zip(outs, mixed)],
                  FadeIn(mixed_lab), run_time=1.1)
        tok_vals = np.clip(np.random.default_rng(3).normal(0, 0.45, D), -1, 1)
        delta = np.clip(np.random.default_rng(4).normal(0, 0.5, D), -1, 1)
        tok = strip_image(value_rgb(tok_vals), 8, 0.45).move_to([0, -2.45, 0])
        tok_new = strip_image(value_rgb(np.clip(tok_vals + 0.5 * delta, -1, 1)), 8, 0.45).move_to(tok)
        tok_lab = Text("token's vector", font_size=22, color=GREY_B).next_to(tok, RIGHT, buff=0.3)
        self.at("result")
        self.play(FadeIn(tok, shift=0.15 * UP), FadeIn(tok_lab), run_time=0.4)
        plus = Text("+", font_size=48).next_to(tok, LEFT, buff=0.35)
        resid = Text("residual: add, don't replace", font_size=22, color=GREY_B).next_to(tok, DOWN, buff=0.3)
        self.at("added")
        self.play(FadeIn(plus, scale=1.3), run_time=0.3)
        self.play(FadeOut(mixed.copy(), shift=tok.get_center() - mixed.get_center()), Transform(tok, tok_new),
                  FadeIn(resid), run_time=0.8)
        self.at("tokens")
        self.play(Circumscribe(tok, color=YELLOW, buff=0.08), run_time=0.8)
        self.end_section()

        # 8. Same cost --------------------------------------------------------------------------------
        self.section(8)
        self.play(*self.clear_anims(), run_time=0.5)
        slab_w, slab_h, slab_gap = 0.25, 3.0, 0.07
        slabs = VGroup(*[Rectangle(width=slab_w, height=slab_h, stroke_color=c, stroke_width=2,
                                   fill_color=c, fill_opacity=0.55) for c in HEAD_COLORS])
        slabs.arrange(RIGHT, buff=slab_gap).move_to([-3.6, 0.5, 0])
        square = Rectangle(width=3.0, height=3.0, stroke_color=WHITE, stroke_width=2,
                           fill_color=GREY_B, fill_opacity=0.35).move_to([3.6, 0.5, 0])
        left_lab = Text("12 heads × 64", font_size=30).next_to(slabs, DOWN, buff=0.35)
        right_lab = Text("1 head × 768", font_size=30).next_to(square, DOWN, buff=0.35)
        self.at("12")
        self.play(LaggedStart(*[GrowFromEdge(s, DOWN) for s in slabs], lag_ratio=0.06), run_time=0.8)
        self.at("64")
        self.play(FadeIn(left_lab, shift=0.15 * UP), run_time=0.4)
        approx = VGroup(Text("≈", font_size=72), Text("same cost", font_size=28)).arrange(DOWN, buff=0.2)
        approx.move_to([0, 0.5, 0])
        self.at("same")
        self.play(FadeIn(approx, scale=0.8), run_time=0.4)
        self.at("one")
        self.play(GrowFromEdge(square, DOWN), run_time=0.6)
        weights_note = Text("Q, K, V weights: 3 × 768 × 768 either way", font_size=24,
                            color=GREY_B).move_to([0, -2.45, 0])
        self.at("768")
        self.play(FadeIn(right_lab, shift=0.15 * UP), FadeIn(weights_note), run_time=0.5)
        self.at("view")
        self.play(LaggedStart(*[s.animate(rate_func=there_and_back).shift(0.2 * UP) for s in slabs],
                              lag_ratio=0.05), run_time=0.7)
        price = Text("12 views, 1 price", font_size=30, color=YELLOW).move_to([0, -3.25, 0])
        self.at("price")
        self.play(FadeIn(price, shift=0.15 * UP), run_time=0.5)
        self.end_section()

        # 9. Code ---------------------------------------------------------------------------------------
        self.section(9)
        code, hl = code_panel(CODE, font_size=20)
        code.move_to(ORIGIN)
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[5])
        self.play(*self.clear_anims(), FadeIn(code), run_time=0.7)
        self.at("split")
        self.play(Create(hl), run_time=0.4)
        for cue, line in [("attention", 6), ("merge", 10), ("projection", 11)]:
            self.at(cue)
            self.play(highlight(hl, code, line), run_time=0.4)
        self.end_section()

        # 10. Outro -------------------------------------------------------------------------------------
        self.section(10)
        self.play(*self.clear_anims(), run_time=0.5)
        divider = Line([0, 3.3, 0], [0, -2.4, 0], color=GREY_D, stroke_width=2)
        left_title = Text("attention: tokens talk", font_size=30).move_to([-3.4, 2.9, 0])
        lrow = token_row(["The", "cat", "slowly", "sat"], buff=0.3).move_to([-3.4, -0.4, 0])
        self.play(FadeIn(left_title, shift=0.15 * DOWN), Create(divider),
                  LaggedStart(*[FadeIn(t) for t in lrow], lag_ratio=0.1), run_time=0.6)
        links = [(3, 1, 0), (3, 2, 1), (2, 1, 2), (1, 0, 3), (3, 0, 4)]
        arcs = VGroup()
        for a, b, k in links:
            ang = PI * (0.5 + 0.08 * (a - b))
            arcs.add(arc_arrow(lrow[a].get_top() + 0.1 * (k % 2 - 0.5) * RIGHT, lrow[b].get_top(), ang,
                               HEAD_COLORS[k], 3))
        self.at("share")
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.2), run_time=1.2)
        self.at("context")
        self.play(lrow[3][0].animate.set_stroke(YELLOW).set_fill(YELLOW, 0.35), run_time=0.5)
        right_title = Text("MLP: each token on its own", font_size=30).move_to([3.4, 2.9, 0])
        solo = lrow[3].copy()
        mlp = VGroup(RoundedRectangle(corner_radius=0.15, width=2.6, height=1.1, stroke_color=WHITE,
                                      fill_color=GREY_E, fill_opacity=1),
                     Text("MLP", font_size=36)).move_to([3.4, 0.0, 0])
        mlp[1].move_to(mlp[0])
        mlp.set_z_index(1)  # the token passes behind the box on its way through
        m_arrow = Arrow([3.4, 1.25, 0], mlp.get_top(), buff=0.08, color=GREY_B, stroke_width=4,
                        max_tip_length_to_length_ratio=0.3)
        self.at("own")
        self.play(solo.animate.move_to([3.4, 1.6, 0]), FadeIn(right_title, shift=0.15 * DOWN),
                  FadeIn(mlp, shift=0.15 * DOWN), GrowArrow(m_arrow), run_time=1.0)
        out_tok = token("sat", color=PURPLE_B).move_to([3.4, -1.6, 0])
        o_arrow = Arrow(mlp.get_bottom(), out_tok.get_top(), buff=0.08, color=GREY_B, stroke_width=4,
                        max_tip_length_to_length_ratio=0.3)
        self.at("half")
        self.play(GrowArrow(o_arrow), TransformFromCopy(solo, out_tok), run_time=0.7)
        layer = Text("every layer: attention, then MLP", font_size=28, color=GREY_B).move_to([0, -3.1, 0])
        self.at("layer")
        self.play(FadeIn(layer, shift=0.15 * UP), run_time=0.5)
        card = next_up_card(NEXT)
        self.at("mlp")
        self.play(*self.clear_anims(), FadeIn(card), run_time=0.8)
        self.end_section()
        late = self.renderer.time - (self.sec_start + self.sec["dur"] + self.SECTION_GAP)
        print(f"[v07] last section ends at {self.renderer.time:.2f}s (late by {late:.2f}s)")
        finish(self)
