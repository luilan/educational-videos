"""How LLMs Work: Deep Dive, episode 8 — The Attention Matrix, Entry by Entry.

Render from the repo root:  ./render.sh deep-dive d08
Every number on screen comes from code/d08_attention_matrix/attention_matrix.py (GPT-2 small, real weights, layer 4,
head 3, counting from zero, on "The cat sat on the mat because it was tired"; checked against transformers 4.57.1).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d08_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 8"
WORDS = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "tired"]
W = [[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.91, 0.09, 0, 0, 0, 0, 0, 0, 0, 0], [0.03, 0.96, 0.01, 0, 0, 0, 0, 0, 0, 0],
     [0.05, 0.89, 0.05, 0.01, 0, 0, 0, 0, 0, 0], [0.19, 0.57, 0.19, 0.04, 0.01, 0, 0, 0, 0, 0],
     [0.4, 0.3, 0.12, 0.07, 0.09, 0.02, 0, 0, 0, 0], [0.14, 0.33, 0.21, 0.18, 0.05, 0.05, 0.03, 0, 0, 0],
     [0.06, 0.84, 0.02, 0.01, 0.01, 0.04, 0.01, 0.02, 0, 0], [0.12, 0.48, 0.02, 0.02, 0.0, 0.11, 0.04, 0.15, 0.05, 0],
     [0.06, 0.47, 0.0, 0.0, 0.0, 0.02, 0.01, 0.35, 0.06, 0.02]]
IT_SCORES = [-2.14, 0.57, -3.10, -3.57, -4.33, -2.58, -3.63, -3.30]
IT_WEIGHTS = [0.056, 0.837, 0.021, 0.013, 0.006, 0.036, 0.013, 0.018]
PRODUCTS = [(-0.107, "q₀k₀"), (1.352, "q₁k₁"), (-1.339, "q₂k₂"), (0.893, "q₃k₃")]
CODE = """q, k, v = c_attn(ln_1(x)).split(768, dim=-1)              # (T, 768) each
q, k, v = (t.view(T, 12, 64).transpose(0, 1) for t in (q, k, v))
scores = q @ k.transpose(-2, -1) / 64 ** 0.5               # (12, T, T)
scores = scores.masked_fill(mask, float("-inf"))           # the future
weights = scores.softmax(dim=-1)
out = c_proj((weights @ v).transpose(0, 1).reshape(T, 768))"""


def heat(values, size, font_size=None, masked=True, colour=GREEN_C):
    n = len(values)
    cells, labels = VGroup(), VGroup()
    for i in range(n):
        for j in range(n):
            v = values[i][j]
            fill = BLACK if (masked and j > i) else interpolate_color(GREY_E, colour, min(1.0, v))
            c = Square(size, stroke_color=GREY_B, stroke_width=1, fill_color=fill, fill_opacity=0.95)
            c.move_to([(j - (n - 1) / 2) * size, ((n - 1) / 2 - i) * size, 0])
            cells.add(c)
            if font_size and not (masked and j > i):
                labels.add(Text(f"{v:.2f}", font=MONO, font_size=font_size).move_to(c))
    return cells, labels


class AttentionMatrixVideo(VoicedScene):
    VIDEO = "d08"

    def construct(self):
        play_token_intro(self, TITLE, 8, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.qkv()           # 2
        self.entry()         # 3
        self.row()           # 4
        self.matrix()        # 5
        self.scaling()       # 6
        self.values()        # 7
        self.other_heads()   # 8
        self.scale()         # 9
        self.code()          # 10
        self.outro()         # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        cells, _ = heat(W, 0.45)
        cells.move_to([-2.5, -0.2, 0])
        self.at("grid")
        self.play(FadeIn(cells, lag_ratio=0.01), run_time=0.8)
        q = Text("where does each\nnumber come from?", font_size=30, line_spacing=0.9).move_to([3.0, 0.6, 0])
        self.at("number")
        self.play(FadeIn(q), Indicate(cells[71], color=YELLOW), run_time=0.6)
        plan = Text("GPT-2, real weights,\nby hand, checked\nagainst the library", font_size=24, color=GREY_A,
                    line_spacing=0.9).move_to([3.0, -1.2, 0])
        self.at("hand")
        self.play(FadeIn(plan), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. q, k, v
    def qkv(self):
        self.section(2)
        self.clear_stage()
        toks = VGroup(*[token(w, font_size=24) for w in WORDS]).arrange(RIGHT, buff=0.12).move_to([0, 2.6, 0])
        self.at("sentence")
        self.play(LaggedStart(*[FadeIn(t) for t in toks], lag_ratio=0.06), run_time=0.8)
        where = Text("layer 4, head 3 (counting from 0)", font_size=24, color=YELLOW).move_to([0, 1.7, 0])
        self.at("layer")
        self.play(FadeIn(where), run_time=0.4)
        x = Rectangle(width=0.5, height=2.4, color=MODEL_COLOR, fill_opacity=0.3).move_to([-4.5, -0.6, 0])
        xl = Text("768", font=MONO, font_size=20).next_to(x, DOWN, buff=0.1)
        xn = Text("token vector\n(normalized)", font_size=18, color=GREY_B, line_spacing=0.8).next_to(x, UP, buff=0.1)
        self.at("768")
        self.play(FadeIn(x), FadeIn(xl), FadeIn(xn), run_time=0.5)
        arr = Arrow(x.get_right(), x.get_right() + 1.9 * RIGHT, buff=0.15)
        ml = Text("× one matrix", font_size=20).next_to(arr, DOWN, buff=0.1)
        self.at("multiplied")
        self.play(GrowArrow(arr), FadeIn(ml), run_time=0.5)
        bars = VGroup()
        for k, (name, col) in enumerate((("query", YELLOW), ("key", BLUE_C), ("value", GREEN_C))):
            chunks = VGroup(*[Rectangle(width=0.32, height=0.42, stroke_width=1, stroke_color=BLACK, fill_color=col,
                                        fill_opacity=0.7) for _ in range(12)]).arrange(RIGHT, buff=0)
            lab = Text(name, font_size=22, color=col).next_to(chunks, LEFT, buff=0.25)
            bars.add(VGroup(lab, chunks))
        bars.arrange(DOWN, buff=0.35, aligned_edge=RIGHT).move_to([1.6, -0.6, 0])
        self.at("query")
        self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.2), run_time=0.8)
        heads = Text("12 heads × 64 numbers", font_size=24, color=GREY_A).next_to(bars, DOWN, buff=0.35)
        self.at("12")
        self.play(FadeIn(heads), *[b[1][3].animate.set_stroke(WHITE, 3) for b in bars], run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 3. one entry
    def entry(self):
        self.section(3)
        self.clear_stage()
        head = Text("one entry:  query of “it” · key of “cat”", font_size=30).to_edge(UP, buff=0.6)
        self.at("entry")
        self.play(FadeIn(head), run_time=0.4)
        rows = VGroup()
        for v, name in PRODUCTS:
            rows.add(VGroup(Text(name, font=MONO, font_size=24, color=GREY_B),
                            Text(f"{v:+.3f}", font=MONO, font_size=24)).arrange(RIGHT, buff=0.5))
        rows.add(Text("⋮  (64 products)", font=MONO, font_size=24, color=GREY_B))
        rows.arrange(DOWN, buff=0.22, aligned_edge=LEFT).move_to([-2.5, -0.3, 0])
        self.at("multiply")
        self.play(LaggedStart(*[FadeIn(r, shift=0.1 * RIGHT) for r in rows], lag_ratio=0.15), run_time=1.0)
        s = Text("sum = 4.55", font=MONO, font_size=32, color=YELLOW).move_to([2.8, 0.6, 0])
        brace = Brace(rows, RIGHT)
        self.at("add")
        self.play(GrowFromCenter(brace), FadeIn(s), run_time=0.5)
        d = Text("÷ √64 = ÷ 8  →  0.57", font=MONO, font_size=32, color=YELLOW).next_to(s, DOWN, buff=0.5)
        self.at("divide")
        self.play(FadeIn(d), run_time=0.5)
        sc = Text("one score", font_size=26, color=GREY_A).next_to(d, DOWN, buff=0.4)
        self.at("score")
        self.play(FadeIn(sc), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4. the row
    def row(self):
        self.section(4)
        self.clear_stage()
        xs = [(-5.85 + 1.3 * j) for j in range(10)]
        labels = VGroup(*[Text(w, font_size=22, color=GREY_B).move_to([xs[j], 2.2, 0]) for j, w in enumerate(WORDS)])
        q = Text("query: “it”", font_size=24, color=YELLOW).move_to([-4.9, 3.1, 0])
        self.play(FadeIn(labels), FadeIn(q), run_time=0.4)
        scores = VGroup(*[Text(f"{IT_SCORES[j]:+.2f}" if j < 8 else "−∞", font=MONO, font_size=22,
                               color=WHITE if j < 8 else RED_B).move_to([xs[j], 1.5, 0]) for j in range(10)])
        sl = Text("scores ÷ 8", font_size=18, color=GREY_B).move_to([-1.5, 3.1, 0])
        self.at("every")
        self.play(LaggedStart(*[FadeIn(scores[j]) for j in range(8)], lag_ratio=0.08), FadeIn(sl), run_time=0.9)
        self.at("mask")
        self.play(FadeIn(scores[8:]), run_time=0.4)
        bars = VGroup()
        for j in range(10):
            v = IT_WEIGHTS[j] if j < 8 else 0.0
            b = Rectangle(width=0.8, height=max(0.02, 3.2 * v), stroke_width=0,
                          fill_color=GREEN_C if j == 1 else GREY_B, fill_opacity=0.85)
            b.move_to([xs[j], -2.6 + b.height / 2, 0])
            lab = Text(f"{v:.3f}", font=MONO, font_size=18).next_to(b, UP, buff=0.08)
            bars.add(VGroup(b, lab))
        self.at("softmax")
        self.play(LaggedStart(*[FadeIn(b, shift=0.1 * UP) for b in bars], lag_ratio=0.05), run_time=0.9)
        pct = Text("84% → cat", font_size=28, color=GREEN_B).move_to([2.5, 0.2, 0])
        self.at("84")
        self.play(FadeIn(pct), Indicate(labels[1], color=GREEN_B), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 5. the matrix
    def matrix(self):
        self.section(5)
        self.clear_stage()
        cells, labels = heat(W, 0.58, font_size=13)
        hm = VGroup(cells, labels).move_to([-1.6, -0.3, 0])
        rl = VGroup(*[Text(w, font_size=16, color=GREY_B).next_to(cells[10 * i], LEFT, buff=0.12)
                      for i, w in enumerate(WORDS)])
        cl = VGroup(*[Text(w, font_size=14, color=GREY_B).rotate(PI / 3).next_to(cells[j], UP, buff=0.1)
                      for j, w in enumerate(WORDS)])
        self.at("query")
        self.play(FadeIn(cells, lag_ratio=0.005), FadeIn(labels), FadeIn(rl), FadeIn(cl), run_time=1.0)
        info = VGroup(Text("10 × 10 = 100 entries", font_size=24),
                      Text("45 masked", font_size=24, color=RED_B)).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        info.move_to([4.4, 1.4, 0])
        self.at("100")
        self.play(FadeIn(info), run_time=0.4)
        same = Text("vs transformers:\ndifference 0.0", font=MONO, font_size=22, color=GREEN_B,
                    line_spacing=0.8).move_to([4.4, 0.0, 0])
        self.at("exactly")
        self.play(FadeIn(same), run_time=0.4)
        col = SurroundingRectangle(VGroup(*[cells[10 * i + 1] for i in range(1, 10)]), color=YELLOW, buff=0.03)
        cn = Text("most words\nlook back at “cat”", font_size=22, color=YELLOW, line_spacing=0.8).move_to([4.4, -1.6, 0])
        self.at("cat")
        self.play(Create(col), FadeIn(cn), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 6. why divide by 8
    def scaling(self):
        self.section(6)
        self.clear_stage()
        head = Text("why divide by 8?", font_size=34).to_edge(UP, buff=0.6)
        self.at("why")
        self.play(FadeIn(head), run_time=0.4)
        spread = Text("random 64-number vectors: dot products spread by 8.00 (= √64)", font_size=24,
                      color=GREY_A).move_to([0, 1.8, 0])
        self.at("spreads")
        self.play(FadeIn(spread), run_time=0.4)
        rows = VGroup()
        for name, val, col in (("÷ 8", 0.68, GREEN_C), ("no ÷", 0.98, RED_C)):
            bar = Rectangle(width=7 * val, height=0.6, stroke_width=0, fill_color=col, fill_opacity=0.85)
            lab = Text(name, font=MONO, font_size=26, color=col)
            num = Text(f"{val:.2f}", font=MONO, font_size=26)
            rows.add(VGroup(lab, bar, num))
        for k, r in enumerate(rows):
            r[0].move_to([-5.0, 0.5 - 1.1 * k, 0])
            r[1].move_to([-4.2, 0.5 - 1.1 * k, 0], aligned_edge=LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.2)
        cap = Text("largest weight per row, on average", font_size=22, color=GREY_B).move_to([0, -1.9, 0])
        self.at("without")
        self.play(FadeIn(rows[0]), FadeIn(cap), run_time=0.5)
        self.at("98")
        self.play(FadeIn(rows[1]), run_time=0.4)
        hot = Text("almost one-hot: much harder to train", font_size=26, color=RED_B).move_to([0, -2.8, 0])
        self.at("one")
        self.play(FadeIn(hot), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. values and heads
    def values(self):
        self.section(7)
        self.clear_stage()
        eq = Text("output(“it”) = 0.837 × value(cat) + 0.056 × value(The) + …", font=MONO, font_size=24)
        eq.move_to([0, 2.4, 0])
        self.at("values")
        self.play(FadeIn(eq), run_time=0.5)
        vec = Text("= [−0.325, +0.455, −0.480, −0.295, … 64 numbers]", font=MONO, font_size=22, color=GREEN_B)
        vec.next_to(eq, DOWN, buff=0.3)
        self.at("average")
        self.play(FadeIn(vec), run_time=0.4)
        chunks = VGroup(*[Rectangle(width=0.5, height=0.5, stroke_width=1, stroke_color=BLACK,
                                    fill_color=GREEN_C if k == 3 else GREEN_E, fill_opacity=0.85) for k in range(12)])
        chunks.arrange(RIGHT, buff=0).move_to([-2.0, -0.6, 0])
        cl = Text("12 heads × 64 = 768", font_size=22, color=GREY_A).next_to(chunks, DOWN, buff=0.2)
        self.at("joined")
        self.play(FadeIn(chunks, lag_ratio=0.1), FadeIn(cl), run_time=0.8)
        mix = Square(1.4, color=MODEL_COLOR, fill_opacity=0.3).move_to([3.3, -0.6, 0])
        ml = Text("768 × 768", font=MONO, font_size=20).move_to(mix)
        arr = Arrow(chunks.get_right(), mix.get_left(), buff=0.2)
        self.at("mixed")
        self.play(GrowArrow(arr), FadeIn(mix), FadeIn(ml), run_time=0.6)
        same = Text("difference from transformers: 0.0", font=MONO, font_size=24, color=GREEN_B).move_to([0, -2.5, 0])
        self.at("identical")
        self.play(FadeIn(same), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. other heads
    def other_heads(self):
        self.section(8)
        self.clear_stage()
        n = 8
        prev = [[1.0 if (j == i - 1 or (i == 0 and j == 0)) else 0.0 for j in range(n)] for i in range(n)]
        first = [[(0.75 if j == 0 else 0.25 / i) if (j <= i and i > 0) else (1.0 if j == 0 else 0.0)
                  for j in range(n)] for i in range(n)]
        c1, _ = heat(prev, 0.42, colour=BLUE_C)
        c2, _ = heat(first, 0.42, colour=GOLD)
        c1.move_to([-3.3, 0.0, 0])
        c2.move_to([3.3, 0.0, 0])
        l1 = Text("layer 4, head 11:\n1.00 on the previous token", font_size=22, color=BLUE_B,
                  line_spacing=0.8).next_to(c1, DOWN, buff=0.35)
        l2 = Text("92 of 144 heads:\nmore than half on the first token", font_size=22, color=GOLD,
                  line_spacing=0.8).next_to(c2, DOWN, buff=0.35)
        sk = Text("schematic", font_size=16, color=GREY_B).next_to(c2, UP, buff=0.15)
        self.at("habits")
        self.play(FadeIn(Text("other heads, other habits", font_size=32).to_edge(UP, buff=0.6)), run_time=0.4)
        self.at("before")
        self.play(FadeIn(c1, lag_ratio=0.01), FadeIn(l1), run_time=0.7)
        self.at("92")
        self.play(FadeIn(c2, lag_ratio=0.01), FadeIn(l2), FadeIn(sk), run_time=0.7)
        nx = Text("why? → the attention-sinks episode", font_size=22, color=GREY_A).to_edge(DOWN, buff=0.5)
        self.at("episode")
        self.play(FadeIn(nx), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. scale
    def scale(self):
        self.section(9)
        self.clear_stage()
        a = Text("12 layers × 12 heads = 144 matrices per forward pass", font_size=28).move_to([0, 1.2, 0])
        self.at("144")
        self.play(FadeIn(a), run_time=0.5)
        b = Text("at 1,024 tokens: 144 × 1,024 × 1,024 = 150,994,944 weights", font=MONO, font_size=24,
                 color=YELLOW).move_to([0, 0.0, 0])
        self.at("151")
        self.play(FadeIn(b), run_time=0.5)
        c = Text("keeping that memory in check: a story of its own", font_size=24, color=GREY_A).move_to([0, -1.2, 0])
        self.at("memory")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. code
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("queries")
        self.play(Create(hl), run_time=0.3)
        self.at("masked")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.at("soft", "softmaxed")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.at("values")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 11. outro
    def outro(self):
        self.section(11)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
