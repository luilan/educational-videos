"""How LLMs Work: Deep Dive, episode 7 — Causal Masking, in Detail.

Render from the repo root:  ./render.sh deep-dive d07
Every number on screen comes from code/d07_causal_mask/causal_mask.py (the 4-token example, GPT-2 small with real
weights, and an 818,241-parameter tiny GPT trained 1,500 steps on Tiny Shakespeare with and without the mask).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d07_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 7"
SCORES = [[2.0, 1.0, 0.5, 3.0], [1.0, 2.0, 1.5, 0.0], [0.5, 1.0, 2.0, 1.0], [1.0, 0.5, 1.0, 2.0]]
WEIGHTS = [[1.0, 0, 0, 0], [0.269, 0.731, 0, 0], [0.140, 0.231, 0.629, 0], [0.188, 0.114, 0.188, 0.510]]
GPT2_TOKENS = ["The", " cat", " sat", " on", " the", " mat"]
GPT2_ATT = [[1.00, 0, 0, 0, 0, 0], [0.70, 0.30, 0, 0, 0, 0], [0.60, 0.11, 0.29, 0, 0, 0],
            [0.45, 0.18, 0.33, 0.04, 0, 0], [0.46, 0.18, 0.28, 0.03, 0.04, 0], [0.32, 0.10, 0.04, 0.10, 0.13, 0.30]]
MASKED_SAMPLE = "Capper:\n\nMARTIUS:\nMy bethat's one faces.\n\nBUCKEND:\nLady a Kingless, it. Look, anoal now"
CODE = """scores = q @ k.transpose(-2, -1) / math.sqrt(d)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float("-inf"))     # the future
weights = scores.softmax(dim=-1)

# or, in PyTorch's attention function, one flag:
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)"""


def grid(values, size=0.9, fmt="{:.1f}", font_size=24, colour_fn=None):
    """A square of cells with numbers; returns (cells, labels) as VGroups in row-major order."""
    n = len(values)
    cells, labels = VGroup(), VGroup()
    for i in range(n):
        for j in range(n):
            v = values[i][j]
            fill = colour_fn(i, j, v) if colour_fn else GREY_E
            c = Square(size, stroke_color=GREY_B, stroke_width=1.5, fill_color=fill, fill_opacity=0.9)
            c.move_to([(j - (n - 1) / 2) * size, ((n - 1) / 2 - i) * size, 0])
            cells.add(c)
            labels.add(Text(fmt.format(v) if isinstance(v, float) else str(v), font=MONO, font_size=font_size).move_to(c))
    return cells, labels


class CausalMaskVideo(VoicedScene):
    VIDEO = "d07"

    def construct(self):
        play_token_intro(self, TITLE, 7, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.by_hand()       # 2
        self.why_inf()       # 3
        self.gpt2()          # 4
        self.experiment()    # 5
        self.results()       # 6
        self.generation()    # 7
        self.efficiency()    # 8
        self.inference()     # 9
        self.code()          # 10
        self.outro()         # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        words = ["The", "cat", "sat", "on", "the", "mat"]
        toks = VGroup(*[token(w) for w in words]).arrange(RIGHT, buff=0.25).move_to([0, 1.8, 0])
        self.at("predict")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * UP) for t in toks], lag_ratio=0.1), run_time=0.8)
        q = Text("next?", font_size=24, color=YELLOW).next_to(toks[2], DOWN, buff=0.5)
        arr = Arrow(toks[2].get_bottom(), q.get_top(), buff=0.08, color=YELLOW)
        self.at("next")
        self.play(GrowArrow(arr), FadeIn(q), run_time=0.4)
        self.at("whole")
        self.play(Indicate(toks[3:], color=RED_B), run_time=0.7)
        peek = CurvedArrow(toks[2].get_top() + 0.05 * UP, toks[3].get_top() + 0.05 * UP, angle=-1.6, color=RED_B)
        pl = Text("peek at the answer?", font_size=24, color=RED_B).next_to(peek, UP, buff=0.1)
        self.at("peaking", "peeking")
        self.play(Create(peek), FadeIn(pl), run_time=0.5)
        cells, _ = grid([[0] * 5 for _ in range(5)], size=0.45, colour_fn=lambda i, j, v: RED_E if j > i else GREEN_E)
        cells.move_to([0, -1.6, 0])
        self.at("triangle")
        self.play(FadeIn(cells, lag_ratio=0.03), run_time=0.8)
        cl = Text("the causal mask", font_size=30, color=YELLOW).next_to(cells, RIGHT, buff=0.5)
        self.at("causal")
        self.play(FadeIn(cl), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. by hand
    def by_hand(self):
        self.section(2)
        self.clear_stage()
        cells, labels = grid(SCORES)
        mat = VGroup(cells, labels).move_to([-3.2, -0.2, 0])
        head = Text("attention scores, 4 tokens", font_size=28).next_to(mat, UP, buff=0.7)
        ql = Text("query ↓", font_size=20, color=GREY_B).next_to(cells, LEFT, buff=0.2)
        kl = Text("key →", font_size=20, color=GREY_B).next_to(cells, UP, buff=0.15)
        self.at("scores")
        self.play(FadeIn(head), FadeIn(mat), run_time=0.6)
        self.at("query")
        self.play(FadeIn(ql), run_time=0.3)
        self.at("key")
        self.play(FadeIn(kl), run_time=0.3)
        future = [4 * i + j for i in range(4) for j in range(4) if j > i]
        self.at("diagonal")
        self.play(*[cells[k].animate.set_fill(RED_E) for k in future], run_time=0.5)
        self.at("infinity")
        self.play(*[Transform(labels[k], Text("−∞", font=MONO, font_size=24, color=RED_B).move_to(labels[k]))
                    for k in future], run_time=0.6)
        w_cells, w_labels = grid(WEIGHTS, fmt="{:.3f}", font_size=18,
                                 colour_fn=lambda i, j, v: interpolate_color(GREY_E, GREEN_C, v))
        wmat = VGroup(w_cells, w_labels).move_to([3.2, -0.2, 0])
        arr = Arrow(mat.get_right(), wmat.get_left(), buff=0.25)
        al = Text("softmax\nper row", font_size=20, line_spacing=0.8).next_to(arr, UP, buff=0.1)
        self.at("softmax")
        self.play(GrowArrow(arr), FadeIn(al), FadeIn(wmat), run_time=0.8)
        zl = Text("e^(−∞) = 0", font=MONO, font_size=24, color=RED_B).next_to(wmat, UP, buff=0.3)
        self.at("zero")
        self.play(FadeIn(zl), *[Indicate(w_labels[k], color=RED_B) for k in future], run_time=0.6)
        sums = VGroup(*[Text("= 1", font_size=20, color=GREEN_B).next_to(w_cells[4 * i + 3], RIGHT, buff=0.15)
                        for i in range(4)])
        self.at("one")
        self.play(FadeIn(sums), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. why -inf
    def why_inf(self):
        self.section(3)
        self.clear_stage()
        q = Text("why −∞, not 0?", font_size=36).to_edge(UP, buff=0.8)
        self.at("why")
        self.play(FadeIn(q), run_time=0.4)
        e0 = Text("e^0 = 1  →  a score of 0 still gets weight", font=MONO, font_size=26, color=YELLOW).move_to([0, 1.9, 0])
        self.at("counts")
        self.play(FadeIn(e0), run_time=0.4)
        row = [0.232, 0.085, 0.052, 0.631]
        bars = VGroup()
        for j, v in enumerate(row):
            b = Rectangle(width=1.2, height=3.4 * v, stroke_width=0, fill_color=RED_C if j > 0 else GREEN_C,
                          fill_opacity=0.85)
            b.move_to([(j - 1.5) * 1.8, -2.6 + 1.7 * v, 0])
            lab = VGroup(Text(f"token {j + 1}", font_size=20, color=GREY_B).next_to(b, DOWN, buff=0.15),
                         Text(f"{v:.3f}", font=MONO, font_size=20).next_to(b, UP, buff=0.1))
            bars.add(VGroup(b, lab))
        cap = Text("token 1's attention, with no mask", font_size=24, color=GREY_A).move_to([0, 1.1, 0])
        self.at("without")
        self.play(FadeIn(cap), LaggedStart(*[FadeIn(b, shift=0.2 * UP) for b in bars], lag_ratio=0.1), run_time=0.8)
        pct = Text("63% on a future word", font_size=26, color=RED_B).next_to(bars[3], LEFT, buff=0.3).shift(0.6 * UP)
        self.at("four")
        self.play(Indicate(bars[3], color=RED_B), FadeIn(pct), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 4. GPT-2
    def gpt2(self):
        self.section(4)
        self.clear_stage()
        head = Text("GPT-2, real weights", font_size=30).to_edge(UP, buff=0.4)
        self.at("gpt")
        self.play(FadeIn(head), run_time=0.4)
        a = VGroup(*[token(w.strip()) for w in GPT2_TOKENS]).arrange(RIGHT, buff=0.55).move_to([0, 2.0, 0])
        b = VGroup(*[token(w.strip()) for w in GPT2_TOKENS[:-1] + [" moon"]])
        for k in range(6):
            b[k].move_to([a[k].get_x(), a.get_y() - 0.75, 0])
        self.at("cat")
        self.play(FadeIn(a), run_time=0.5)
        self.at("moon")
        self.play(FadeIn(b), run_time=0.4)
        self.play(b[5][0].animate.set_stroke(YELLOW).set_fill(YELLOW, 0.3), run_time=0.3)
        diffs = VGroup(*[Text("Δ 0.0", font=MONO, font_size=18, color=GREEN_B).next_to(b[k], DOWN, buff=0.2)
                         for k in range(5)])
        self.at("identical")
        self.play(FadeIn(diffs, lag_ratio=0.1), run_time=0.6)
        last = Text("Δ up to\n14.7", font=MONO, font_size=18, color=RED_B, line_spacing=0.8).next_to(b[5], DOWN, buff=0.2)
        self.at("last")
        self.play(FadeIn(last), run_time=0.4)
        top = VGroup(a, b, diffs, last)
        self.at("pattern")
        cells, labels = grid(GPT2_ATT, size=0.55, fmt="{:.2f}", font_size=14,
                             colour_fn=lambda i, j, v: interpolate_color(GREY_E, GREEN_C, v) if j <= i else BLACK)
        hm = VGroup(cells, labels).move_to([0.6, -1.5, 0])
        rl = VGroup(*[Text(w.strip(), font_size=16, color=GREY_B).next_to(cells[6 * i], LEFT, buff=0.15)
                      for i, w in enumerate(GPT2_TOKENS)])
        cap = Text("layer 1, head 1", font_size=20, color=GREY_B).next_to(cells, RIGHT, buff=0.4)
        self.play(top.animate.scale(0.8).shift(0.4 * UP), FadeIn(hm), FadeIn(rl), FadeIn(cap), run_time=0.8)
        upper = [6 * i + j for i in range(6) for j in range(6) if j > i]
        self.at("diagonal")
        self.play(*[Indicate(labels[k], color=RED_B) for k in upper], run_time=0.6)
        self.at("itself")
        self.play(cells[0].animate.set_stroke(YELLOW, 4), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. experiment
    def experiment(self):
        self.section(5)
        self.clear_stage()
        head = Text("what if we drop the mask?", font_size=34).to_edge(UP, buff=0.7)
        self.at("drop")
        self.play(FadeIn(head), run_time=0.4)
        boxes = VGroup()
        for flag, col in (("is_causal=True", GREEN_C), ("is_causal=False", RED_C)):
            r = RoundedRectangle(width=5.0, height=2.0, corner_radius=0.2, color=col)
            t = VGroup(Text("tiny GPT", font_size=28), Text(flag, font=MONO, font_size=24, color=col)).arrange(DOWN)
            boxes.add(VGroup(r, t.move_to(r)))
        boxes.arrange(RIGHT, buff=1.0).move_to([0, 0.2, 0])
        self.at("twice")
        self.play(FadeIn(boxes[0]), FadeIn(boxes[1]), run_time=0.6)
        info = Text("Shakespeare · 818,241 parameters · 1,500 steps", font_size=26, color=GREY_A).move_to([0, -1.8, 0])
        self.at("818")
        self.play(FadeIn(info), run_time=0.4)
        self.at("flag")
        self.play(Indicate(boxes[0][1][1]), Indicate(boxes[1][1][1]), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 6. results
    def results(self):
        self.section(6)
        self.clear_stage()
        hdr = VGroup(Dot(radius=0.01, fill_opacity=0), Text("mask on", font_size=26, color=GREEN_C),
                     Text("mask off", font_size=26, color=RED_C))
        r1 = VGroup(Text("training loss", font_size=26), Text("1.52", font=MONO, font_size=30),
                    Text("0.04", font=MONO, font_size=30, color=RED_B))
        r2 = VGroup(Text("honest loss (past only)", font_size=26), Text("1.73", font=MONO, font_size=30),
                    Text("7.36", font=MONO, font_size=30, color=RED_B))
        xs, ys = [-3.2, 1.0, 4.0], [1.6, 0.6, -0.5]
        for row, y in zip((hdr, r1, r2), ys):
            for m, x in zip(row, xs):
                m.move_to([x, y, 0])
        self.play(FadeIn(hdr), FadeIn(r1[0]), run_time=0.4)
        self.at("04")
        self.play(FadeIn(r1[2], scale=1.3), run_time=0.4)
        cheat = Text("it just reads the next character", font_size=22, color=RED_B).next_to(r1[2], DOWN, buff=0.12)
        self.at("reads")
        self.play(FadeIn(cheat), run_time=0.4)
        self.at("honestly")
        self.play(FadeOut(cheat), FadeIn(r2[0]), run_time=0.4)
        self.at("36")
        self.play(FadeIn(r2[2], scale=1.3), run_time=0.4)
        rnd = Text("random guessing over 65 characters: ln 65 = 4.17", font_size=24, color=YELLOW).move_to([0, -1.8, 0])
        self.at("random")
        self.play(FadeIn(rnd), run_time=0.4)
        self.at("52")
        self.play(FadeIn(r1[1]), run_time=0.3)
        self.at("73")
        self.play(FadeIn(r2[1]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. generation
    def generation(self):
        self.section(7)
        self.clear_stage()
        left_h = Text("mask on", font_size=28, color=GREEN_C).move_to([-3.5, 2.6, 0])
        right_h = Text("mask off", font_size=28, color=RED_C).move_to([3.5, 2.6, 0])
        self.at("write")
        self.play(FadeIn(left_h), FadeIn(right_h), run_time=0.4)
        good = Text(MASKED_SAMPLE, font=MONO, font_size=18, line_spacing=0.7).move_to([-3.5, 0.0, 0])
        self.at("shakespeare")
        self.play(AddTextLetterByLetter(good, time_per_char=0.02), run_time=1.6)
        copy = Text("learned to copy, not to predict", font_size=24, color=RED_B).move_to([3.5, 1.6, 0])
        self.at("copy")
        self.play(FadeIn(copy), run_time=0.4)
        nl = Text("↵ ↵ ↵ ↵ ↵ ↵ ↵ ↵\n… 65 empty lines …", font=MONO, font_size=22, color=GREY_B,
                  line_spacing=0.8).move_to([3.5, 0.3, 0])
        self.at("empty")
        self.play(FadeIn(nl), run_time=0.5)
        bad = Text("OWoNG a miox, a oalf gaie the\nEs nopgr too cutelor dous", font=MONO, font_size=18,
                   line_spacing=0.7, color=RED_B).move_to([3.5, -1.0, 0])
        self.at("gibberish")
        self.play(FadeIn(bad), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 8. efficiency
    def efficiency(self):
        self.section(8)
        self.clear_stage()
        text = "To be, or not to be"
        boxes = VGroup(*[Square(0.55, stroke_color=GREY_B, stroke_width=1.5) for _ in text]).arrange(RIGHT, buff=0.05)
        boxes.move_to([0, 1.2, 0])
        chars = VGroup(*[Text(c if c != " " else "␣", font=MONO, font_size=22).move_to(b) for c, b in zip(text, boxes)])
        self.at("efficient")
        self.play(FadeIn(boxes), FadeIn(chars), run_time=0.6)
        preds = VGroup(*[Text(text[k + 1] if text[k + 1] != " " else "␣", font=MONO, font_size=20, color=YELLOW)
                         .next_to(boxes[k], DOWN, buff=0.6) for k in range(len(text) - 1)])
        arrs = VGroup(*[Arrow(boxes[k].get_bottom(), preds[k].get_top(), buff=0.05, stroke_width=3,
                              max_tip_length_to_length_ratio=0.3, color=YELLOW) for k in range(len(text) - 1)])
        self.at("prediction")
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(p)) for a, p in zip(arrs, preds)],
                              lag_ratio=0.08), run_time=1.6)
        cap = Text("each from only what came before", font_size=24, color=GREY_A).move_to([0, -1.0, 0])
        self.at("before")
        self.play(FadeIn(cap), run_time=0.4)
        big = Text("64 characters → 64 training examples, one pass", font_size=30, color=YELLOW).move_to([0, -2.2, 0])
        self.at("64")
        self.play(FadeIn(big), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 9. inference
    def inference(self):
        self.section(9)
        self.clear_stage()
        keys = VGroup(*[Square(0.55, stroke_color=GREY_B, fill_color=GREEN_E, fill_opacity=0.9) for _ in range(7)])
        keys.arrange(RIGHT, buff=0.05).move_to([-2.6, 1.0, 0])
        kl = Text("cached keys", font_size=22, color=GREY_B).next_to(keys, UP, buff=0.2)
        ql = Text("new token's query: sees all of them, no mask needed", font_size=22, color=GREEN_B)
        ql.next_to(keys, DOWN, buff=0.3)
        self.at("inference")
        self.play(FadeIn(keys), FadeIn(kl), run_time=0.5)
        self.at("cache")
        self.play(FadeIn(ql), run_time=0.4)
        cells, _ = grid([[0] * 5 for _ in range(5)], size=0.4, colour_fn=lambda i, j, v: GREEN_E)
        bert = VGroup(cells, Text("BERT: no mask,\nsees both sides", font_size=22, line_spacing=0.8)
                      .next_to(cells, RIGHT, buff=0.3)).move_to([3.0, -1.7, 0])
        cant = Text("but cannot write left to right", font_size=20, color=RED_B).next_to(bert, DOWN, buff=0.2)
        self.at("bert")
        self.play(FadeIn(bert), run_time=0.5)
        self.at("cannot")
        self.play(FadeIn(cant), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. code
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.2, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("fill")
        self.play(Create(hl), run_time=0.3)
        self.at("flag")
        self.play(highlight(hl, code, 6), run_time=0.4)
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
