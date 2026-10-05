"""How LLMs Work: Deep Dive, episode 11 — Sliding Windows and Sparse Attention.

Render from the repo root:  ./render.sh deep-dive d11
Every number on screen comes from code/d11_sliding_window/sliding_window.py (pair counts; reach measured with gradients;
a tiny GPT trained 1,500 steps on 256-character Shakespeare texts and on a 128-symbol copy task, full vs windows).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d11_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 11"
CODE = """def window_mask(T, W):
    i, j = torch.arange(T)[:, None], torch.arange(T)[None]
    return (j <= i) & (i - j < W)      # not in the future, and less than W back

att = F.scaled_dot_product_attention(q, k, v, attn_mask=window_mask(T, W))"""


def mask_grid(n, allowed, size, on=BLUE_C, off=BLACK):
    g = VGroup()
    for i in range(n):
        for j in range(n):
            g.add(Square(size, stroke_width=0.8, stroke_color=GREY_D, fill_color=on if allowed(i, j) else off,
                         fill_opacity=0.9).move_to([j * size, -i * size, 0]))
    return g


def bar_rows(rows, scale, x0=-2.0, y0=1.2, dy=0.9, fmt="{:.3f}"):
    out = VGroup()
    for k, (name, v, col) in enumerate(rows):
        y = y0 - dy * k
        lab = Text(name, font_size=22).move_to([x0 - 0.3, y, 0], aligned_edge=RIGHT)
        b = Rectangle(width=max(0.04, v * scale), height=0.5, stroke_width=0, fill_color=col, fill_opacity=0.85)
        b.move_to([x0, y, 0], aligned_edge=LEFT)
        num = Text(fmt.format(v), font=MONO, font_size=22).next_to(b, RIGHT, buff=0.15)
        out.add(VGroup(lab, b, num))
    return out


class SlidingWindowVideo(VoicedScene):
    VIDEO = "d11"

    def construct(self):
        play_token_intro(self, TITLE, 11, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.mask()          # 2
        self.stacking()      # 3
        self.shakespeare()   # 4
        self.copy_task()     # 5
        self.mixing()        # 6
        self.sparse()        # 7
        self.cache()         # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        n = 16
        full = mask_grid(n, lambda i, j: j <= i, 0.25).move_to([-3.4, -0.2, 0])
        self.at("square")
        self.play(FadeIn(full, lag_ratio=0.002), run_time=0.8)
        a = Text("131,072 tokens:\n8.6 billion pairs\nper head, per layer", font_size=26, line_spacing=0.9)
        a.move_to([2.6, 0.8, 0])
        self.at("131")
        self.play(FadeIn(a), run_time=0.5)
        band = mask_grid(n, lambda i, j: j <= i and i - j < 4, 0.25).move_to(full)
        w = Text("sliding window:\nonly the last few", font_size=26, color=YELLOW, line_spacing=0.9).move_to([2.6, -1.4, 0])
        self.at("window")
        self.play(Transform(full, band), FadeIn(w), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 2. mask
    def mask(self):
        self.section(2)
        self.clear_stage()
        g = mask_grid(8, lambda i, j: j <= i and i - j < 4, 0.55).move_to([-3.0, -0.2, 0])
        lab = Text("window of 4, 8 tokens", font_size=24).next_to(g, UP, buff=0.3)
        self.at("four")
        self.play(FadeIn(g, lag_ratio=0.01), FadeIn(lab), run_time=0.8)
        b = Text("a band along the diagonal", font_size=26, color=BLUE_B).move_to([3.0, 0.8, 0])
        self.at("band")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("window 4,096 on 131,072 tokens:\n528 million of 8.6 billion pairs\n= 6.2%", font_size=24,
                 color=YELLOW, line_spacing=0.9).move_to([3.0, -0.8, 0])
        self.at("6")
        self.play(FadeIn(c), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 3. stacking
    def stacking(self):
        self.section(3)
        self.clear_stage()
        n = 64
        dots = VGroup(*[Dot(radius=0.05, color=GREY_B) for _ in range(n)]).arrange(RIGHT, buff=0.11).move_to([0, 2.2, 0])
        last = dots[-1]
        last.set_color(YELLOW).scale(1.6)
        lab = Text("window 16: how far back the last token reaches", font_size=24).move_to([0, 2.9, 0])
        self.at("hop")
        self.play(FadeIn(dots), FadeIn(lab), run_time=0.5)
        spans = VGroup()
        for k, reach in enumerate((15, 30, 45, 60)):
            y = 1.4 - 0.75 * k
            start = dots[n - 1 - reach].get_x()
            bar = Rectangle(width=last.get_x() - start, height=0.45, stroke_width=0, fill_color=BLUE_C,
                            fill_opacity=0.35).move_to([(start + last.get_x()) / 2, y, 0])
            t = Text(f"{k + 1} layer{'s' if k else ''}: {reach} tokens", font_size=18)
            t.move_to(bar.get_left() + 0.12 * RIGHT, aligned_edge=LEFT)
            spans.add(VGroup(bar, t))
        for k, cue in enumerate(("15", "30", "45", "60")):
            self.at(cue)
            self.play(GrowFromEdge(spans[k][0], RIGHT), FadeIn(spans[k][1]), run_time=0.4)
        g = Text("measured with gradients", font_size=20, color=GREY_B).move_to([3.5, -1.6, 0])
        self.at("gradients")
        self.play(FadeIn(g), run_time=0.3)
        m = Text("Mistral 7B: window 4,096 × 32 layers ≈ 131,072 tokens (in theory)", font_size=24,
                 color=YELLOW).move_to([0, -2.6, 0])
        self.at("mistral")
        self.play(FadeIn(m), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 4. shakespeare
    def shakespeare(self):
        self.section(4)
        self.clear_stage()
        head = Text("tiny GPT, 256-character Shakespeare texts, 1,500 steps", font_size=26).to_edge(UP, buff=0.6)
        self.at("hurt")
        self.play(FadeIn(head), run_time=0.4)
        rows = bar_rows([("full attention (100% of pairs)", 1.780, GREY_B), ("window 64 (44%)", 1.750, BLUE_C),
                         ("window 16 (12%)", 1.693, GREEN_C)], scale=3.0, x0=0.2, y0=1.2, dy=1.0)
        cap = Text("validation loss (lower is better)", font_size=20, color=GREY_B).move_to([0, -1.6, 0])
        for k, cue in enumerate(("78", "75", "69")):
            self.at(cue)
            self.play(FadeIn(rows[k], shift=0.1 * RIGHT), *([FadeIn(cap)] if k == 0 else []), run_time=0.4)
        better = Text("the windows did better", font_size=28, color=GREEN_B).move_to([0, -2.4, 0])
        self.at("better")
        self.play(FadeIn(better), run_time=0.4)
        why = Text("nearby characters matter most", font_size=24, color=GREY_A).move_to([0, -3.1, 0])
        self.at("shakespeare")
        self.play(FadeIn(why), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. copy task
    def copy_task(self):
        self.section(5)
        self.clear_stage()
        seq = Text("K D A P F … M   |   K D A P F … M", font=MONO, font_size=28).move_to([0, 2.6, 0])
        arr = CurvedArrow(seq.get_right() + 2.6 * LEFT + 0.25 * UP, seq.get_left() + 0.35 * RIGHT + 0.25 * UP,
                          angle=0.6, color=YELLOW)
        al = Text("128 back", font_size=20, color=YELLOW).next_to(arr, UP, buff=0.05)
        self.at("range")
        self.play(FadeIn(seq), run_time=0.5)
        self.play(Create(arr), FadeIn(al), run_time=0.5)
        rows = bar_rows([("full attention", 0.000, GREEN_C), ("window 64", 2.775, RED_C), ("window 16", 2.775, RED_C)],
                        scale=2.2, x0=-0.6, y0=0.6, dy=0.9)
        guess = DashedLine([-0.6 + 2.77 * 2.2, 1.0, 0], [-0.6 + 2.77 * 2.2, -1.6, 0], color=GREY_B)
        gl = Text("guessing: ln 16 = 2.77", font_size=18, color=GREY_B).next_to(guess, DOWN, buff=0.1)
        self.at("copy")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("77")
        self.play(FadeIn(rows[1]), FadeIn(rows[2]), Create(guess), FadeIn(gl), run_time=0.6)
        note = Text("window 64: reach on paper 252 tokens, never used", font_size=24, color=RED_B).move_to([0, -2.8, 0])
        self.at("252")
        self.play(FadeIn(note), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. mixing
    def mixing(self):
        self.section(6)
        self.clear_stage()
        pros = VGroup(Text("✓ cheap", font_size=28, color=GREEN_B), Text("✓ fine for local patterns", font_size=28,
                                                                       color=GREEN_B),
                      Text("✗ loses exact recall far back", font_size=28, color=RED_B)).arrange(DOWN, buff=0.3,
                                                                                            aligned_edge=LEFT)
        pros.move_to([-3.2, 0.3, 0])
        self.at("cheap")
        self.play(FadeIn(pros, lag_ratio=0.3), run_time=0.9)
        layers = VGroup()
        for k in range(6):
            local = k % 2 == 0
            r = Rectangle(width=3.4, height=0.5, stroke_width=1, color=BLUE_C if local else GOLD, fill_opacity=0.35)
            t = Text("sliding window" if local else "full attention", font_size=20).move_to(r)
            layers.add(VGroup(r, t))
        layers.arrange(UP, buff=0.12).move_to([3.4, 0.2, 0])
        self.at("mix")
        self.play(FadeIn(layers, lag_ratio=0.1), run_time=0.8)
        g = Text("Gemma 2: alternating layers", font_size=22, color=GREY_A).next_to(layers, DOWN, buff=0.3)
        self.at("gemma")
        self.play(FadeIn(g), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. sparse
    def sparse(self):
        self.section(7)
        self.clear_stage()
        n, s = 12, 0.26
        pats = [("sliding window", lambda i, j: j <= i and i - j < 3),
                ("+ global tokens", lambda i, j: j <= i and (i - j < 3 or j == 0 or i == 0)),
                ("strided", lambda i, j: j <= i and (i - j < 3 or (i - j) % 4 == 0))]
        grids = VGroup()
        for name, f in pats:
            g = mask_grid(n, f, s)
            grids.add(VGroup(g, Text(name, font_size=22).next_to(g, DOWN, buff=0.25)))
        grids.arrange(RIGHT, buff=1.0).move_to([0, 0.3, 0])
        self.at("sparse")
        self.play(FadeIn(grids[0]), run_time=0.5)
        self.at("global")
        self.play(FadeIn(grids[1]), run_time=0.5)
        self.at("stride")
        self.play(FadeIn(grids[2]), run_time=0.5)
        cap = Text("always: choose which pairs are worth computing", font_size=26, color=YELLOW).to_edge(DOWN, buff=0.7)
        self.at("pairs")
        self.play(FadeIn(cap), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. cache
    def cache(self):
        self.section(8)
        self.clear_stage()
        head = Text("at inference: a rolling KV cache of W tokens", font_size=28).to_edge(UP, buff=0.8)
        self.at("inference")
        self.play(FadeIn(head), run_time=0.4)
        slots = VGroup(*[Square(0.8, color=GREEN_C, fill_opacity=0.3) for _ in range(6)]).arrange(RIGHT, buff=0.08)
        slots.move_to([0, 0.2, 0])
        toks = VGroup(*[Text(f"t{k}", font=MONO, font_size=22).move_to(slots[k]) for k in range(6)])
        self.at("cache")
        self.play(FadeIn(slots), FadeIn(toks), run_time=0.5)
        self.at("buffer")
        for step in range(2):
            new = Text(f"t{6 + step}", font=MONO, font_size=22, color=YELLOW).next_to(slots, RIGHT, buff=0.4)
            self.play(FadeIn(new), run_time=0.2)
            shifted = VGroup(*toks[1:], new)
            self.play(FadeOut(toks[0], shift=LEFT), *[t.animate.move_to(slots[k]) for k, t in enumerate(shifted)],
                      run_time=0.5)
            toks = shifted
        cap = Text("fixed size, however long the text", font_size=24, color=YELLOW).move_to([0, -1.6, 0])
        self.play(FadeIn(cap), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=21)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("future")
        self.play(Create(hl), run_time=0.3)
        self.at("kernels")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10. outro
    def outro(self):
        self.section(10)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
