"""How LLMs Work: Deep Dive, episode 10 — FlashAttention: Same Math, Less Memory.

Render from the repo root:  ./render.sh deep-dive d10
Every number on screen comes from code/d10_flash_attention/flash_attention.py (online softmax on 8 scores; a 20-line
tiled attention vs standard attention; standard attention vs PyTorch's fused kernel, 12 heads, CPU, torch 2.14.0).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d10_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 10"
SCORES = [1.0, 3.0, 0.5, 2.0, 4.0, 1.5, 0.0, 2.5]
CODE = """for j in range(0, i + B, B):                   # key blocks, skipping the future
    s = qi @ k[j:j + B].T / math.sqrt(d)       # one B x B tile of scores
    m_new = torch.maximum(m, s.amax(-1))       # new running max
    fix = torch.exp(m - m_new)                 # rescale what we had
    p = torch.exp(s - m_new[:, None])
    l = l * fix + p.sum(-1)                    # running sum
    acc = acc * fix[:, None] + p @ v[j:j + B]  # running output
    m = m_new
out = acc / l[:, None]

# the fused version:  F.scaled_dot_product_attention(q, k, v, is_causal=True)"""


class FlashAttentionVideo(VoicedScene):
    VIDEO = "d10"

    def construct(self):
        play_token_intro(self, TITLE, 10, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.problem()       # 2
        self.online()        # 3
        self.tiling()        # 4
        self.same()          # 5
        self.measured()      # 6
        self.why_fast()      # 7
        self.unchanged()     # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        n = 16
        cells = VGroup(*[Square(0.27, stroke_width=0.5, stroke_color=GREY_D, fill_color=BLUE_E if j <= i else BLACK,
                                fill_opacity=0.9).move_to([(j - n / 2) * 0.27, (n / 2 - i) * 0.27, 0])
                         for i in range(n) for j in range(n)])
        cells.move_to([-3.4, -0.2, 0])
        lab = Text("every token × every token", font_size=22, color=GREY_B).next_to(cells, UP, buff=0.2)
        self.at("matrix")
        self.play(FadeIn(cells, lag_ratio=0.002), FadeIn(lab), run_time=1.0)
        a = Text("4,096 tokens, 12 heads:\n1.5 GiB for one layer (measured)", font_size=24, line_spacing=0.9)
        a.move_to([2.8, 1.2, 0])
        self.at("one")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("131,072 tokens:\n768 GiB per layer", font_size=24, color=RED_B, line_spacing=0.9).move_to([2.8, -0.3, 0])
        self.at("131")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("FlashAttention: same result,\nnever stores the matrix", font_size=26, color=YELLOW,
                 line_spacing=0.9).move_to([2.8, -2.0, 0])
        self.at("flash")
        self.play(FadeIn(c), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. problem
    def problem(self):
        self.section(2)
        self.clear_stage()
        row = VGroup(*[VGroup(Square(0.8, color=BLUE_C, fill_opacity=0.25), Text(f"{s:g}", font=MONO, font_size=24))
                       for s in SCORES]).arrange(RIGHT, buff=0.08)
        for r in row:
            r[1].move_to(r[0])
        row.move_to([0, 1.0, 0])
        self.at("softmax")
        self.play(FadeIn(row, lag_ratio=0.05), run_time=0.6)
        need = Text("softmax needs the max and the total of the whole row", font_size=26).move_to([0, -0.4, 0])
        self.at("largest")
        self.play(FadeIn(need), run_time=0.4)
        brace = Brace(row, DOWN, buff=0.15)
        self.play(GrowFromCenter(brace), run_time=0.4)
        keep = Text("…so keep the whole row?", font_size=28, color=RED_B).move_to([0, -1.7, 0])
        self.at("keep")
        self.play(FadeIn(keep), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. online softmax
    def online(self):
        self.section(3)
        self.clear_stage()
        head = Text("online softmax", font_size=34).to_edge(UP, buff=0.5)
        self.at("online")
        self.play(FadeIn(head), run_time=0.4)
        row = VGroup(*[VGroup(Square(0.75, color=BLUE_C, fill_opacity=0.25), Text(f"{s:g}", font=MONO, font_size=22))
                       for s in SCORES]).arrange(RIGHT, buff=0.06)
        for r in row:
            r[1].move_to(r[0])
        row.move_to([0, 1.7, 0])
        vals = Text("values: 0, 10, 20, … 70", font=MONO, font_size=20, color=GREY_B).next_to(row, DOWN, buff=0.2)
        div = DashedLine(row.get_top() + 0.15 * UP, row.get_bottom() + 0.15 * DOWN, color=YELLOW).move_to(
            (row[3].get_right() + row[4].get_left()) / 2)
        self.at("chunks")
        self.play(FadeIn(row), FadeIn(vals), Create(div), run_time=0.6)
        names = ["running max", "running sum", "output so far"]
        boxes = VGroup()
        for k, nm in enumerate(names):
            b = RoundedRectangle(width=3.4, height=1.1, corner_radius=0.15, color=GREY_B)
            t = Text(nm, font_size=20, color=GREY_B).next_to(b, UP, buff=0.1)
            boxes.add(VGroup(b, t))
        boxes.arrange(RIGHT, buff=0.5).move_to([0, -0.6, 0])
        self.at("three")
        self.play(FadeIn(boxes), run_time=0.5)
        nums = [None, None, None]

        def set_nums(values, colour):
            anims = []
            for k, v in enumerate(values):
                t = Text(v, font=MONO, font_size=28, color=colour).move_to(boxes[k][0])
                anims.append(FadeIn(t) if nums[k] is None else Transform(nums[k], t))
                if nums[k] is None:
                    nums[k] = t
            return anims

        hl1 = SurroundingRectangle(VGroup(*row[:4]), color=YELLOW, buff=0.06)
        self.at("first")
        self.play(Create(hl1), *set_nums(["3.0", "1.5853", "14.3052"], WHITE), run_time=0.6)
        hl2 = SurroundingRectangle(VGroup(*row[4:]), color=YELLOW, buff=0.06)
        self.at("four")
        self.play(ReplacementTransform(hl1, hl2), Indicate(row[4], color=YELLOW), run_time=0.5)
        fix = Text("rescale old sum and output by e^(3 − 4)", font=MONO, font_size=22, color=YELLOW).move_to([0, -1.9, 0])
        self.at("rescale")
        self.play(FadeIn(fix), run_time=0.4)
        self.at("add")
        self.play(*set_nums(["4.0", "1.9067", "36.2742"], GREEN_B), run_time=0.6)
        res = Text("full softmax: 36.2742  ✓", font=MONO, font_size=26, color=GREEN_B).move_to([0, -2.8, 0])
        self.at("result")
        self.play(FadeIn(res), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. tiling
    def tiling(self):
        self.section(4)
        self.clear_stage()
        n, s = 8, 0.62
        tiles = {}
        grid = VGroup()
        for i in range(n):
            for j in range(n):
                t = Square(s, stroke_width=1.5, stroke_color=GREY_B, fill_color=BLACK, fill_opacity=0.9)
                t.move_to([(j - (n - 1) / 2) * s, ((n - 1) / 2 - i) * s, 0])
                tiles[i, j] = t
                grid.add(t)
        grid.move_to([-2.8, -0.3, 0])
        ql = Text("query\nblocks\nof 64 ↓", font_size=20, color=GREY_B, line_spacing=0.8).next_to(grid, LEFT, buff=0.2)
        kl = Text("key blocks of 64 →", font_size=20, color=GREY_B).next_to(grid, UP, buff=0.15)
        self.at("tiles")
        self.play(FadeIn(grid), FadeIn(ql), FadeIn(kl), run_time=0.6)
        row_i = 5
        self.at("block")
        for j in range(row_i + 1):
            col = GOLD if j == row_i else GREEN_C
            self.play(tiles[row_i, j].animate.set_fill(col, 0.8), run_time=0.22)
        upd = Text("each tile updates its rows'\nrunning max, sum, output", font_size=22, line_spacing=0.85)
        upd.move_to([3.4, 1.6, 0])
        self.at("update")
        self.play(FadeIn(upd), run_time=0.4)
        future = [tiles[i, j] for i in range(n) for j in range(n) if j > i]
        diag = [tiles[i, i] for i in range(n)]
        lower = [tiles[i, j] for i in range(n) for j in range(n) if j < i]
        sk = Text("future blocks: skipped", font_size=22, color=GREY_B).move_to([3.4, 0.2, 0])
        self.at("skipped")
        self.play(*[t.animate.set_fill(GREY_E, 0.4) for t in future], FadeIn(sk), run_time=0.5)
        dg = Text("diagonal blocks: masked", font_size=22, color=GOLD).move_to([3.4, -0.5, 0])
        self.at("diagonal")
        self.play(*[t.animate.set_fill(GOLD, 0.8) for t in diag], *[t.animate.set_fill(GREEN_C, 0.6) for t in lower],
                  FadeIn(dg), run_time=0.5)
        big = Text("largest piece stored:\n64 × 64 = 16 KiB\ninstead of 4 MiB", font=MONO, font_size=22, color=YELLOW,
                   line_spacing=0.85).move_to([3.4, -2.0, 0])
        self.at("biggest")
        self.play(FadeIn(big), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5. same math
    def same(self):
        self.section(5)
        self.clear_stage()
        a = Text("not an approximation", font_size=36).move_to([0, 1.0, 0])
        self.at("approximation")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("20-line tiled version vs standard:\nlargest difference 4.8 × 10⁻⁷ (rounding)", font=MONO, font_size=26,
                 color=GREEN_B, line_spacing=0.9).move_to([0, -0.5, 0])
        self.at("20")
        self.play(FadeIn(b), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 6. measured
    def measured(self):
        self.section(6)
        self.clear_stage()
        head = Text("12 heads, extra memory for attention (measured)", font_size=28).to_edge(UP, buff=0.5)
        self.at("12")
        self.play(FadeIn(head), run_time=0.4)
        data = [("1,024 tokens", 99, 8), ("2,048 tokens", 386, 4), ("4,096 tokens", 1557, 11)]
        scale = 8.0 / 1557
        rows = VGroup()
        for k, (name, st, fu) in enumerate(data):
            y = 1.6 - 1.5 * k
            lab = Text(name, font_size=22).move_to([-6.2, y + 0.35, 0], aligned_edge=LEFT)
            b1 = Rectangle(width=st * scale, height=0.32, stroke_width=0, fill_color=RED_C, fill_opacity=0.85)
            b2 = Rectangle(width=max(0.06, fu * scale), height=0.32, stroke_width=0, fill_color=GREEN_C, fill_opacity=0.9)
            b1.move_to([-6.2, y - 0.05, 0], aligned_edge=LEFT)
            b2.move_to([-6.2, y - 0.45, 0], aligned_edge=LEFT)
            t1 = Text(f"standard {st:,} MiB", font=MONO, font_size=18).next_to(b1, RIGHT, buff=0.15)
            t2 = Text(f"fused {fu} MiB", font=MONO, font_size=18, color=GREEN_B).next_to(b2, RIGHT, buff=0.15)
            rows.add(VGroup(lab, b1, t1, b2, t2))
        self.at("1000")
        self.play(FadeIn(rows[0]), run_time=0.5)
        self.play(FadeIn(rows[1]), run_time=0.4)
        self.at("4000")
        self.play(FadeIn(rows[2]), run_time=0.5)
        sp = Text("time at 4,096: 0.74 s → 0.06 s  (12× faster)", font=MONO, font_size=24, color=YELLOW)
        sp.move_to([0, -2.9, 0])
        self.at("faster")
        self.play(FadeIn(sp), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. why faster
    def why_fast(self):
        self.section(7)
        self.clear_stage()
        chip = RoundedRectangle(width=4.2, height=3.0, corner_radius=0.2, color=MODEL_COLOR).move_to([-3.2, 0.2, 0])
        cl = Text("GPU chip", font_size=22, color=GREY_B).next_to(chip, UP, buff=0.1)
        sram = RoundedRectangle(width=3.2, height=1.1, corner_radius=0.1, color=GREEN_C, fill_opacity=0.3)
        sram.move_to(chip).shift(0.5 * DOWN)
        sl = Text("fast on-chip memory\n(small)", font_size=17, line_spacing=0.8).move_to(sram)
        alu = Text("compute", font_size=22).move_to(chip).shift(0.8 * UP)
        hbm = RoundedRectangle(width=3.6, height=3.0, corner_radius=0.2, color=RED_C, fill_opacity=0.2).move_to([3.4, 0.2, 0])
        hl = Text("main memory\n(large, slower)", font_size=22, line_spacing=0.8).move_to(hbm)
        self.at("gpu")
        self.play(FadeIn(chip), FadeIn(cl), FadeIn(alu), FadeIn(hbm), FadeIn(hl), run_time=0.6)
        slow = Text("the slow part: moving data", font_size=24, color=YELLOW).to_edge(DOWN, buff=1.2)
        self.at("moving")
        self.play(FadeIn(slow), run_time=0.4)
        self.at("tiles")
        self.play(FadeIn(sram), FadeIn(sl), run_time=0.5)
        arrows = VGroup(Arrow(chip.get_right(), hbm.get_left(), buff=0.1, color=RED_B).shift(0.3 * UP),
                        Arrow(hbm.get_left(), chip.get_right(), buff=0.1, color=RED_B).shift(0.3 * DOWN))
        self.at("travel")
        self.play(GrowArrow(arrows[0]), GrowArrow(arrows[1]), run_time=0.6)
        fewer = Text("fewer trips → faster attention", font_size=26, color=GREEN_B).to_edge(DOWN, buff=0.5)
        self.at("fewer")
        self.play(FadeIn(fewer), arrows.animate.set_opacity(0.25), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 8. unchanged
    def unchanged(self):
        self.section(8)
        self.clear_stage()
        a = Text("still compares every query with every earlier key", font_size=28).move_to([0, 1.4, 0])
        self.at("every")
        self.play(FadeIn(a), run_time=0.4)
        sq = VGroup()
        for k, (n, lab) in enumerate(((3, "length L"), (6, "length 2L: 4× the work"))):
            g = VGroup(*[Square(0.3, stroke_width=1, stroke_color=GREY_B, fill_color=BLUE_E if j <= i else BLACK,
                                fill_opacity=0.9).move_to([j * 0.3, -i * 0.3, 0]) for i in range(n) for j in range(n)])
            sq.add(VGroup(g, Text(lab, font_size=20, color=GREY_B).next_to(g, DOWN, buff=0.2)))
        sq.arrange(RIGHT, buff=1.5, aligned_edge=DOWN).move_to([0, -0.7, 0])
        self.at("square")
        self.play(FadeIn(sq[0]), run_time=0.3)
        self.play(FadeIn(sq[1]), run_time=0.4)
        skip = Text("to cut the work: skip some pairs", font_size=26, color=YELLOW).to_edge(DOWN, buff=0.6)
        self.at("skip")
        self.play(FadeIn(skip), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=19)
        code.move_to([0, 0.1, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("loop")
        self.play(Create(hl), run_time=0.3)
        self.at("max")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("rescale")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.at("add")
        self.play(highlight(hl, code, 6), run_time=0.3)
        self.at("calling")
        self.play(highlight(hl, code, 10), run_time=0.3)
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
