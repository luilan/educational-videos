"""How LLMs Work: Deep Dive, episode 13 — The Residual Stream.

Render from the repo root:  ./render.sh deep-dive d13
Every number on screen comes from code/d13_residual_stream/residual_stream.py (GPT-2 small, real weights, 512 tokens of
Tiny Shakespeare; and an 8-layer tiny GPT trained 1,500 steps with and without residual connections).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d13_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 13"
STREAM = [4.6, 49.5, 57.2, 61.1, 65.0, 69.8, 77.0, 85.5, 100.1, 119.5, 150.0, 216.1, 554.9]
ATT = [7.12, 0.22, 0.15, 0.15, 0.16, 0.19, 0.21, 0.26, 0.25, 0.28, 0.31, 1.41]
MLP = [5.27, 0.24, 0.24, 0.25, 0.24, 0.26, 0.27, 0.29, 0.31, 0.34, 0.53, 0.70]
DELETE = [7.95, 4.24, 3.98, 3.83, 4.33, 4.28, 3.74, 4.18, 4.03, 4.14, 3.94, 6.12]
STEPS = [100, 250, 500, 1000, 1500]
WITH = [2.46, 2.18, 1.96, 1.77, 1.69]
WITHOUT = [3.37, 3.36, 3.36, 3.36, 3.36]
STREAM_COL = BLUE_C
CODE = """def forward(self, x):
    x = x + self.attn(self.ln1(x))     # attention adds to the stream
    x = x + self.mlp(self.ln2(x))      # the MLP adds to the stream
    return x

# without the stream (the experiment):
#   x = self.attn(self.ln1(x));  x = self.mlp(self.ln2(x))"""


def bars(values, x0, width, base_y, scale, colour, clip=None, gap=0.12):
    out = VGroup()
    n = len(values)
    w = (width - gap * (n - 1)) / n
    for k, v in enumerate(values):
        h = min(v, clip) * scale if clip else v * scale
        b = Rectangle(width=w, height=max(0.03, h), stroke_width=0, fill_color=colour, fill_opacity=0.85)
        b.move_to([x0 + k * (w + gap) + w / 2, base_y + b.height / 2, 0])
        out.add(b)
    return out


class ResidualStreamVideo(VoicedScene):
    VIDEO = "d13"

    def construct(self):
        play_token_intro(self, TITLE, 13, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.picture()       # 2
        self.size()          # 3
        self.edits()         # 4
        self.delete()        # 5
        self.why()           # 6
        self.train()         # 7
        self.gradient()      # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        a = Text("x  ←  x + attention(x)", font=MONO, font_size=36).move_to([0, 1.2, 0])
        b = Text("x  ←  x + MLP(x)", font=MONO, font_size=36).next_to(a, DOWN, buff=0.4).align_to(a, LEFT)
        self.at("adds")
        self.play(FadeIn(a), run_time=0.5)
        self.at("mlp")
        self.play(FadeIn(b), run_time=0.5)
        plus = VGroup(a[3], b[3])
        name = Text("the residual stream", font_size=34, color=STREAM_COL).move_to([0, -1.2, 0])
        self.at("residual")
        self.play(FadeIn(name), Indicate(plus, color=YELLOW), run_time=0.6)
        ws = Text("the model's shared workspace", font_size=26, color=GREY_A).next_to(name, DOWN, buff=0.3)
        self.at("workspace")
        self.play(FadeIn(ws), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. picture
    def picture(self):
        self.section(2)
        self.clear_stage()
        stream = Arrow([-6.3, -0.6, 0], [6.3, -0.6, 0], buff=0, stroke_width=14, color=STREAM_COL,
                       max_tip_length_to_length_ratio=0.03)
        el = Text("embedding", font_size=20).next_to(stream.get_start(), DOWN, buff=0.25).shift(0.6 * RIGHT)
        ol = Text("output", font_size=20).next_to(stream.get_end(), DOWN, buff=0.25).shift(0.4 * LEFT)
        vl = Text("one vector per token: 768 numbers", font_size=24, color=STREAM_COL).move_to([0, -1.5, 0])
        self.at("vector")
        self.play(GrowArrow(stream), FadeIn(el), FadeIn(ol), run_time=0.8)
        self.at("768")
        self.play(FadeIn(vl), run_time=0.4)
        blocks = VGroup()
        for k in range(24):
            x = -5.6 + k * 0.49
            col = GOLD if k % 2 == 0 else GREEN_C
            blocks.add(Rectangle(width=0.36, height=0.7, stroke_width=1, color=col, fill_opacity=0.5).move_to([x, 1.0, 0]))
        leg = VGroup(Square(0.25, color=GOLD, fill_opacity=0.5), Text("attention", font_size=18),
                     Square(0.25, color=GREEN_C, fill_opacity=0.5), Text("MLP", font_size=18)).arrange(RIGHT, buff=0.15)
        leg.move_to([0, 2.2, 0])
        self.at("24")
        self.play(FadeIn(blocks, lag_ratio=0.03), FadeIn(leg), run_time=0.8)
        k = 9
        rd = Arrow(stream.get_center() * 0 + [blocks[k].get_x() - 0.08, -0.45, 0], blocks[k].get_bottom() + 0.08 * LEFT,
                   buff=0.02, stroke_width=3, color=GREY_A, max_tip_length_to_length_ratio=0.25)
        wr = Arrow(blocks[k].get_bottom() + 0.08 * RIGHT, [blocks[k].get_x() + 0.08, -0.45, 0], buff=0.02,
                   stroke_width=3, color=YELLOW, max_tip_length_to_length_ratio=0.25)
        rl = Text("reads", font_size=18, color=GREY_A).next_to(rd, LEFT, buff=0.05)
        wl = Text("writes (+)", font_size=18, color=YELLOW).next_to(wr, RIGHT, buff=0.05)
        self.at("reads")
        self.play(GrowArrow(rd), FadeIn(rl), blocks[k].animate.set_fill(opacity=0.9), run_time=0.4)
        self.at("writes")
        self.play(GrowArrow(wr), FadeIn(wl), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. size
    def size(self):
        self.section(3)
        self.clear_stage()
        head = Text("size of the stream entering each layer (GPT-2, average token)", font_size=24).to_edge(UP, buff=0.4)
        self.at("measure")
        self.play(FadeIn(head), run_time=0.4)
        b = bars(STREAM, -5.6, 10.6, -2.6, 4.6 / 555, STREAM_COL)
        labels = VGroup(*[Text(str(k) if k < 12 else "end", font_size=16, color=GREY_B).next_to(b[k], DOWN, buff=0.1)
                          for k in range(13)])
        vals = VGroup(*[Text(f"{v:g}", font=MONO, font_size=15).next_to(b[k], UP, buff=0.06) for k, v in enumerate(STREAM)])
        self.play(FadeIn(labels), run_time=0.3)
        self.at("4")
        self.play(FadeIn(b[0]), FadeIn(vals[0]), run_time=0.3)
        self.at("50")
        self.play(FadeIn(b[1]), FadeIn(vals[1]), run_time=0.3)
        self.at("216")
        self.play(LaggedStart(*[AnimationGroup(FadeIn(b[k]), FadeIn(vals[k])) for k in range(2, 13)], lag_ratio=0.1),
                  run_time=1.0)
        first = Text("first token: past 3,000\n(3,067 entering layer 11)", font_size=22, color=GOLD,
                     line_spacing=0.85).move_to([-2.8, 1.4, 0])
        self.at("different")
        self.play(FadeIn(first), run_time=0.4)
        sink = Text("the attention sink", font_size=24, color=GOLD).next_to(first, DOWN, buff=0.25)
        self.at("sink")
        self.play(FadeIn(sink), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. edits
    def edits(self):
        self.section(4)
        self.clear_stage()
        head = Text("each update ÷ the stream it is added to", font_size=26).to_edge(UP, buff=0.4)
        self.at("compare")
        self.play(FadeIn(head), run_time=0.4)
        scale, clip = 3.0, 1.15
        x0, gap = -5.6, 0.92
        ba, bm = VGroup(), VGroup()
        for k in range(12):
            for vals, grp, col, dx in ((ATT, ba, GOLD, 0.0), (MLP, bm, GREEN_C, 0.36)):
                v = vals[k]
                r = Rectangle(width=0.32, height=min(v, clip) * scale, stroke_width=0, fill_color=col, fill_opacity=0.85)
                r.move_to([x0 + k * gap + dx, -2.4 + r.height / 2, 0])
                grp.add(r)
        labels = VGroup(*[Text(str(k), font_size=16, color=GREY_B).move_to([x0 + k * gap + 0.18, -2.65, 0])
                          for k in range(12)])
        leg = VGroup(Square(0.22, color=GOLD, fill_opacity=0.85), Text("attention", font_size=18),
                     Square(0.22, color=GREEN_C, fill_opacity=0.85), Text("MLP", font_size=18)).arrange(RIGHT, buff=0.15)
        leg.move_to([4.6, 2.5, 0])
        one = DashedLine([x0 - 0.3, -2.4 + scale, 0], [x0 + 11 * gap + 0.6, -2.4 + scale, 0], color=GREY_B)
        ol = Text("100%", font_size=16, color=GREY_B).next_to(one, LEFT, buff=0.08)
        self.play(FadeIn(ba), FadeIn(bm), FadeIn(labels), FadeIn(leg), Create(one), FadeIn(ol), run_time=0.8)
        band = SurroundingRectangle(VGroup(*ba[1:10], *bm[1:10]), color=YELLOW, buff=0.06)
        bl = Text("layers 1–9: 15% to 34%", font_size=22, color=YELLOW).next_to(band, UP, buff=0.12)
        self.at("15")
        self.play(Create(band), FadeIn(bl), run_time=0.5)
        ed = Text("edits, not rewrites", font_size=30, color=YELLOW).move_to([0, 1.7, 0])
        self.at("edits")
        self.play(FadeIn(ed), run_time=0.4)
        l0 = Text("layer 0: 7.1× / 5.3× (off the chart)", font=MONO, font_size=18, color=GREY_A)
        l0.next_to(ba[0], UP, buff=0.1).align_to(ba[0], LEFT)
        self.at("first")
        self.play(FadeIn(l0), run_time=0.3)
        self.at("last")
        self.play(Indicate(VGroup(ba[11], bm[11], bm[10]), color=WHITE), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5. delete
    def delete(self):
        self.section(5)
        self.clear_stage()
        head = Text("delete one whole layer: GPT-2's loss", font_size=26).to_edge(UP, buff=0.4)
        self.at("delete")
        self.play(FadeIn(head), run_time=0.4)
        scale = 0.55
        b = bars(DELETE, -5.6, 9.8, -2.6, scale, GREY_B)
        for k in (0, 11):
            b[k].set_fill(RED_C)
        labels = VGroup(*[Text(str(k), font_size=16, color=GREY_B).next_to(b[k], DOWN, buff=0.1) for k in range(12)])
        vals = VGroup(*[Text(f"{v:.2f}", font=MONO, font_size=15).next_to(b[k], UP, buff=0.06)
                        for k, v in enumerate(DELETE)])
        base = DashedLine([-5.8, -2.6 + 4.13 * scale, 0], [4.4, -2.6 + 4.13 * scale, 0], color=YELLOW)
        bl = Text("all layers: 4.13", font_size=20, color=YELLOW).next_to(base, RIGHT, buff=0.1)
        self.at("13")
        self.play(Create(base), FadeIn(bl), FadeIn(labels), run_time=0.5)
        self.at("middle")
        self.play(LaggedStart(*[AnimationGroup(FadeIn(b[k]), FadeIn(vals[k])) for k in range(1, 11)], lag_ratio=0.08),
                  run_time=1.0)
        help_ = Text("some even help, on this text", font_size=20, color=GREY_A).move_to([0, 1.6, 0])
        self.at("help")
        self.play(FadeIn(help_), run_time=0.3)
        self.at("first")
        self.play(FadeIn(b[0]), FadeIn(vals[0]), run_time=0.4)
        self.at("last")
        self.play(FadeIn(b[11]), FadeIn(vals[11]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. why
    def why(self):
        self.section(6)
        self.clear_stage()
        stream = Arrow([-6.0, 1.2, 0], [6.0, 1.2, 0], buff=0, stroke_width=12, color=STREAM_COL,
                       max_tip_length_to_length_ratio=0.03)
        blocks = VGroup(*[Rectangle(width=0.8, height=0.6, stroke_width=1, color=GREEN_C, fill_opacity=0.5)
                          .move_to([-4.5 + 1.5 * k, 2.2, 0]) for k in range(7)])
        tl = Text("residual: each layer adds an edit", font_size=22, color=STREAM_COL).move_to([0, 0.5, 0])
        self.at("adds")
        self.play(GrowArrow(stream), FadeIn(blocks), FadeIn(tl), run_time=0.6)
        cross = Cross(blocks[3], stroke_color=RED_C)
        self.at("deleting")
        self.play(blocks[3].animate.set_opacity(0.15), Create(cross), run_time=0.4)
        fl = Text("the stream still flows", font_size=22, color=YELLOW).next_to(stream, UP, buff=0.9).shift(0.2 * DOWN)
        self.at("flows")
        self.play(Indicate(stream, color=YELLOW), run_time=0.6)
        chain = VGroup(*[Rectangle(width=0.8, height=0.6, stroke_width=1, color=GREY_B, fill_opacity=0.4) for _ in range(7)])
        chain.arrange(RIGHT, buff=0.45).move_to([0, -1.5, 0])
        links = VGroup(*[Arrow(chain[k].get_right(), chain[k + 1].get_left(), buff=0.05, stroke_width=3)
                         for k in range(6)])
        cl = Text("plain chain: every layer needs the exact output of the one before", font_size=22,
                  color=GREY_A).next_to(chain, DOWN, buff=0.3)
        self.at("chain")
        self.play(FadeIn(chain), FadeIn(links), FadeIn(cl), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 7. train
    def train(self):
        self.section(7)
        self.clear_stage()
        head = Text("tiny GPT, 8 layers: validation loss during training", font_size=26).to_edge(UP, buff=0.4)
        self.at("train")
        self.play(FadeIn(head), run_time=0.4)
        ax = Axes(x_range=[0, 1500, 500], y_range=[1.5, 3.6, 0.5], x_length=8.0, y_length=4.2, tips=False,
                  axis_config={"color": GREY_B}).move_to([-1.0, -0.5, 0])
        xl = VGroup(*[Text(str(v), font_size=16, color=GREY_B).next_to(ax.c2p(v, 1.5), DOWN, buff=0.1)
                      for v in (0, 500, 1000, 1500)])
        yl = VGroup(*[Text(f"{v:.1f}", font_size=16, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.1)
                      for v in (1.5, 2.0, 2.5, 3.0, 3.5)])
        st = Text("steps", font_size=18, color=GREY_B).next_to(xl, DOWN, buff=0.1)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(st), run_time=0.5)
        cw = VMobject(color=GREEN_C, stroke_width=5).set_points_as_corners([ax.c2p(s, v) for s, v in zip(STEPS, WITH)])
        co = VMobject(color=RED_C, stroke_width=5).set_points_as_corners([ax.c2p(s, v) for s, v in zip(STEPS, WITHOUT)])
        lw = Text("x = x + block(x): 2.46 → 1.69", font_size=20, color=GREEN_B).move_to([4.6, 0.0, 0])
        lo = Text("x = block(x): stuck at 3.36", font_size=20, color=RED_B).move_to([4.6, 1.4, 0])
        self.at("residual")
        self.play(Create(cw), FadeIn(lw), run_time=0.8)
        self.at("stuck")
        self.play(Create(co), FadeIn(lo), run_time=0.6)
        fq = DashedLine(ax.c2p(0, 3.35), ax.c2p(1500, 3.35), color=GREY_B)
        fl = Text("letter frequencies only: 3.35", font_size=18, color=GREY_B).move_to([4.6, 0.9, 0])
        self.at("letter")
        self.play(Create(fq), FadeIn(fl), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 8. gradient
    def gradient(self):
        self.section(8)
        self.clear_stage()
        eq = Text("d/dx [ x + f(x) ]  =  1  +  f′(x)", font=MONO, font_size=36).move_to([0, 1.5, 0])
        self.at("derivative")
        self.play(FadeIn(eq), run_time=0.5)
        one = SurroundingRectangle(eq[13:14] if len(eq) > 14 else eq, color=YELLOW, buff=0.1)
        self.at("one")
        self.play(Create(one), run_time=0.4)
        blocks = VGroup(*[Rectangle(width=0.9, height=0.6, stroke_width=1, color=GREEN_C, fill_opacity=0.4)
                          for _ in range(8)]).arrange(RIGHT, buff=0.3).move_to([0, -0.6, 0])
        back = Arrow(blocks.get_right() + 0.2 * RIGHT + 0.7 * DOWN, blocks.get_left() + 0.2 * LEFT + 0.7 * DOWN,
                     buff=0, color=YELLOW, stroke_width=8)
        bl = Text("learning signal, straight back to the early layers", font_size=22, color=YELLOW)
        bl.next_to(back, DOWN, buff=0.2)
        self.at("signal")
        self.play(FadeIn(blocks), GrowArrow(back), FadeIn(bl), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("attention")
        self.play(Create(hl), run_time=0.3)
        self.at("mlp")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("removing")
        self.play(highlight(hl, code, 6), run_time=0.3)
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
