"""How LLMs Work: Deep Dive, episode 16 — Pre-Norm, Post-Norm, and Stability.

Render from the repo root:  ./render.sh deep-dive d16
Every number on screen comes from code/d16_pre_post_norm/pre_post_norm.py (a 12-layer tiny GPT trained 1,500 steps,
pre-norm vs post-norm, learning rates 0.001 and 0.003, with and without 300 warmup steps).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d16_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 16"
PRE_C, POST_C = GREEN_C, PURPLE_B
STEPS = [1, 5, 10, 20, 50, 100]
EARLY_PRE = [4.40, 3.39, 3.11, 2.88, 2.57, 2.40]
EARLY_POST = [4.38, 3.42, 3.38, 3.35, 3.33, 3.28]
CODE = """# pre-norm (GPT-2, Llama, Qwen): normalize what the block reads
x = x + attn(ln1(x))
x = x + mlp(ln2(x))

# post-norm (original Transformer, BERT): normalize the stream itself
x = ln1(x + attn(x))
x = ln2(x + mlp(x))"""


def box(label, colour, w=1.3, h=0.6):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.1, color=colour, fill_opacity=0.35)
    return VGroup(r, Text(label, font_size=20).move_to(r))


class PrePostNormVideo(VoicedScene):
    VIDEO = "d16"

    def construct(self):
        play_token_intro(self, TITLE, 16, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.two()           # 2
        self.highway()       # 3
        self.low()           # 4
        self.high()          # 5
        self.warmup()        # 6
        self.where()         # 7
        self.why()           # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def bars(self, rows, y0=1.0, scale=1.6, x0=-1.6):
        out = VGroup()
        for k, (name, v, col) in enumerate(rows):
            y = y0 - 1.0 * k
            lab = Text(name, font_size=24, color=col).move_to([x0 - 0.3, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=v * scale, height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([x0, y, 0], aligned_edge=LEFT)
            num = Text(f"{v:.2f}", font=MONO, font_size=24).next_to(b, RIGHT, buff=0.15)
            out.add(VGroup(lab, b, num))
        return out

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        q = Text("where does the norm go?", font_size=36).move_to([0, 1.8, 0])
        self.play(FadeIn(q), run_time=0.4)
        a = Text("after each block (post-norm):\noriginal Transformer, BERT", font_size=26, color=POST_C,
                 line_spacing=0.85).move_to([-3.4, 0.0, 0])
        b = Text("before each block (pre-norm):\nGPT-2, Llama, Qwen", font_size=26, color=PRE_C,
                 line_spacing=0.85).move_to([3.4, 0.0, 0])
        self.at("after")
        self.play(FadeIn(a), run_time=0.4)
        self.at("before")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("it decides whether training survives a high learning rate", font_size=24, color=YELLOW)
        c.move_to([0, -2.0, 0])
        self.at("survives")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. the two
    def two(self):
        self.section(2)
        self.clear_stage()
        # post-norm, left
        ps = Line([-6.2, -1.5, 0], [-0.8, -1.5, 0], color=BLUE_D, stroke_width=8)
        pb = box("block", GREEN_E).move_to([-4.6, 0.3, 0])
        pp = Circle(radius=0.25, color=WHITE).move_to([-3.2, -1.5, 0])
        ppl = Text("+", font_size=26).move_to(pp)
        pn = box("norm", POST_C, w=1.1).move_to([-1.9, -1.5, 0])
        pa = VGroup(Arrow([-5.4, -1.45, 0], pb.get_bottom(), buff=0.05, stroke_width=3),
                    Arrow(pb.get_right(), pp.get_top(), buff=0.05, stroke_width=3))
        pt = Text("post-norm: x = norm(x + block(x))", font=MONO, font_size=20, color=POST_C).move_to([-3.5, 1.6, 0])
        # pre-norm, right
        qs = Line([0.8, -1.5, 0], [6.2, -1.5, 0], color=BLUE_D, stroke_width=8)
        qn = box("norm", PRE_C, w=1.1).move_to([2.0, -0.2, 0])
        qb = box("block", GREEN_E).move_to([3.6, 0.6, 0])
        qp = Circle(radius=0.25, color=WHITE).move_to([4.9, -1.5, 0])
        qpl = Text("+", font_size=26).move_to(qp)
        qa = VGroup(Arrow([1.4, -1.45, 0], qn.get_bottom(), buff=0.05, stroke_width=3),
                    Arrow(qn.get_right(), qb.get_left(), buff=0.05, stroke_width=3),
                    Arrow(qb.get_right(), qp.get_top(), buff=0.05, stroke_width=3))
        qt = Text("pre-norm: x = x + block(norm(x))", font=MONO, font_size=20, color=PRE_C).move_to([3.5, 1.6, 0])
        self.at("post")
        self.play(Create(ps), FadeIn(pb), FadeIn(pp), FadeIn(ppl), FadeIn(pn), FadeIn(pa), FadeIn(pt), run_time=0.8)
        self.at("prenorm", "pre")
        self.play(Create(qs), FadeIn(qn), FadeIn(qb), FadeIn(qp), FadeIn(qpl), FadeIn(qa), FadeIn(qt), run_time=0.8)
        fin = Text("+ one final norm before the output", font_size=20, color=PRE_C).move_to([3.5, -2.3, 0])
        self.at("final")
        self.play(FadeIn(fin), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. the highway
    def highway(self):
        self.section(3)
        self.clear_stage()
        pre = Arrow([-6, 1.2, 0], [6, 1.2, 0], buff=0, stroke_width=10, color=PRE_C, max_tip_length_to_length_ratio=0.03)
        pl = Text("pre-norm: a clean highway, only additions", font_size=24, color=PRE_C).next_to(pre, UP, buff=0.2)
        self.at("highway")
        self.play(GrowArrow(pre), FadeIn(pl), run_time=0.7)
        post = Arrow([-6, -1.2, 0], [6, -1.2, 0], buff=0, stroke_width=10, color=POST_C,
                     max_tip_length_to_length_ratio=0.03)
        norms = VGroup(*[Rectangle(width=0.22, height=0.55, color=WHITE, fill_color=POST_C, fill_opacity=0.9)
                         .move_to([-5.5 + 0.47 * k, -1.2, 0]) for k in range(24)])
        ql = Text("post-norm: interrupted by a norm 24 times (12 layers)", font_size=24, color=POST_C)
        ql.next_to(post, DOWN, buff=0.4)
        self.at("interrupted")
        self.play(GrowArrow(post), FadeIn(norms, lag_ratio=0.04), FadeIn(ql), run_time=0.9)
        self.end_section()

    # ------------------------------------------------------------------ 4. low learning rate
    def low(self):
        self.section(4)
        self.clear_stage()
        head = Text("12-layer tiny GPT, 1,500 steps, learning rate 0.001", font_size=26).to_edge(UP, buff=0.6)
        self.at("train")
        self.play(FadeIn(head), run_time=0.4)
        rows = self.bars([("pre-norm", 1.67, PRE_C), ("post-norm", 1.68, POST_C)])
        self.at("67")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("68")
        self.play(FadeIn(rows[1]), run_time=0.4)
        nd = Text("no difference", font_size=28, color=YELLOW).move_to([0, -1.8, 0])
        self.at("difference")
        self.play(FadeIn(nd), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5. high learning rate
    def high(self):
        self.section(5)
        self.clear_stage()
        head = Text("learning rate × 3 (0.003), no warmup", font_size=28).to_edge(UP, buff=0.6)
        self.at("triple")
        self.play(FadeIn(head), run_time=0.4)
        rows = self.bars([("pre-norm", 1.67, PRE_C), ("post-norm", 3.36, RED_C)])
        self.at("67")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("36")
        self.play(FadeIn(rows[1]), run_time=0.4)
        st = Text("stuck from step 100 to 1,500", font_size=22, color=RED_B).next_to(rows[1], DOWN, buff=0.3)
        self.play(FadeIn(st), run_time=0.3)
        lf = Text("= the letter-frequency loss (episode 13): it learned nothing else", font_size=24, color=GREY_A)
        lf.move_to([0, -2.6, 0])
        self.at("letter")
        self.play(FadeIn(lf), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. warmup
    def warmup(self):
        self.section(6)
        self.clear_stage()
        ax = Axes(x_range=[0, 600, 300], y_range=[0, 1.2, 1], x_length=4.0, y_length=2.0, tips=False,
                  axis_config={"color": GREY_B}).move_to([-3.8, 1.2, 0])
        ramp = VMobject(color=YELLOW, stroke_width=5).set_points_as_corners(
            [ax.c2p(0, 0.02), ax.c2p(300, 1.0), ax.c2p(600, 1.0)])
        rl = Text("learning rate: 0 → full over 300 steps", font_size=20, color=YELLOW).next_to(ax, DOWN, buff=0.2)
        self.at("warm")
        self.play(Create(ax), Create(ramp), FadeIn(rl), run_time=0.8)
        rows = self.bars([("post-norm", 1.72, POST_C), ("pre-norm", 1.66, PRE_C)], y0=1.6, scale=1.2, x0=2.6)
        head = Text("lr 0.003 with warmup", font_size=22, color=GREY_A).next_to(rows, UP, buff=0.3)
        self.at("72")
        self.play(FadeIn(head), FadeIn(rows[0]), run_time=0.4)
        self.at("66")
        self.play(FadeIn(rows[1]), run_time=0.4)
        fam = Text("post-norm transformers were famously hard to train without warmup", font_size=24, color=GREY_A)
        fam.move_to([0, -2.3, 0])
        self.at("famously")
        self.play(FadeIn(fam), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. where it goes wrong
    def where(self):
        self.section(7)
        self.clear_stage()
        init = Text("at initialization: similar gradients at every layer in both", font_size=24, color=GREY_A)
        init.to_edge(UP, buff=0.5)
        self.at("initialization")
        self.play(FadeIn(init), run_time=0.4)
        ax = Axes(x_range=[0, 100, 25], y_range=[2.2, 4.6, 0.5], x_length=6.4, y_length=4.0, tips=False,
                  axis_config={"color": GREY_B}).move_to([-2.6, -0.6, 0])
        xl = VGroup(*[Text(str(v), font_size=16, color=GREY_B).next_to(ax.c2p(v, 2.2), DOWN, buff=0.1)
                      for v in (0, 25, 50, 75, 100)])
        yl = VGroup(*[Text(f"{v:.1f}", font_size=16, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.1)
                      for v in (2.5, 3.0, 3.5, 4.0, 4.5)])
        cap = Text("training loss, first 100 steps, lr 0.003, no warmup", font_size=18, color=GREY_B).next_to(xl, DOWN, 0.1)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(cap), run_time=0.5)
        cpre = VMobject(color=PRE_C, stroke_width=5).set_points_as_corners([ax.c2p(s, v) for s, v in zip(STEPS, EARLY_PRE)])
        cpost = VMobject(color=POST_C, stroke_width=5).set_points_as_corners(
            [ax.c2p(s, v) for s, v in zip(STEPS, EARLY_POST)])
        lpost = Text("post-norm:\n3.38 at step 10,\nthen stalls", font_size=22, color=POST_C, line_spacing=0.85).move_to([3.9, 1.0, 0])
        lpre = Text("pre-norm:\nkeeps falling (2.40)", font_size=22, color=PRE_C, line_spacing=0.85).move_to([3.9, -0.6, 0])
        self.at("10")
        self.play(Create(cpost), FadeIn(lpost), run_time=0.8)
        self.at("falling")
        self.play(Create(cpre), FadeIn(lpre), run_time=0.8)
        wm = Text("warmup makes those\nfirst updates small", font_size=22, color=YELLOW, line_spacing=0.85).move_to([3.9, -2.1, 0])
        self.at("small")
        self.play(FadeIn(wm), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. why pre-norm won
    def why(self):
        self.section(8)
        self.clear_stage()
        items = VGroup(Text("modern models use pre-norm", font_size=32, color=PRE_C),
                       Text("✓ stable at higher learning rates", font_size=26),
                       Text("✓ less careful tuning", font_size=26),
                       Text("✓ even in very deep networks", font_size=26)).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        items.move_to([0, 0.3, 0])
        for k, cue in enumerate(("modern", "stably", "tuning", "deep")):
            self.at(cue)
            self.play(FadeIn(items[k]), run_time=0.3)
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
        self.at("pre")
        self.play(Create(hl), run_time=0.3)
        self.at("post")
        self.play(highlight(hl, code, 5), run_time=0.3)
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
