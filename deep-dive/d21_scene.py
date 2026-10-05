"""How LLMs Work: Deep Dive, episode 21 — Batch Size and Gradient Accumulation.

Render from the repo root:  ./render.sh deep-dive d21
Every number on screen comes from code/d21_batch_size/batch_size.py (a 4-layer tiny GPT: gradient accumulation check,
gradient similarity by batch size, and 48,000 training sequences at batch sizes 8, 32 and 128; CPU, 8 threads).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d21_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 21"
SIM = [(1, 0.17), (4, 0.19), (16, 0.50), (64, 0.72), (256, 0.90)]
CODE = """opt.zero_grad()
for k in range(4):                                  # 4 micro-batches of 8
    loss = lm_loss(model, x[8*k:8*k+8], y[8*k:8*k+8]) / 4
    loss.backward()                                 # gradients add up
opt.step()                                          # one step, as for a batch of 32"""


def hbars(rows, x0=-1.0, y0=1.4, dy=0.85, scale=4.0, fmt="{:.2f}", size=22):
    out = VGroup()
    for k, (name, v, col) in enumerate(rows):
        y = y0 - dy * k
        lab = Text(name, font_size=size, color=col).move_to([x0 - 0.3, y, 0], aligned_edge=RIGHT)
        b = Rectangle(width=max(0.05, v * scale), height=0.5, stroke_width=0, fill_color=col, fill_opacity=0.85)
        b.move_to([x0, y, 0], aligned_edge=LEFT)
        out.add(VGroup(lab, b, Text(fmt.format(v), font=MONO, font_size=size).next_to(b, RIGHT, buff=0.15)))
    return out


class BatchSizeVideo(VoicedScene):
    VIDEO = "d21"

    def construct(self):
        play_token_intro(self, TITLE, 21, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.noise()         # 2
        self.same_data()     # 3
        self.lr()            # 4
        self.speed()         # 5
        self.tradeoff()      # 6
        self.accumulate()    # 7
        self.check()         # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        seqs = VGroup(*[Rectangle(width=4.0, height=0.28, stroke_width=1, color=BLUE_C, fill_opacity=0.25)
                        for _ in range(8)]).arrange(DOWN, buff=0.08).move_to([-2.6, 0.2, 0])
        bl = Text("a batch of sequences", font_size=22, color=GREY_B).next_to(seqs, UP, buff=0.2)
        arr = Arrow(seqs.get_right(), seqs.get_right() + 2.0 * RIGHT, buff=0.2)
        g = Text("one averaged\ngradient", font_size=24, color=YELLOW, line_spacing=0.85).next_to(arr, RIGHT, buff=0.2)
        self.at("averages")
        self.play(FadeIn(seqs, lag_ratio=0.1), FadeIn(bl), GrowArrow(arr), FadeIn(g), run_time=0.8)
        q = Text("how big?   and if it doesn't fit in memory?", font_size=26).move_to([0, -2.4, 0])
        self.at("memory")
        self.play(FadeIn(q), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. noise
    def noise(self):
        self.section(2)
        self.clear_stage()
        head = Text("similarity (cosine) between a batch's gradient and the average over 4,096 sequences", font_size=22)
        head.to_edge(UP, buff=0.5)
        self.at("compare")
        self.play(FadeIn(head), run_time=0.4)
        rows = hbars([(f"batch {b}", s, interpolate_color(RED_C, GREEN_C, s)) for b, s in SIM], scale=6.0, y0=1.6)
        cues = ("17", "17", "5", "72", "9")
        for k in range(5):
            if k == 1:
                self.play(FadeIn(rows[1]), run_time=0.25)
                continue
            self.at(cues[k])
            self.play(FadeIn(rows[k]), run_time=0.3)
        d = Text("bigger batches point more reliably downhill", font_size=24, color=YELLOW).move_to([0, -2.6, 0])
        self.at("downhill")
        self.play(FadeIn(d), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. same data
    def same_data(self):
        self.section(3)
        self.clear_stage()
        head = Text("48,000 training sequences in every run", font_size=28).to_edge(UP, buff=0.5)
        self.at("48")
        self.play(FadeIn(head), run_time=0.4)
        xs = [-4.0, -1.4, 1.4, 4.2]
        hdr = VGroup(*[Text(t, font_size=22, color=GREY_B).move_to([x, 1.8, 0]) for t, x in
                       zip(("batch", "learning rate", "steps", "val loss"), xs)])
        self.play(FadeIn(hdr), run_time=0.3)
        data = [("8", "0.001", "6,000", "1.660"), ("32", "0.001", "1,500", "1.709"), ("128", "0.001", "375", "1.930"),
                ("128", "0.002", "375", "1.811")]
        self.rows = VGroup()
        for k, r in enumerate(data):
            self.rows.add(VGroup(*[Text(v, font=MONO, font_size=26).move_to([x, 1.0 - 0.75 * k, 0]) for v, x in zip(r, xs)]))
        for k, cue in enumerate(("66", "71", "93")):
            self.at(cue)
            self.play(FadeIn(self.rows[k]), run_time=0.35)
        self.end_section()

    # ------------------------------------------------------------------ 4. learning rate
    def lr(self):
        self.section(4)
        self.at("81")
        self.play(FadeIn(self.rows[3]), run_time=0.4)
        w = Text("fixed data: more, noisier steps won here", font_size=26, color=YELLOW).move_to([0, -2.4, 0])
        self.at("won")
        self.play(FadeIn(w), Indicate(self.rows[0][3], color=GREEN_B), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 5. speed
    def speed(self):
        self.section(5)
        self.clear_stage()
        head = Text("time per training sequence, this CPU", font_size=28).to_edge(UP, buff=0.6)
        self.at("speed")
        self.play(FadeIn(head), run_time=0.4)
        rows = hbars([("batch 8", 2.82, RED_C), ("batch 32", 1.56, GREEN_C), ("batch 128", 1.68, GREEN_C)],
                     scale=1.6, y0=1.2, dy=1.0, fmt="{:.2f} ms")
        self.at("82")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("56")
        self.play(FadeIn(rows[1]), FadeIn(rows[2]), run_time=0.4)
        s = Text("saturated at 32 here; GPUs keep gaining up to much larger batches", font_size=22, color=GREY_A)
        s.move_to([0, -2.2, 0])
        self.at("saturated")
        self.play(FadeIn(s), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. trade-off
    def tradeoff(self):
        self.section(6)
        self.clear_stage()
        a = VGroup(Text("small batches", font_size=30, color=BLUE_C), Text("use the data best", font_size=24))
        b = VGroup(Text("big batches", font_size=30, color=GOLD), Text("use the hardware best", font_size=24))
        for g, x in ((a, -3.2), (b, 3.2)):
            g.arrange(DOWN, buff=0.3).move_to([x, 0.6, 0])
        self.at("data")
        self.play(FadeIn(a), run_time=0.4)
        self.at("hardware")
        self.play(FadeIn(b), run_time=0.4)
        m = Text("LLMs: batches of millions of tokens, learning rates tuned to match", font_size=24, color=YELLOW)
        m.move_to([0, -1.8, 0])
        self.at("millions")
        self.play(FadeIn(m), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. accumulation
    def accumulate(self):
        self.section(7)
        self.clear_stage()
        head = Text("gradient accumulation", font_size=32).to_edge(UP, buff=0.6)
        self.at("accumulation")
        self.play(FadeIn(head), run_time=0.4)
        micro = VGroup()
        for k in range(4):
            box = VGroup(*[Rectangle(width=1.4, height=0.16, stroke_width=0.8, color=BLUE_C, fill_opacity=0.3)
                           for _ in range(8)]).arrange(DOWN, buff=0.03)
            lab = Text(f"loss {k + 1} ÷ 4", font=MONO, font_size=18).next_to(box, DOWN, buff=0.15)
            micro.add(VGroup(box, lab))
        micro.arrange(RIGHT, buff=0.5).move_to([-1.8, 0.6, 0])
        bucket = RoundedRectangle(width=2.4, height=1.2, corner_radius=0.15, color=YELLOW).move_to([4.2, 0.6, 0])
        bl = Text(".grad\n(adds up)", font_size=22, color=YELLOW, line_spacing=0.85).move_to(bucket)
        self.at("micro")
        self.play(FadeIn(micro, lag_ratio=0.2), FadeIn(bucket), FadeIn(bl), run_time=0.8)
        self.at("divide")
        self.play(*[Indicate(m[1]) for m in micro], run_time=0.5)
        arrows = VGroup(*[CurvedArrow(m[0].get_top() + 0.05 * UP, bucket.get_top() + 0.05 * UP, angle=-0.9,
                                      stroke_width=2, color=GREY_B) for m in micro])
        self.at("backward")
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.2), run_time=0.8)
        st = Text("then one optimizer step", font_size=24, color=GREEN_B).next_to(bucket, DOWN, buff=0.4)
        self.at("add")
        self.play(FadeIn(st), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. check
    def check(self):
        self.section(8)
        self.clear_stage()
        a = Text("4 micro-batches of 8 vs one batch of 32:\nlargest gradient difference 1.5 × 10⁻⁸", font=MONO,
                 font_size=26, color=GREEN_B, line_spacing=0.9).move_to([0, 0.9, 0])
        self.at("four")
        self.play(FadeIn(a), run_time=0.5)
        b = Text("the memory of 8, the gradient of 32", font_size=28, color=YELLOW).move_to([0, -0.6, 0])
        self.at("memory")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("(but no speed-up)", font_size=24, color=GREY_A).move_to([0, -1.4, 0])
        self.at("speedup")
        self.play(FadeIn(c), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.at("encode", "code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("divided")
        self.play(Create(hl), run_time=0.3)
        self.at("optimizer")
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
