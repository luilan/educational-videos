"""How LLMs Work: Deep Dive, episode 17 — Cross-Entropy, Deeper.

Render from the repo root:  ./render.sh deep-dive d17
Every number on screen comes from code/d17_cross_entropy/cross_entropy.py (GPT-2 small, real weights, on one sentence
and on 1,024 tokens of Tiny Shakespeare; the gradient checked with autograd).
"""
import math

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d17_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 17"
TOKS = ["capital", "of", "France", "is", "Paris", ",", "and", "the", "capital", "of", "Italy", "is", "Rome", "."]
P = [0.0001, 0.1642, 0.0065, 0.1217, 0.0322, 0.4505, 0.1938, 0.1621, 0.2090, 0.6402, 0.0188, 0.7643, 0.5769, 0.7341]
CALIB = [(0.11, 0.14, 400), (0.29, 0.33, 217), (0.49, 0.46, 90), (0.70, 0.67, 72), (0.96, 0.67, 244)]
GRAD = [0.612, 0.098, -0.985, 0.231, 0.044]
CODE = """logp = logits.log_softmax(dim=-1)                  # (T, vocabulary)
loss = -logp[torch.arange(T), targets].mean()      # pick the right token, negate, average

# the same thing, built in:
loss = F.cross_entropy(logits, targets)"""


class CrossEntropyVideo(VoicedScene):
    VIDEO = "d17"

    def construct(self):
        play_token_intro(self, TITLE, 17, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.tokens()        # 2
        self.average()       # 3
        self.names()         # 4
        self.confident()     # 5
        self.calibration()   # 6
        self.top()           # 7
        self.gradient()      # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        n = Text("loss  1.69", font=MONO, font_size=64).move_to([0, 0.5, 0])
        self.at("loss")
        self.play(FadeIn(n, scale=1.2), run_time=0.5)
        p = Text("part 5: training", font_size=30, color=YELLOW).move_to([0, -1.0, 0])
        self.at("training")
        self.play(FadeIn(p), run_time=0.4)
        a = Text("let's take that number apart", font_size=26, color=GREY_A).move_to([0, -1.8, 0])
        self.at("apart")
        self.play(FadeIn(a), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. token by token
    def tokens(self):
        self.section(2)
        self.clear_stage()
        f = Text("loss of one token = −ln(probability of the right token)", font_size=28).to_edge(UP, buff=0.5)
        self.at("minus")
        self.play(FadeIn(f), run_time=0.5)
        cols = VGroup()
        x0, dx = -6.2, 0.92
        for k, (w, p) in enumerate(zip(TOKS, P)):
            loss = -math.log(p)
            bar = Rectangle(width=0.6, height=loss * 0.38, stroke_width=0,
                            fill_color=interpolate_color(GREEN_C, RED_C, min(1, loss / 6)), fill_opacity=0.85)
            bar.move_to([x0 + k * dx, -2.3 + bar.height / 2, 0])
            lab = Text(w, font_size=18).move_to([x0 + k * dx, -2.6, 0])
            pt = Text(f"{p:.0%}" if p >= 0.01 else f"{p:.2%}", font=MONO, font_size=14, color=GREY_A).next_to(bar, UP, 0.06)
            cols.add(VGroup(bar, lab, pt))
        cap = Text("bar height = loss", font_size=18, color=GREY_B).move_to([4.6, 2.2, 0])
        self.at("capital")
        self.play(LaggedStart(*[FadeIn(c, shift=0.1 * UP) for c in cols], lag_ratio=0.08), FadeIn(cap), run_time=1.6)
        pa = Text("Paris: 3% → loss 3.43", font=MONO, font_size=22, color=YELLOW).move_to([-2.5, 1.2, 0])
        self.at("paris")
        self.at("paris")
        self.play(Indicate(cols[4]), FadeIn(pa), run_time=0.5)
        ro = Text("Rome: 58% → loss 0.55", font=MONO, font_size=22, color=GREEN_B).move_to([2.8, 1.2, 0])
        self.at("rome")
        self.play(Indicate(cols[12]), FadeIn(ro), run_time=0.5)
        self.cols = cols
        self.end_section()

    # ------------------------------------------------------------------ 3. average
    def average(self):
        self.section(3)
        avg = Text("average over 14 tokens: 2.344", font=MONO, font_size=26, color=YELLOW).move_to([0, 0.4, 0])
        self.at("average")
        self.play(FadeIn(avg), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. perplexity and bits
    def names(self):
        self.section(4)
        self.clear_stage()
        a = Text("perplexity = e^loss = e^2.344 = 10.4", font=MONO, font_size=30).move_to([0, 1.6, 0])
        self.at("perplexity")
        self.play(FadeIn(a), run_time=0.5)
        b = Text("like choosing evenly among ~10 tokens", font_size=24, color=GREY_A).next_to(a, DOWN, buff=0.25)
        self.at("evenly")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("bits = loss ÷ ln 2 = 3.38 bits per token", font=MONO, font_size=30).move_to([0, -0.3, 0])
        self.at("bits")
        self.play(FadeIn(c), run_time=0.5)
        d = Text("Shakespeare (1,024 tokens): 1.87 bits per character", font_size=26, color=YELLOW).move_to([0, -1.6, 0])
        self.at("shakespeare")
        self.play(FadeIn(d), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. confident mistakes
    def confident(self):
        self.section(5)
        self.clear_stage()
        ax = Axes(x_range=[0, 1, 0.25], y_range=[0, 7.5, 1], x_length=7.5, y_length=4.5, tips=False,
                  axis_config={"color": GREY_B}).move_to([-1.4, -0.2, 0])
        xl = Text("probability of the right token", font_size=18, color=GREY_B).next_to(ax, DOWN, buff=0.15)
        yl = Text("loss", font_size=18, color=GREY_B).next_to(ax, LEFT, buff=0.15)
        curve = ax.plot(lambda p: -math.log(p), x_range=[0.0006, 1], color=RED_C, stroke_width=5)
        self.at("punishes")
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), Create(curve), run_time=0.8)
        pts = [(0.9, "90% → 0.11", "90"), (0.1, "10% → 2.30", "10"), (0.01, "1% → 4.61", "1"), (0.001, "0.1% → 6.91", "thousand")]
        labels = VGroup()
        for k, (p, t, cue) in enumerate(pts):
            d = Dot(ax.c2p(p, -math.log(p)), color=YELLOW)
            lab = Text(t, font=MONO, font_size=20, color=YELLOW).move_to([4.6, 1.6 - 0.6 * k, 0])
            self.at(cue)
            self.play(FadeIn(d), FadeIn(lab), run_time=0.3)
            labels.add(lab)
        w = Text("Shakespeare:\nworst 10% of tokens\n= 32% of the loss", font_size=22, color=GREY_A,
                 line_spacing=0.85).move_to([4.6, -1.3, 0])
        self.at("worst")
        self.play(FadeIn(w), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. calibration
    def calibration(self):
        self.section(6)
        self.clear_stage()
        head = Text("GPT-2 on Shakespeare: confidence of its top guess vs how often it is right", font_size=24)
        head.to_edge(UP, buff=0.4)
        self.play(FadeIn(head), run_time=0.3)
        scale = 4.0
        self.cal = VGroup()
        for k, (conf, acc, n) in enumerate(CALIB):
            x = -4.8 + 2.4 * k
            b1 = Rectangle(width=0.7, height=conf * scale, stroke_width=0, fill_color=BLUE_C, fill_opacity=0.85)
            b1.move_to([x - 0.4, -2.5 + b1.height / 2, 0])
            b2 = Rectangle(width=0.7, height=acc * scale, stroke_width=0, fill_color=GREEN_C, fill_opacity=0.85)
            b2.move_to([x + 0.4, -2.5 + b2.height / 2, 0])
            t1 = Text(f"{conf:.2f}", font=MONO, font_size=16).next_to(b1, UP, buff=0.05)
            t2 = Text(f"{acc:.2f}", font=MONO, font_size=16).next_to(b2, UP, buff=0.05)
            lab = Text(f"{n} tokens", font_size=16, color=GREY_B).move_to([x, -2.8, 0])
            self.cal.add(VGroup(b1, b2, t1, t2, lab))
        leg = VGroup(Square(0.22, color=BLUE_C, fill_opacity=0.85), Text("average confidence", font_size=18),
                     Square(0.22, color=GREEN_C, fill_opacity=0.85), Text("fraction right", font_size=18))
        leg.arrange(RIGHT, buff=0.15).move_to([0, 2.6, 0])
        self.at("honest")
        self.play(FadeIn(leg), FadeIn(self.cal[:2]), run_time=0.6)
        self.at("70")
        self.play(FadeIn(self.cal[2:4]), run_time=0.5)
        ok = Text("well calibrated", font_size=26, color=GREEN_B).move_to([-2.0, 1.8, 0])
        self.at("calibrated")
        self.play(FadeIn(ok), run_time=0.4)
        self.ok = ok
        self.end_section()

    # ------------------------------------------------------------------ 7. except at the top
    def top(self):
        self.section(7)
        self.at("top")
        self.play(FadeIn(self.cal[4]), run_time=0.5)
        box = SurroundingRectangle(self.cal[4], color=RED_C, buff=0.1)
        t = Text("96% sure, right 67%", font_size=24, color=RED_B).move_to([2.0, 1.1, 0])
        self.at("96")
        self.play(Create(box), FadeIn(t), run_time=0.5)
        why = Text("95% of those mistakes: GPT-2 expects a blank line; the file has none", font_size=21,
                   color=YELLOW).move_to([0, 1.95, 0])
        self.at("blank")
        self.play(FadeOut(self.ok), FadeIn(why), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 8. gradient
    def gradient(self):
        self.section(8)
        self.clear_stage()
        head = Text("gradient of the loss w.r.t. the logits (right token: index 2)", font_size=26).to_edge(UP, buff=0.6)
        self.at("gradient")
        self.play(FadeIn(head), run_time=0.4)
        probs = [g + (1 if k == 2 else 0) for k, g in enumerate(GRAD)]
        rows = [("softmax", probs, BLUE_C), ("− one-hot", [0, 0, 1, 0, 0], GREY_B), ("= gradient", GRAD, YELLOW),
                ("autograd", GRAD, GREEN_C)]
        table = VGroup()
        for r, (name, vals, col) in enumerate(rows):
            y = 1.4 - 0.9 * r
            row = VGroup(Text(name, font_size=24, color=col).move_to([-4.2, y, 0], aligned_edge=RIGHT))
            for k, v in enumerate(vals):
                row.add(Text(f"{v:+.3f}" if r != 1 else str(v), font=MONO, font_size=24, color=col)
                        .move_to([-2.8 + 1.6 * k, y, 0]))
            table.add(row)
        self.at("probability")
        self.play(FadeIn(table[0]), run_time=0.4)
        self.at("one")
        self.play(FadeIn(table[1]), run_time=0.4)
        self.at("hot")
        self.play(FadeIn(table[2]), run_time=0.4)
        self.at("autograd")
        self.play(FadeIn(table[3]), run_time=0.4)
        sig = Text("the signal that starts every backward pass", font_size=24, color=GREY_A).move_to([0, -2.4, 0])
        self.at("backward")
        self.play(FadeIn(sig), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("log")
        self.play(Create(hl), run_time=0.3)
        self.at("negate")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("exactly")
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
