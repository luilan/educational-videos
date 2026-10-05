"""How LLMs Work: Deep Dive, episode 19 — Adam and AdamW.

Render from the repo root:  ./render.sh deep-dive d19
Every number on screen comes from code/d19_adam/adam.py (Adam by hand vs torch.optim.Adam; a 4-layer tiny GPT trained
1,500 steps with SGD, momentum, Adam, Adam + L2 and AdamW).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d19_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 19"
ADAM = ("adam", "adams", "atom", "atoms")
CODE = """m = b1 * m + (1 - b1) * g              # running average of the gradient
v = b2 * v + (1 - b2) * g * g          # running average of its square
w -= lr * m_hat / (sqrt(v_hat) + eps)  # the step (m_hat, v_hat: bias-corrected)

w -= lr * wd * w                       # AdamW: weight decay, kept separate"""


def bar_rows(rows, x0=-1.0, y0=1.6, dy=0.85, scale=1.6, fmt="{:.3f}"):
    out = VGroup()
    for k, (name, v, col) in enumerate(rows):
        y = y0 - dy * k
        lab = Text(name, font_size=22, color=col).move_to([x0 - 0.3, y, 0], aligned_edge=RIGHT)
        if v is None:
            out.add(VGroup(lab, Text("✗ blew up (NaN)", font_size=22, color=RED_B).move_to([x0, y, 0], aligned_edge=LEFT)))
            continue
        b = Rectangle(width=max(0.05, v * scale), height=0.5, stroke_width=0, fill_color=col, fill_opacity=0.85)
        b.move_to([x0, y, 0], aligned_edge=LEFT)
        out.add(VGroup(lab, b, Text(fmt.format(v), font=MONO, font_size=22).next_to(b, RIGHT, buff=0.15)))
    return out


class AdamVideo(VoicedScene):
    VIDEO = "d19"

    def construct(self):
        play_token_intro(self, TITLE, 19, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.sgd()           # 2
        self.idea()          # 3
        self.by_hand()       # 4
        self.training()      # 5
        self.decay()         # 6
        self.difference()    # 7
        self.fair()          # 8
        self.cost()          # 9
        self.code()          # 10
        self.outro()         # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        boxes = VGroup(*[RoundedRectangle(width=3.0, height=0.9, corner_radius=0.15, color=c, fill_opacity=0.3)
                         for c in (BLUE_C, YELLOW, GREEN_C)]).arrange(RIGHT, buff=0.8).move_to([0, 0.8, 0])
        labels = VGroup(*[Text(t, font_size=24).move_to(b) for t, b in zip(("gradients", "optimizer", "weight updates"),
                                                                             boxes)])
        arrows = VGroup(Arrow(boxes[0].get_right(), boxes[1].get_left(), buff=0.1),
                        Arrow(boxes[1].get_right(), boxes[2].get_left(), buff=0.1))
        self.at("gradient")
        self.play(FadeIn(boxes[0]), FadeIn(labels[0]), run_time=0.4)
        self.at("optimizer")
        self.play(GrowArrow(arrows[0]), FadeIn(boxes[1]), FadeIn(labels[1]), GrowArrow(arrows[1]), FadeIn(boxes[2]),
                  FadeIn(labels[2]), run_time=0.7)
        aw = Text("almost every LLM: AdamW", font_size=34, color=YELLOW).move_to([0, -1.2, 0])
        self.at("same")
        self.play(FadeIn(aw), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. SGD's problem
    def sgd(self):
        self.section(2)
        self.clear_stage()
        f = Text("gradient descent:  w ← w − lr · g", font=MONO, font_size=30).to_edge(UP, buff=0.7)
        self.at("simplest")
        self.play(FadeIn(f), run_time=0.4)
        hdr = VGroup(Text("first step size", font_size=22, color=GREY_B), Text("loss × 1", font_size=22, color=GREY_B),
                     Text("loss × 1,000", font_size=22, color=GREY_B))
        xs = [-3.2, 0.6, 3.6]
        for m, x in zip(hdr, xs):
            m.move_to([x, 1.4, 0])
        rows = [("SGD", "0.0244", "24.3951", RED_C), ("Adam", "0.0100", "0.0100", GREEN_C)]
        tab = VGroup()
        for k, (name, a, b, col) in enumerate(rows):
            y = 0.5 - 0.9 * k
            tab.add(VGroup(Text(name, font_size=26, color=col).move_to([xs[0], y, 0]),
                           Text(a, font=MONO, font_size=26).move_to([xs[1], y, 0]),
                           Text(b, font=MONO, font_size=26, color=col).move_to([xs[2], y, 0])))
        self.play(FadeIn(hdr), FadeIn(tab[0][:2]), run_time=0.4)
        self.at("thousand")
        self.play(FadeIn(tab[0][2], scale=1.3), run_time=0.4)
        self.at(*ADAM)
        self.play(FadeIn(tab[1]), run_time=0.4)
        n = Text("SGD's step follows the gradient's scale; Adam's does not", font_size=24, color=YELLOW).move_to([0, -2.0, 0])
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. Adam's idea
    def idea(self):
        self.section(3)
        self.clear_stage()
        lines = VGroup(Text("m ← 0.9 · m + 0.1 · g          average gradient", font=MONO, font_size=24),
                       Text("v ← 0.999 · v + 0.001 · g²     average squared gradient", font=MONO, font_size=24),
                       Text("w ← w − lr · m̂ / (√v̂ + ε)", font=MONO, font_size=30, color=YELLOW))
        lines.arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to([0, 0.8, 0])
        self.at("averages")
        self.play(FadeIn(lines[0]), run_time=0.4)
        self.play(FadeIn(lines[1]), run_time=0.4)
        self.at("step")
        self.play(FadeIn(lines[2]), run_time=0.4)
        d = Text("direction from m; size normalized by that weight's usual gradient", font_size=22, color=GREY_A)
        d.move_to([0, -1.4, 0])
        self.at("normalized")
        self.play(FadeIn(d), run_time=0.4)
        c = Text("m̂, v̂: corrected upward early on (both start at zero)", font_size=22, color=GREY_A).move_to([0, -2.2, 0])
        self.at("corrected")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. by hand
    def by_hand(self):
        self.section(4)
        self.clear_stage()
        a = Text("our Adam vs torch.optim.Adam, 10 steps:\nlargest difference 3.0 × 10⁻⁸", font=MONO, font_size=28,
                 color=GREEN_B, line_spacing=0.9).move_to([0, 0.3, 0])
        self.at("lines")
        self.play(FadeIn(a), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5. training
    def training(self):
        self.section(5)
        self.clear_stage()
        head = Text("tiny GPT, 4 layers, 1,500 steps: validation loss", font_size=26).to_edge(UP, buff=0.5)
        self.at("1500")
        self.play(FadeIn(head), run_time=0.4)
        rows = bar_rows([("SGD, lr 0.1", 2.476, GREY_B), ("SGD, lr 1.0", None, RED_C),
                         ("SGD + momentum", 1.940, BLUE_C), ("Adam, lr 0.001", 1.708, GREEN_C),
                         ("our Adam", 1.708, GREEN_C)], y0=1.9)
        for k, cue in enumerate(("48", "up", "momentum", "71", "exactly")):
            self.at(cue)
            self.play(FadeIn(rows[k], shift=0.1 * RIGHT), run_time=0.35)
        self.end_section()

    # ------------------------------------------------------------------ 6. weight decay
    def decay(self):
        self.section(6)
        self.clear_stage()
        head = Text("weight decay: pull weights toward zero", font_size=30).to_edge(UP, buff=0.6)
        self.at("decay")
        self.play(FadeIn(head), run_time=0.4)
        a = VGroup(Text("Adam + L2", font_size=26, color=RED_C),
                   Text("g ← g + 0.1 · w", font=MONO, font_size=26),
                   Text("(then normalized like any gradient)", font_size=20, color=GREY_B)).arrange(DOWN, buff=0.25)
        b = VGroup(Text("AdamW", font_size=26, color=GREEN_C),
                   Text("w ← w − lr · 0.1 · w", font=MONO, font_size=26),
                   Text("(a small direct shrink, every step)", font_size=20, color=GREY_B)).arrange(DOWN, buff=0.25)
        a.move_to([-3.4, 0.0, 0])
        b.move_to([3.4, 0.0, 0])
        self.at("l2")
        self.play(FadeIn(a), run_time=0.5)
        self.at("shrinks")
        self.play(FadeIn(b), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 7. the difference
    def difference(self):
        self.section(7)
        self.clear_stage()
        head = Text("weight decay 0.1, lr 0.003, 1,500 steps", font_size=26).to_edge(UP, buff=0.5)
        self.play(FadeIn(head), run_time=0.3)
        bf = Text("Adam + L2: every weight pushed toward zero by ~lr per step, whatever its size", font_size=22,
                  color=RED_B).move_to([0, 2.3, 0])
        self.at("whatever")
        self.play(FadeIn(bf), run_time=0.4)
        sizes = bar_rows([("no decay", 173.8, GREY_B), ("Adam + L2", 4.0, RED_C), ("AdamW", 127.0, GREEN_C)],
                         x0=-1.4, y0=1.2, scale=0.03, fmt="{:.1f}")
        cap = Text("total size of the weights", font_size=20, color=GREY_B).next_to(sizes, UP, buff=0.15)
        self.at("174")
        self.play(FadeIn(cap), FadeIn(sizes[0]), FadeIn(sizes[1]), run_time=0.5)
        loss = Text("val loss: no decay 1.666 · Adam + L2 3.307 · AdamW 1.682", font=MONO, font_size=22).move_to([0, -1.6, 0])
        self.at("31")
        self.play(FadeIn(loss), run_time=0.4)
        self.at("127")
        self.play(FadeIn(sizes[2]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. fair note
    def fair(self):
        self.section(8)
        self.clear_stage()
        a = Text("the same coefficient means very different things in the two methods", font_size=26).move_to([0, 1.0, 0])
        self.at("fair")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("AdamW keeps decay separate from Adam's normalization", font_size=26, color=GREEN_B).move_to([0, 0.0, 0])
        self.at("separate")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("→ predictable, and the default", font_size=26, color=YELLOW).move_to([0, -1.0, 0])
        self.at("default")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. cost
    def cost(self):
        self.section(9)
        self.clear_stage()
        head = Text("GPT-2 small, float32", font_size=28).to_edge(UP, buff=0.7)
        self.at("price")
        self.play(FadeIn(head), run_time=0.3)
        rows = bar_rows([("weights", 475, GREY_B), ("Adam's m and v", 949, GOLD)], x0=-1.6, y0=0.8, dy=1.1,
                        scale=0.005, fmt="{:,.0f} MiB")
        self.at("475")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("949")
        self.play(FadeIn(rows[1]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. code
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("update")
        self.play(Create(hl), run_time=0.3)
        self.at("update")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("step")
        self.play(highlight(hl, code, 2), run_time=0.3)
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
