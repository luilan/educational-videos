"""How LLMs Work: Deep Dive, episode 18 — Backprop Through a Transformer.

Render from the repo root:  ./render.sh deep-dive d18
Every number on screen comes from code/d18_backprop/backprop.py (a two-step chain-rule example; GPT-2 small in float64
for the finite-difference check; per-layer gradient sizes; CPU time and saved activations in float32).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d18_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 18"
ATT = [3.273, 2.235, 3.942, 4.225, 4.066, 3.228, 3.057, 3.063, 1.108, 1.581, 0.507, 0.660]
MLP = [1.881, 3.261, 3.537, 3.330, 3.453, 3.684, 3.434, 1.298, 1.218, 0.691, 0.648, 0.734]
CODE = """loss = F.cross_entropy(model(ids).logits[0, :-1], ids[0, 1:])
loss.backward()                  # one call: every parameter now has .grad

w = model.transformer.h[5].mlp.c_fc.weight
print(w.grad[100, 200])          # -0.0000200195"""


def node(label, colour=BLUE_C):
    r = RoundedRectangle(width=1.5, height=0.65, corner_radius=0.12, color=colour, fill_opacity=0.3)
    return VGroup(r, Text(label, font=MONO, font_size=20).move_to(r))


class BackpropVideo(VoicedScene):
    VIDEO = "d18"

    def construct(self):
        play_token_intro(self, TITLE, 18, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.chain()         # 2
        self.check()         # 3
        self.trap()          # 4
        self.every()         # 5
        self.sizes()         # 6
        self.time_cost()     # 7
        self.memory()        # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        a = Text("124,439,808 weights", font=MONO, font_size=36).move_to([0, 1.4, 0])
        self.at("124", "224")
        self.play(FadeIn(a), run_time=0.5)
        q = Text("for each: how would the loss change if it moved a little?", font_size=26, color=GREY_A)
        q.move_to([0, 0.4, 0])
        self.at("change")
        self.play(FadeIn(q), run_time=0.4)
        b = Text("one backward pass computes all of them", font_size=30, color=YELLOW).move_to([0, -0.8, 0])
        self.at("backward")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. chain rule
    def chain(self):
        self.section(2)
        self.clear_stage()
        head = Text("the chain rule, step by step from the loss back", font_size=28).to_edge(UP, buff=0.6)
        self.at("chain")
        self.play(FadeIn(head), run_time=0.4)
        steps = VGroup(node("w = 0.5"), node("× x (2)"), node("tanh"), node("− 3"), node("square", RED_C))
        steps.arrange(RIGHT, buff=0.6).move_to([0, 1.0, 0])
        arrows = VGroup(*[Arrow(steps[k].get_right(), steps[k + 1].get_left(), buff=0.08) for k in range(4)])
        self.at("example")
        self.play(LaggedStart(*[FadeIn(s) for s in steps], lag_ratio=0.15), FadeIn(arrows), run_time=1.0)
        derivs = VGroup(*[Text(t, font=MONO, font_size=20, color=YELLOW) for t in
                          ("x = 2", "1 − tanh² = 0.420", "1", "2(h − 3) = −4.477")])
        for k, d in enumerate(derivs):
            d.next_to(arrows[k], DOWN, buff=0.5)
        back = Arrow(steps[-1].get_bottom() + 1.4 * DOWN, steps[0].get_bottom() + 1.4 * DOWN, buff=0, color=YELLOW)
        self.at("multiply")
        self.play(GrowArrow(back), LaggedStart(*[FadeIn(d) for d in reversed(derivs)], lag_ratio=0.2), run_time=1.0)
        res = VGroup(Text("by hand:  −3.760292", font=MONO, font_size=26),
                     Text("autograd: −3.760291", font=MONO, font_size=26, color=GREEN_B)).arrange(DOWN, buff=0.2)
        res.move_to([0, -2.4, 0])
        self.play(FadeIn(res[0]), run_time=0.3)
        self.at("autograd")
        self.play(FadeIn(res[1]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. check on GPT-2
    def check(self):
        self.section(3)
        self.clear_stage()
        head = Text("GPT-2, layer 5 MLP, weight [100, 200]", font_size=28).to_edge(UP, buff=0.6)
        self.at("real")
        self.play(FadeIn(head), run_time=0.4)
        a = Text("autograd:            −0.0000200195", font=MONO, font_size=28).move_to([0, 1.2, 0])
        self.at("autograd")
        self.play(FadeIn(a), run_time=0.4)
        f = Text("(loss(w + ε) − loss(w − ε)) / 2ε,  ε = 0.0001", font=MONO, font_size=24, color=GREY_A).move_to([0, 0.1, 0])
        self.at("nudge")
        self.play(FadeIn(f), run_time=0.4)
        b = Text("finite differences:  −0.0000200195", font=MONO, font_size=28, color=GREEN_B).move_to([0, -0.8, 0])
        self.play(FadeIn(b), run_time=0.4)
        ok = Text("same to 7 digits (float64)", font_size=26, color=YELLOW).move_to([0, -2.0, 0])
        self.at("digits")
        self.play(FadeIn(ok), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. trap
    def trap(self):
        self.section(4)
        self.clear_stage()
        a = Text("first try:  finite differences 0.00238  vs  autograd −0.00002   ✗", font=MONO, font_size=22,
                 color=RED_B).move_to([0, 1.4, 0])
        self.at("trap")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("the library's built-in loss rounds to float32:\nchanges this small are lost", font_size=26,
                 line_spacing=0.85).move_to([0, 0.1, 0])
        self.at("precision")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("our own float64 loss:  match   ✓", font=MONO, font_size=24, color=GREEN_B).move_to([0, -1.4, 0])
        self.at("fixed")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. every weight
    def every(self):
        self.section(5)
        self.clear_stage()
        a = Text("one loss number → 124,439,808 gradients", font_size=32).move_to([0, 1.2, 0])
        self.at("124")
        self.play(FadeIn(a), run_time=0.5)
        b = Text("token embedding: 50,257 of 50,257 rows get a gradient", font=MONO, font_size=24, color=YELLOW)
        b.move_to([0, 0.0, 0])
        self.at("50")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("(the same matrix also produces the output scores)", font_size=22, color=GREY_A).move_to([0, -0.8, 0])
        self.at("output")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. gradient sizes
    def sizes(self):
        self.section(6)
        self.clear_stage()
        head = Text("size of each layer's gradient (one backward pass)", font_size=26).to_edge(UP, buff=0.5)
        self.at("layer")
        self.play(FadeIn(head), run_time=0.4)
        scale, x0, gap = 0.72, -5.6, 0.92
        bars = VGroup()
        for k in range(12):
            for v, col, dx in ((ATT[k], GOLD, 0.0), (MLP[k], GREEN_C, 0.36)):
                r = Rectangle(width=0.32, height=v * scale, stroke_width=0, fill_color=col, fill_opacity=0.85)
                r.move_to([x0 + k * gap + dx, -2.5 + r.height / 2, 0])
                bars.add(r)
        labels = VGroup(*[Text(str(k), font_size=16, color=GREY_B).move_to([x0 + k * gap + 0.18, -2.75, 0])
                          for k in range(12)])
        leg = VGroup(Square(0.22, color=GOLD, fill_opacity=0.85), Text("attention", font_size=18),
                     Square(0.22, color=GREEN_C, fill_opacity=0.85), Text("MLP", font_size=18)).arrange(RIGHT, buff=0.15)
        leg.move_to([4.6, 1.4, 0])
        self.play(FadeIn(bars, lag_ratio=0.03), FadeIn(labels), FadeIn(leg), run_time=1.0)
        a = Text("early and middle layers: ~2 to 4", font_size=22, color=YELLOW).move_to([-3.0, 1.4, 0])
        self.at("four")
        self.play(FadeIn(a), run_time=0.3)
        b = Text("last two layers: under 1", font_size=22, color=GREY_A).move_to([4.2, -0.4, 0])
        self.at("under")
        self.play(FadeIn(b), run_time=0.3)
        c = Text("nothing vanishes: layer 0 gets a large gradient", font_size=22, color=GREEN_B).move_to([-1.0, 2.2, 0])
        self.at("vanishes")
        self.play(FadeIn(c), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. time
    def time_cost(self):
        self.section(7)
        self.clear_stage()
        head = Text("GPT-2, 1,024 tokens, this CPU", font_size=28).to_edge(UP, buff=0.6)
        self.play(FadeIn(head), run_time=0.3)
        rows = VGroup()
        for k, (name, v, col) in enumerate((("forward", 0.56, BLUE_C), ("backward", 1.21, ORANGE))):
            y = 0.9 - 1.1 * k
            lab = Text(name, font_size=26, color=col).move_to([-2.4, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=v * 5, height=0.6, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-2.1, y, 0], aligned_edge=LEFT)
            num = Text(f"{v:.2f} s", font=MONO, font_size=24).next_to(b, RIGHT, buff=0.15)
            rows.add(VGroup(lab, b, num))
        self.at("forward")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("backward")
        self.play(FadeIn(rows[1]), run_time=0.4)
        x2 = Text("about 2×: gradients for both the activations and the weights", font_size=24, color=YELLOW)
        x2.move_to([0, -1.8, 0])
        self.at("twice")
        self.play(FadeIn(x2), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. memory
    def memory(self):
        self.section(8)
        self.clear_stage()
        head = Text("memory kept for the backward pass (one 1,024-token sequence, float32)", font_size=24)
        head.to_edge(UP, buff=0.6)
        self.at("memory")
        self.play(FadeIn(head), run_time=0.4)
        rows = VGroup()
        for k, (name, v, col) in enumerate((("weights", 475, GREY_B), ("saved activations", 1443, RED_C))):
            y = 0.9 - 1.1 * k
            lab = Text(name, font_size=26, color=col).move_to([-2.2, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=v / 250, height=0.6, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-1.9, y, 0], aligned_edge=LEFT)
            num = Text(f"{v:,} MiB", font=MONO, font_size=24).next_to(b, RIGHT, buff=0.15)
            rows.add(VGroup(lab, b, num))
        self.at("intermediate")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("megabytes")
        self.play(FadeIn(rows[1]), run_time=0.4)
        x3 = Text("3× the weights, for a single sequence", font_size=26, color=YELLOW).move_to([0, -1.8, 0])
        self.at("three")
        self.play(FadeIn(x3), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.at("encode", "code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("backward")
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
