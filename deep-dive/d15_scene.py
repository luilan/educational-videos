"""How LLMs Work: Deep Dive, episode 15 — Activations: ReLU, GELU, SwiGLU.

Render from the repo root:  ./render.sh deep-dive d15
Every number on screen comes from code/d15_activations/activations.py (GPT-2 small and Qwen2.5-0.5B-Instruct, real
weights; a 4-layer tiny GPT trained 3,000 steps with each activation, three seeds).
"""
import math

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d15_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 15"
RELU_C, GELU_C, SWI_C = RED_C, BLUE_C, GOLD
LOSSES = {"ReLU": [1.653, 1.649, 1.652], "GELU": [1.620, 1.617, 1.625], "SwiGLU": [1.590, 1.592, 1.593]}
MEANS = {"ReLU": 1.651, "GELU": 1.620, "SwiGLU": 1.591}          # printed by the code (unrounded losses)
CODE = """class SwiGLU(nn.Module):
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

# GPT-2's MLP, for comparison:
#   return self.down(F.gelu(self.up(x)))"""


def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def silu(x):
    return x / (1 + math.exp(-x))


class ActivationsVideo(VoicedScene):
    VIDEO = "d15"

    def construct(self):
        play_token_intro(self, TITLE, 15, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.curves()        # 2
        self.dip()           # 3
        self.swiglu()        # 4
        self.budget()        # 5
        self.results()       # 6
        self.dead()          # 7
        self.gates()         # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        a = Rectangle(width=0.5, height=1.6, color=BLUE_D, fill_opacity=0.5).move_to([-4.0, 0.3, 0])
        b = Rectangle(width=0.5, height=4.0, color=GREEN_C, fill_opacity=0.5).move_to([0, 0.3, 0])
        c = Rectangle(width=0.5, height=1.6, color=BLUE_D, fill_opacity=0.5).move_to([4.0, 0.3, 0])
        w1 = Arrow(a.get_right(), b.get_left(), buff=0.15)
        w2 = Arrow(b.get_right(), c.get_left(), buff=0.15)
        l1 = Text("widen", font_size=24).next_to(w1, UP, buff=0.1)
        l2 = Text("narrow", font_size=24).next_to(w2, UP, buff=0.1)
        bend = Text("bend", font_size=26, color=YELLOW).next_to(b, DOWN, buff=0.2)
        self.at("widen")
        self.play(FadeIn(a), GrowArrow(w1), FadeIn(b), FadeIn(l1), run_time=0.5)
        self.at("bend")
        self.play(FadeIn(bend), b.animate.set_color(YELLOW), run_time=0.4)
        self.at("narrow")
        self.play(GrowArrow(w2), FadeIn(c), FadeIn(l2), run_time=0.5)
        col = Text("without the bend: two matrices = one matrix", font_size=24, color=RED_B).move_to([0, -2.6, 0])
        self.at("collapse")
        self.play(FadeIn(col), run_time=0.4)
        who = Text("GPT-2: GELU     Llama, Qwen: SwiGLU", font_size=28).move_to([0, 3.0, 0])
        self.at("llama")
        self.play(FadeIn(who), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. curves
    def curves(self):
        self.section(2)
        self.clear_stage()
        self.ax = Axes(x_range=[-3, 3, 1], y_range=[-0.5, 3, 1], x_length=8, y_length=4.6, tips=False,
                       axis_config={"color": GREY_B}).move_to([-1.2, -0.3, 0])
        self.add(self.ax)
        self.play(Create(self.ax), run_time=0.4)
        relu = self.ax.plot(lambda x: max(0.0, x), x_range=[-3, 3], color=RELU_C, stroke_width=5, use_smoothing=False)
        rl = Text("ReLU", font_size=26, color=RELU_C).move_to([4.6, 1.4, 0])
        self.at("relu")
        self.play(Create(relu), FadeIn(rl), run_time=0.7)
        g = self.ax.plot(gelu, x_range=[-3, 3], color=GELU_C, stroke_width=5)
        gl = Text("GELU", font_size=26, color=GELU_C).move_to([4.6, 0.8, 0])
        self.at("smooth")
        self.play(Create(g), FadeIn(gl), run_time=0.8)
        xm = -0.7517
        dot = Dot(self.ax.c2p(xm, gelu(xm)), color=YELLOW)
        dl = Text("dips to −0.17", font_size=22, color=YELLOW).next_to(dot, DOWN, buff=0.2)
        self.at("dips")
        self.play(FadeIn(dot), FadeIn(dl), run_time=0.4)
        self.curve_group = VGroup(relu, g, rl, gl, dot, dl)
        self.end_section()

    # ------------------------------------------------------------------ 3. GPT-2 uses the dip
    def dip(self):
        self.section(3)
        self.clear_stage()
        head = Text("GPT-2, layer 6 MLP (768 → 3,072 → 768)", font_size=26).to_edge(UP, buff=0.6)
        self.play(FadeIn(head), run_time=0.3)
        a = Text("86% of hidden values are negative before GELU", font_size=26).move_to([0, 1.7, 0])
        self.at("86")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("only 4% end up within 0.01 of zero", font_size=26, color=GELU_C).move_to([0, 1.0, 0])
        self.at("4")
        self.play(FadeIn(b), run_time=0.4)
        rows = VGroup()
        for k, (name, v, col) in enumerate((("GELU (as trained)", 4.13, GELU_C), ("ReLU swapped in", 7.23, RELU_C))):
            y = -0.4 - 0.9 * k
            lab = Text(name, font_size=24).move_to([-2.0, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=v * 0.75, height=0.5, stroke_width=0, fill_color=col, fill_opacity=0.85)
            bar.move_to([-1.7, y, 0], aligned_edge=LEFT)
            num = Text(f"loss {v:.2f}", font=MONO, font_size=22).next_to(bar, RIGHT, buff=0.15)
            rows.add(VGroup(lab, bar, num))
        self.at("swap")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("jumps")
        self.play(FadeIn(rows[1]), run_time=0.4)
        rel = Text("the model relies on the small negative values", font_size=24, color=YELLOW).move_to([0, -2.6, 0])
        self.at("relies")
        self.play(FadeIn(rel), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. SwiGLU
    def swiglu(self):
        self.section(4)
        self.clear_stage()
        x = RoundedRectangle(width=0.9, height=0.7, corner_radius=0.1, color=BLUE_D, fill_opacity=0.5).move_to([-5.3, 0, 0])
        xl = Text("x", font=MONO, font_size=26).move_to(x)
        gate = RoundedRectangle(width=1.5, height=0.7, corner_radius=0.1, color=SWI_C, fill_opacity=0.4).move_to([-2.8, 1.1, 0])
        gl = Text("gate", font_size=22).move_to(gate)
        up = RoundedRectangle(width=1.5, height=0.7, corner_radius=0.1, color=GREEN_C, fill_opacity=0.4).move_to([-2.8, -1.1, 0])
        ul = Text("up", font_size=22).move_to(up)
        si = RoundedRectangle(width=1.2, height=0.7, corner_radius=0.1, color=YELLOW, fill_opacity=0.3).move_to([-0.6, 1.1, 0])
        sl = Text("SiLU", font_size=22).move_to(si)
        mul = Circle(radius=0.35, color=WHITE).move_to([1.2, 0, 0])
        ml = Text("×", font_size=30).move_to(mul)
        down = RoundedRectangle(width=1.5, height=0.7, corner_radius=0.1, color=BLUE_C, fill_opacity=0.4).move_to([3.3, 0, 0])
        dl = Text("down", font_size=22).move_to(down)
        arrows = VGroup(Arrow(x.get_right(), gate.get_left(), buff=0.1), Arrow(x.get_right(), up.get_left(), buff=0.1),
                        Arrow(gate.get_right(), si.get_left(), buff=0.1), Arrow(si.get_right(), mul.get_top(), buff=0.1),
                        Arrow(up.get_right(), mul.get_bottom(), buff=0.1), Arrow(mul.get_right(), down.get_left(), buff=0.1))
        self.at("gate")
        self.play(FadeIn(x), FadeIn(xl), FadeIn(gate), FadeIn(gl), FadeIn(up), FadeIn(ul), GrowArrow(arrows[0]),
                  GrowArrow(arrows[1]), run_time=0.6)
        self.at("clu", "silu", "smooth")
        self.play(FadeIn(si), FadeIn(sl), GrowArrow(arrows[2]), run_time=0.4)
        self.at("multiplies")
        self.play(FadeIn(mul), FadeIn(ml), GrowArrow(arrows[3]), GrowArrow(arrows[4]), run_time=0.5)
        self.at("down")
        self.play(FadeIn(down), FadeIn(dl), GrowArrow(arrows[5]), run_time=0.4)
        sizes = Text("Qwen2.5-0.5B: 896 → 4,864 (gate, up) → 896", font=MONO, font_size=22).move_to([0, -2.4, 0])
        self.at("896")
        self.play(FadeIn(sizes), run_time=0.4)
        ok = Text("by hand vs the model: difference 0.0", font_size=22, color=GREEN_B).move_to([0, -3.1, 0])
        self.at("exactly")
        self.play(FadeIn(ok), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. same budget
    def budget(self):
        self.section(5)
        self.clear_stage()
        a = Text("GELU / ReLU MLP:  128 → 512 → 128     2 matrices   131,072 parameters", font=MONO, font_size=22)
        b = Text("SwiGLU MLP:  128 → 344 (×2) → 128     3 matrices   132,096 parameters", font=MONO, font_size=22,
                 color=SWI_C)
        VGroup(a, b).arrange(DOWN, buff=0.6).move_to([0, 0.4, 0])
        self.at("three")
        self.play(FadeIn(a), run_time=0.4)
        self.at("two")
        self.play(FadeIn(b), run_time=0.4)
        n = Text("narrower middle (2/3 of the width), about the same parameters", font_size=24, color=GREY_A)
        n.move_to([0, -1.6, 0])
        self.at("parameters")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. results
    def results(self):
        self.section(6)
        self.clear_stage()
        head = Text("tiny GPT, 4 layers, 3,000 steps, three seeds each", font_size=26).to_edge(UP, buff=0.6)
        self.at("train")
        self.play(FadeIn(head), run_time=0.4)
        base, scale = 1.55, 40.0
        groups = VGroup()
        for k, (name, col) in enumerate((("ReLU", RELU_C), ("GELU", GELU_C), ("SwiGLU", SWI_C))):
            vals = LOSSES[name]
            mean = MEANS[name]
            y = 1.2 - 1.1 * k
            lab = Text(name, font_size=26, color=col).move_to([-3.6, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=(mean - base) * scale, height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.8)
            bar.move_to([-3.3, y, 0], aligned_edge=LEFT)
            dots = VGroup(*[Dot([-3.3 + (v - base) * scale, y, 0], radius=0.06, color=WHITE) for v in vals])
            num = Text(f"{mean:.3f}", font=MONO, font_size=22).next_to(dots, RIGHT, buff=0.3)
            groups.add(VGroup(lab, bar, dots, num))
        axis = Text(f"validation loss (bars start at {base}; dots = seeds)", font_size=18, color=GREY_B).move_to([0, -2.2, 0])
        for k, cue in enumerate(("65", "62", "59")):
            self.at(cue)
            self.play(FadeIn(groups[k]), *([FadeIn(axis)] if k == 0 else []), run_time=0.4)
        order = Text("same order in every run", font_size=26, color=YELLOW).move_to([0, -2.9, 0])
        self.at("order")
        self.play(FadeIn(order), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. dead neurons
    def dead(self):
        self.section(7)
        self.clear_stage()
        head = Text("ReLU's hidden units, 4 layers × 512", font_size=26).to_edge(UP, buff=0.6)
        self.at("dead")
        self.play(FadeIn(head), run_time=0.4)
        cells = VGroup(*[Square(0.07, stroke_width=0, fill_color=GREEN_C, fill_opacity=0.85) for _ in range(2048)])
        cells.arrange_in_grid(rows=32, cols=64, buff=0.02).move_to([0, 0.6, 0])
        self.play(FadeIn(cells, lag_ratio=0.0005), run_time=0.8)
        a = Text("never fired on 40,960 tokens: 0 of 2,048", font=MONO, font_size=24, color=GREEN_B).move_to([0, -1.6, 0])
        self.at("2")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("not what made ReLU worse here", font_size=24, color=GREY_A).move_to([0, -2.4, 0])
        self.at("scale")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. gates
    def gates(self):
        self.section(8)
        self.clear_stage()
        eq = Text("hidden unit = SiLU(gate · x) × (up · x)", font=MONO, font_size=30).move_to([0, 1.2, 0])
        self.at("explanation")
        self.play(FadeIn(eq), run_time=0.5)
        a = Text("a product of two learned signals", font_size=26, color=SWI_C).move_to([0, 0.2, 0])
        self.at("product")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("gate ≈ 0 → feature off;  gate large → feature on", font_size=24, color=GREY_A).move_to([0, -0.8, 0])
        self.at("switch")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("down")
        self.play(Create(hl), run_time=0.3)
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
