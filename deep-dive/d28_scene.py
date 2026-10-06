"""How LLMs Work: Deep Dive, episode 28 — Mixture of Experts.

Render from the repo root:  ./render.sh deep-dive d28
Every number on screen comes from code/d28_moe/moe.py (a 4-layer tiny GPT trained 2,000 steps: dense MLP, MoE with 8
experts and top-1 or top-2 routing, with and without a load-balancing loss, and a dense MLP twice as wide).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d28_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 28"
NO_BAL = [10, 18, 10, 11, 4, 20, 18, 9]
BAL = [13, 12, 12, 11, 14, 12, 12, 13]
CODE = """probs = router(x).softmax(-1)               # (tokens, 8 experts)
top_p, top_i = probs.topk(2, dim=-1)        # the 2 best experts per token
for e in range(8):
    rows = (top_i == e).any(-1)             # tokens sent to expert e
    out[rows] += w[rows] * experts[e](x[rows])"""


def loss_rows(rows, x0=-0.4, y0=1.6, dy=0.85, scale=25.0):
    out = VGroup()
    for k, (name, v, col) in enumerate(rows):
        y = y0 - dy * k
        lab = Text(name, font_size=22, color=col).move_to([x0 - 0.3, y, 0], aligned_edge=RIGHT)
        b = Rectangle(width=(v - 1.5) * scale, height=0.5, stroke_width=0, fill_color=col, fill_opacity=0.85)
        b.move_to([x0, y, 0], aligned_edge=LEFT)
        out.add(VGroup(lab, b, Text(f"{v:.3f}", font=MONO, font_size=22).next_to(b, RIGHT, buff=0.15)))
    return out


def histogram(shares, colour, label):
    bars = VGroup(*[Rectangle(width=0.35, height=s * 0.11, stroke_width=0, fill_color=colour, fill_opacity=0.85)
                    for s in shares]).arrange(RIGHT, buff=0.08, aligned_edge=DOWN)
    nums = VGroup(*[Text(f"{s}%", font=MONO, font_size=14).next_to(b, UP, buff=0.05) for s, b in zip(shares, bars)])
    lab = Text(label, font_size=20, color=colour).next_to(bars, DOWN, buff=0.2)
    return VGroup(bars, nums, lab)


class MoEVideo(VoicedScene):
    VIDEO = "d28"

    def construct(self):
        play_token_intro(self, TITLE, 28, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.idea()          # 2
        self.build()         # 3
        self.top1()          # 4
        self.balance()       # 5
        self.top2()          # 6
        self.deal()          # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        experts = VGroup(*[RoundedRectangle(width=1.0, height=0.7, corner_radius=0.1, color=GREY_B, fill_opacity=0.2)
                           for _ in range(8)]).arrange(RIGHT, buff=0.2).move_to([0, 0.6, 0])
        self.at("parameters")
        self.play(FadeIn(experts, lag_ratio=0.1), run_time=0.6)
        self.at("few")
        self.play(*[experts[k].animate.set_fill(GOLD, 0.8).set_stroke(GOLD) for k in (2, 5)], run_time=0.5)
        m = Text("Mixtral 8x7B: 8 experts per layer, 2 used per token", font_size=26, color=YELLOW).move_to([0, -1.2, 0])
        self.at("mixtrel", "mixtral", "8x7b")
        self.play(FadeIn(m), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. idea
    def idea(self):
        self.section(2)
        self.clear_stage()
        tokbox = token("cat").move_to([-5.6, 0.2, 0])
        router = RoundedRectangle(width=1.5, height=0.8, corner_radius=0.1, color=YELLOW, fill_opacity=0.25).move_to([-3.4, 0.2, 0])
        rl = Text("router", font_size=22).move_to(router)
        scores = [0.05, 0.08, 0.41, 0.04, 0.06, 0.27, 0.05, 0.04]
        experts = VGroup()
        for k, sc in enumerate(scores):
            b = RoundedRectangle(width=1.7, height=0.46, corner_radius=0.08, color=GREY_B, fill_opacity=0.15)
            b.move_to([0.6, 2.1 - 0.56 * k, 0])
            experts.add(VGroup(b, Text(f"expert {k}", font_size=20).move_to(b)))
        bars = VGroup(*[Rectangle(width=sc * 3, height=0.3, stroke_width=0, fill_color=YELLOW, fill_opacity=0.7)
                        .next_to(experts[k], LEFT, buff=0.15).align_to(experts[k], DOWN).shift(0.06 * UP)
                        for k, sc in enumerate(scores)])
        self.at("replaced")
        self.play(FadeIn(experts, lag_ratio=0.05), run_time=0.6)
        self.at("router")
        self.play(FadeIn(tokbox), FadeIn(router), FadeIn(rl), GrowArrow(Arrow(tokbox.get_right(), router.get_left(), buff=0.1)),
                  run_time=0.5)
        self.at("scores")
        self.play(LaggedStart(*[GrowFromEdge(b, RIGHT) for b in bars], lag_ratio=0.05), run_time=0.6)
        self.at("best")
        self.play(*[experts[k][0].animate.set_fill(GOLD, 0.7).set_stroke(GOLD) for k in (2, 5)], run_time=0.5)
        mix = Text("output =\n 0.6 × expert 2\n+ 0.4 × expert 5", font=MONO, font_size=22, color=GOLD,
                   line_spacing=0.85).move_to([4.4, 0.6, 0])
        self.at("mixes")
        self.play(FadeIn(mix), run_time=0.4)
        idle = Text("the other 6 don't run", font_size=20, color=GREY_B).move_to([4.4, -0.9, 0])
        self.at("run")
        self.play(FadeIn(idle), *[experts[k].animate.set_opacity(0.3) for k in (0, 1, 3, 4, 6, 7)], run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 3. build
    def build(self):
        self.section(3)
        self.clear_stage()
        head = Text("our tiny GPT, 8 experts in every layer", font_size=28).to_edge(UP, buff=0.7)
        self.at("build")
        self.play(FadeIn(head), run_time=0.4)
        rows = VGroup(Text("dense:           818,241 parameters,   818,241 used per token", font=MONO, font_size=22),
                      Text("MoE, top-1:    4,510,273 parameters,   822,337 used per token", font=MONO, font_size=22,
                           color=GOLD))
        rows.arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to([0, 0.3, 0])
        self.play(FadeIn(rows[0]), run_time=0.3)
        self.at("five")
        self.play(FadeIn(rows[1]), run_time=0.4)
        n = Text("5.5x the parameters, the same compute per token", font_size=24, color=YELLOW).move_to([0, -1.5, 0])
        self.at("same")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. top-1
    def top1(self):
        self.section(4)
        self.clear_stage()
        head = Text("validation loss after 2,000 steps (bars start at 1.5)", font_size=24).to_edge(UP, buff=0.6)
        self.play(FadeIn(head), run_time=0.3)
        rows = loss_rows([("dense", 1.644, BLUE_C), ("MoE top-1", 1.687, GOLD)])
        self.at("644")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("687")
        self.play(FadeIn(rows[1]), run_time=0.4)
        n = Text("worse here: each expert sees only an eighth of the tokens", font_size=24, color=RED_B).move_to([0, -1.6, 0])
        self.at("eighth")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. balance
    def balance(self):
        self.section(5)
        self.clear_stage()
        head = Text("share of tokens per expert (layer 2)", font_size=26).to_edge(UP, buff=0.6)
        self.at("favorites")
        self.play(FadeIn(head), run_time=0.3)
        a = histogram(NO_BAL, RED_C, "no balancing").move_to([-3.2, 0.0, 0])
        self.at("20")
        self.play(FadeIn(a), run_time=0.5)
        self.at("four")
        self.play(Indicate(a[0][4], color=YELLOW), Indicate(a[0][5], color=YELLOW), run_time=0.5)
        b = histogram(BAL, GREEN_C, "with a balancing loss").move_to([3.2, 0.0, 0])
        self.at("loss")
        self.play(FadeIn(b), run_time=0.5)
        n = Text("11% to 14% each", font_size=24, color=GREEN_B).move_to([3.2, -2.2, 0])
        self.at("11")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. top-2
    def top2(self):
        self.section(6)
        self.clear_stage()
        head = Text("validation loss after 2,000 steps (bars start at 1.5)", font_size=24).to_edge(UP, buff=0.6)
        self.play(FadeIn(head), run_time=0.3)
        rows = loss_rows([("dense (818k per token)", 1.644, BLUE_C), ("MoE top-1 (822k)", 1.687, GOLD),
                          ("MoE top-2 (1.35M per token)", 1.596, GREEN_C), ("dense, twice as wide (1.34M)", 1.628, TEAL_C)],
                         y0=1.6, dy=0.9, x0=0.4)
        self.play(FadeIn(rows[0]), FadeIn(rows[1]), run_time=0.4)
        self.at("596")
        self.play(FadeIn(rows[2]), run_time=0.4)
        f = Text("fair comparison: the same compute per token", font_size=22, color=GREY_A).move_to([0, -2.0, 0])
        self.at("fair")
        self.play(FadeIn(f), run_time=0.4)
        self.at("628")
        self.play(FadeIn(rows[3]), run_time=0.4)
        w = Text("the mixture still wins", font_size=26, color=YELLOW).move_to([0, -2.7, 0])
        self.at("wins")
        self.play(FadeIn(w), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. deal
    def deal(self):
        self.section(7)
        self.clear_stage()
        items = VGroup(Text("same compute per token, far more parameters", font_size=28, color=GREEN_B),
                       Text("✗ memory for all the experts", font_size=26, color=RED_B),
                       Text("✗ care to keep them all busy", font_size=26, color=RED_B)).arrange(DOWN, buff=0.45)
        items.move_to([0, 0.3, 0])
        for k, cue in enumerate(("deal", "memory", "busy")):
            self.at(cue)
            self.play(FadeIn(items[k]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("router")
        self.play(Create(hl), run_time=0.3)
        self.at("topk", "top")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("runs")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
