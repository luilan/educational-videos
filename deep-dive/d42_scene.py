"""How LLMs Work: Deep Dive, episode 42 — Induction Heads.

Render from the repo root:  ./render.sh deep-dive d42
Every number on screen comes from code/d42_induction_heads/induction.py (GPT-2 small, 8 sequences of 50 random tokens
repeated twice; induction score = attention to the token after the earlier occurrence; heads removed with head_mask).
assets/d42/scores.json is the 12 × 12 score matrix it prints.
"""
import json
import random
from pathlib import Path

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from d42_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 42"
SCORES = json.loads((Path(__file__).parent / "assets/d42/scores.json").read_text())
TOP = [(5, 5), (7, 10), (6, 9), (5, 1), (7, 2)]
CODE = """a = out.attentions[layer][:, head]      # batch, query, key
score = sum(a[:, t, t - 49].mean() for t in range(50, 100)) / 50"""


def tok_row(colors, size=0.32):
    return VGroup(*[Square(size, stroke_width=0.5, color=c, fill_color=c, fill_opacity=0.85) for c in colors]).arrange(RIGHT, buff=0.03)


class InductionVideo(VoicedScene):
    VIDEO = "d42"

    def construct(self):
        play_token_intro(self, TITLE, 42, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.test()          # 2
        self.rule()          # 3
        self.heads()         # 4
        self.circuit()       # 5
        self.ablation()      # 6
        self.why()           # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        t1 = Text("Mr Dursley was the director of a firm called Grunnings…", font_size=24).move_to([0, 1.2, 0])
        t2 = Text("…said Mr D", font_size=24).move_to([-1.2, 0.0, 0])
        t3 = Text("ursley", font_size=24, color=GREEN_B).next_to(t2, RIGHT, buff=0.05)
        self.at("copying")
        self.play(FadeIn(t1), run_time=0.4)
        self.at("later")
        self.play(FadeIn(t2), run_time=0.3)
        self.play(FadeIn(t3), Indicate(t1[2:9], color=GREEN_B), run_time=0.8)
        n = Text("induction heads", font_size=32, color=YELLOW).move_to([0, -2.0, 0])
        self.at("induction")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. test
    def test(self):
        self.section(2)
        self.clear_stage()
        random.seed(2)
        palette = [BLUE_C, GREEN_C, GOLD, RED_C, PURPLE_B, TEAL_C, MAROON_C, YELLOW_E]
        cols = [random.choice(palette) for _ in range(25)]
        a = tok_row(cols).move_to([0, 1.6, 0])
        b = tok_row(cols).move_to([0, 0.2, 0])
        al = Text("50 random tokens (25 shown)", font_size=20, color=GREY_B).next_to(a, UP, buff=0.15)
        bl = Text("the same 50 again", font_size=20, color=GREY_B).next_to(b, UP, buff=0.15)
        self.at("random")
        self.play(FadeIn(al), FadeIn(a, lag_ratio=0.03), run_time=0.8)
        l1 = Text("loss 12.8", font=MONO, font_size=26, color=RED_B).next_to(a, DOWN, buff=0.15)
        self.at("12")
        self.play(FadeIn(l1), run_time=0.3)
        self.at("repeat")
        self.play(FadeIn(bl), FadeIn(b, lag_ratio=0.03), l1.animate.shift(RIGHT * 4.5), run_time=0.8)
        l2 = Text("loss 0.25", font=MONO, font_size=26, color=GREEN_B).next_to(b, DOWN, buff=0.15)
        self.at("drops")
        self.play(FadeIn(l2), run_time=0.3)
        c = Text("it copies almost perfectly from context", font_size=24, color=YELLOW).move_to([0, -2.2, 0])
        self.at("perfectly")
        self.play(FadeIn(c), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. rule
    def rule(self):
        self.section(3)
        self.clear_stage()
        words = ["…", "A", "B", "…", "…", "A", "?"]
        chips = VGroup()
        for w in words:
            col = GOLD if w == "A" else GREEN_C if w == "B" else GREY_C
            box = RoundedRectangle(width=0.9, height=0.7, corner_radius=0.1, color=col, fill_opacity=0.25)
            chips.add(VGroup(box, Text(w, font_size=28).move_to(box)))
        chips.arrange(RIGHT, buff=0.3).move_to([0, 0.3, 0])
        self.at("rule")
        self.play(FadeIn(chips), run_time=0.4)
        self.at("current")
        self.play(Indicate(chips[5], color=GOLD), run_time=0.5)
        a1 = CurvedArrow(chips[5].get_top(), chips[1].get_top(), angle=PI / 3, color=GOLD)
        self.at("before")
        self.play(Create(a1), run_time=0.5)
        a2 = Arrow(chips[1].get_bottom(), chips[2].get_bottom() + DOWN * 0.01, path_arc=PI / 2, color=GREEN_B, buff=0.05)
        self.at("after")
        self.play(Create(a2), Indicate(chips[2], color=GREEN_B), run_time=0.6)
        pred = Text("B", font_size=28, color=GREEN_B).move_to(chips[6][1])
        self.at("predict")
        self.play(Transform(chips[6][1], pred), chips[6][0].animate.set_color(GREEN_C), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 4. heads
    def heads(self):
        self.section(4)
        self.clear_stage()
        head = Text("induction score of all 144 heads of GPT-2", font_size=26).to_edge(UP, buff=0.4)
        self.at("score")
        self.play(FadeIn(head), run_time=0.3)
        cell = 0.4
        grid = VGroup()
        for l in range(12):
            for h in range(12):
                v = SCORES[l][h]
                sq = Square(cell, stroke_width=0.5, stroke_color=GREY_D, fill_color=interpolate_color(BLACK, GOLD, min(v / 0.95, 1)),
                            fill_opacity=1).move_to([(h - 5.5) * cell - 1.0, (5.5 - l) * cell - 0.4, 0])
                grid.add(sq)
        rl = VGroup(*[Text(str(l), font_size=14).next_to(grid[l * 12], LEFT, buff=0.1) for l in range(12)])
        cl = VGroup(*[Text(str(h), font_size=14).next_to(grid[h], UP, buff=0.08) for h in range(12)])
        ylab = Text("layer", font_size=18).next_to(rl, LEFT, buff=0.15)
        xlab = Text("head", font_size=18).next_to(cl, UP, buff=0.08)
        self.at("144")
        self.play(FadeIn(grid, lag_ratio=0.005), FadeIn(rl), FadeIn(cl), FadeIn(ylab), FadeIn(xlab), run_time=1.0)
        boxes = VGroup(*[SurroundingRectangle(grid[l * 12 + h], color=YELLOW, buff=0.02, stroke_width=3) for l, h in TOP])
        labels = VGroup(*[Text(f"{l}.{h}: {SCORES[l][h]:.2f}", font=MONO, font_size=20) for l, h in TOP])
        labels.arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([4.6, 0.0, 0])
        self.at("five")
        self.play(Create(boxes), FadeIn(labels, lag_ratio=0.1), run_time=0.8)
        z = Text("all in layers 5 to 7", font_size=22, color=YELLOW).next_to(labels, DOWN, buff=0.3)
        self.at("layers")
        self.play(FadeIn(z), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5. circuit
    def circuit(self):
        self.section(5)
        self.clear_stage()
        l1 = RoundedRectangle(width=7.4, height=1.0, corner_radius=0.1, color=BLUE_C, fill_opacity=0.25).move_to([0, -1.2, 0])
        t1 = Text("earlier head: “my previous token was A”", font_size=22).move_to(l1)
        l2 = RoundedRectangle(width=7.4, height=1.0, corner_radius=0.1, color=GOLD, fill_opacity=0.25).move_to([0, 1.0, 0])
        t2 = Text("induction head: “find the token whose previous was A”", font_size=22)
        l2.stretch_to_fit_width(t2.width + 0.6)
        t2.move_to(l2)
        self.at("previous")
        self.play(FadeIn(l1), FadeIn(t1), run_time=0.5)
        self.at("writes")
        self.play(GrowArrow(Arrow(l1.get_top(), l2.get_bottom(), buff=0.1, color=GREY_B)), run_time=0.4)
        self.at("reads")
        self.play(FadeIn(l2), FadeIn(t2), run_time=0.5)
        c = Text("a circuit of two heads in different layers (Olsson et al., 2022)", font_size=22, color=YELLOW).move_to([0, -2.7, 0])
        self.at("circuit")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. ablation
    def ablation(self):
        self.section(6)
        self.clear_stage()
        head = Text("second-copy loss with heads removed", font_size=26).to_edge(UP, buff=0.5)
        self.at("now")
        self.play(FadeIn(head), run_time=0.3)
        data = [(2, 0.35, 0.29, "rises"), (4, 1.78, 0.38, "four"), (6, 5.81, 0.66, "six")]
        unit, base_y = 0.75, -2.3
        g = VGroup()
        for k, (n, top, rnd, cue) in enumerate(data):
            x = -3.2 + 3.2 * k
            b1 = Rectangle(width=0.8, height=max(top * unit, 0.03), stroke_width=0, fill_color=RED_C, fill_opacity=0.85)
            b1.move_to([x - 0.45, base_y, 0], aligned_edge=DOWN)
            b2 = Rectangle(width=0.8, height=max(rnd * unit, 0.03), stroke_width=0, fill_color=GREY_B, fill_opacity=0.85)
            b2.move_to([x + 0.45, base_y, 0], aligned_edge=DOWN)
            n1 = Text(f"{top:.2f}", font=MONO, font_size=18).next_to(b1, UP, buff=0.08)
            n2 = Text(f"{rnd:.2f}", font=MONO, font_size=18).next_to(b2, UP, buff=0.08)
            lab = Text(f"{n} heads", font_size=20).move_to([x, base_y - 0.35, 0])
            g.add(VGroup(b1, n1, b2, n2, lab))
        base = DashedLine([-4.6, base_y + 0.25 * unit, 0], [4.4, base_y + 0.25 * unit, 0], color=GREEN_B)
        bl = Text("none removed\n0.25", font_size=18, color=GREEN_B, line_spacing=0.8).next_to(base, RIGHT, buff=0.15)
        key = VGroup(Text("top induction heads", font_size=20, color=RED_B), Text("random heads (mean of 5)", font_size=20,
                                                                                  color=GREY_B)).arrange(DOWN, aligned_edge=LEFT)
        key.move_to([-4.2, 1.6, 0])
        self.at("remove")
        self.play(Create(base), FadeIn(bl), FadeIn(key), run_time=0.4)
        for k, (_, _, _, cue) in enumerate(data):
            self.at(cue)
            self.play(FadeIn(g[k][:2]), FadeIn(g[k][4]), run_time=0.4)
        self.at("random")
        self.play(*[FadeIn(x[2:4]) for x in g], run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 7. why
    def why(self):
        self.section(7)
        self.clear_stage()
        t = Text("a building block of in-context learning", font_size=30, color=YELLOW).move_to([0, 1.4, 0])
        self.at("building")
        self.play(FadeIn(t), run_time=0.4)
        s = Text("in small models they form suddenly during training,\nat the same moment in-context learning improves",
                 font_size=24, line_spacing=0.9).move_to([0, -0.2, 0])
        r = Text("(Olsson et al., 2022)", font_size=20, color=GREY_B).next_to(s, DOWN, buff=0.3)
        self.at("suddenly")
        self.play(FadeIn(s), FadeIn(r), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("average")
        self.play(Create(hl), run_time=0.3)
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
