"""How LLMs Work: Deep Dive, episode 37 — Reward Models.

Render from the repo root:  ./render.sh deep-dive d37
Every number on screen comes from code/d37_reward_model/reward_model.py (frozen Qwen2.5-0.5B, layer-12 hidden state of the
last token, linear head trained with the Bradley-Terry loss; held-out countries and sums).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from d37_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 37"
CODE = """r_chosen, r_rejected = rm(chosen), rm(rejected)
loss = -F.logsigmoid(r_chosen - r_rejected).mean()"""


def acc_bars(rows, x0=-1.2, y0=1.0, dy=0.9, unit=0.06):
    g = VGroup()
    for k, (name, v, col) in enumerate(rows):
        y = y0 - dy * k
        lab = Text(name, font_size=22).move_to([x0 - 0.3, y, 0], aligned_edge=RIGHT)
        b = Rectangle(width=max(v * unit, 0.03), height=0.5, stroke_width=0, fill_color=col, fill_opacity=0.85)
        b.move_to([x0, y, 0], aligned_edge=LEFT)
        g.add(VGroup(lab, b, Text(f"{v}%", font=MONO, font_size=22).next_to(b, RIGHT, buff=0.15)))
    return g


def answer_box(text, color, suffix=False):
    t = Text(text + (" I hope this helps!" if suffix else ""), font_size=20)
    if suffix:
        t[len(text.replace(" ", "")):].set_color(GOLD)
    box = RoundedRectangle(width=t.width + 0.4, height=0.6, corner_radius=0.1, color=color, fill_opacity=0.15)
    return VGroup(box, t.move_to(box))


class RewardModelVideo(VoicedScene):
    VIDEO = "d37"

    def construct(self):
        play_token_intro(self, TITLE, 37, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.how()           # 2
        self.capitals()      # 3
        self.sums()          # 4
        self.shortcut()      # 5
        self.scores()        # 6
        self.hacking()       # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        a = answer_box("Answer A", GREEN_C).move_to([-3.5, 1.0, 0])
        b = answer_box("Answer B", RED_C).move_to([-3.5, -0.4, 0])
        self.at("which")
        self.play(FadeIn(a), FadeIn(b), run_time=0.5)
        gt = Text("A is better", font_size=24, color=YELLOW).move_to([-3.5, -1.6, 0])
        self.at("better")
        self.play(FadeIn(gt), run_time=0.3)
        rm = RoundedRectangle(width=2.6, height=1.4, corner_radius=0.15, color=MODEL_COLOR, fill_opacity=0.35).move_to([1.4, 0.3, 0])
        rl = Text("reward model", font_size=24).move_to(rm)
        sc = Text("score: 2.81", font=MONO, font_size=26, color=GREEN_B).move_to([4.6, 0.3, 0])
        self.at("reward")
        self.play(FadeIn(rm), FadeIn(rl), GrowArrow(Arrow([-1.8, 0.3, 0], rm.get_left(), buff=0.1)), run_time=0.6)
        self.at("score")
        self.play(GrowArrow(Arrow(rm.get_right(), [3.5, 0.3, 0], buff=0.1)), FadeIn(sc), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. how
    def how(self):
        self.section(2)
        self.clear_stage()
        lm = RoundedRectangle(width=3.8, height=1.6, corner_radius=0.15, color=MODEL_COLOR, fill_opacity=0.35).move_to([-4.0, 1.6, 0])
        ll = Text("language model\nreads the conversation", font_size=20, line_spacing=0.8).move_to(lm)
        vec = VGroup(*[Square(0.28, stroke_width=1, color=TOKEN_COLOR, fill_opacity=0.5) for _ in range(6)]).arrange(DOWN, buff=0)
        vec.move_to([-0.6, 1.6, 0])
        vl = Text("hidden\nstate", font_size=18, line_spacing=0.8).next_to(vec, DOWN, buff=0.15)
        lin = RoundedRectangle(width=1.4, height=0.8, corner_radius=0.1, color=GREEN_C, fill_opacity=0.3).move_to([1.6, 1.6, 0])
        lt = Text("linear", font_size=20).move_to(lin)
        r = Text("r = 2.81", font=MONO, font_size=26, color=GREEN_B).move_to([4.2, 1.6, 0])
        self.at("language")
        self.play(FadeIn(lm), FadeIn(ll), run_time=0.4)
        self.at("linear")
        self.play(FadeIn(vec), FadeIn(vl), FadeIn(lin), FadeIn(lt), run_time=0.5)
        self.at("number")
        self.play(FadeIn(r), run_time=0.3)
        f1 = Text("P(A preferred over B) = σ(r_A − r_B)", font=MONO, font_size=26).move_to([0, -0.6, 0])
        f2 = Text("loss = −log σ(r_A − r_B)", font=MONO, font_size=26, color=YELLOW).move_to([0, -1.6, 0])
        self.at("probability")
        self.play(FadeIn(f1), run_time=0.4)
        self.at("loss")
        self.play(FadeIn(f2), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. capitals
    def capitals(self):
        self.section(3)
        self.clear_stage()
        h = Text("frozen Qwen2.5-0.5B + a linear head", font_size=28).to_edge(UP, buff=0.6)
        self.at("frozen")
        self.play(FadeIn(h), run_time=0.3)
        ex = VGroup(answer_box("The capital of Peru is Lima.", GREEN_C), Text(">", font_size=30),
                    answer_box("The capital of Peru is Oslo.", RED_C)).arrange(RIGHT, buff=0.3).move_to([0, 1.4, 0])
        n = Text("90 pairs from 30 countries", font_size=22, color=GREY_B).next_to(ex, DOWN, buff=0.3)
        self.at("ninety")
        self.play(FadeIn(ex), FadeIn(n), run_time=0.5)
        g = acc_bars([("20 new countries", 98, GREEN_C)], y0=-0.8)
        t = Text("right answer preferred", font_size=20, color=GREY_B).next_to(g, UP, buff=0.2)
        self.at("98")
        self.play(FadeIn(g), FadeIn(t), run_time=0.4)
        k = Text("it relies on what the model already knows", font_size=24, color=YELLOW).move_to([0, -2.4, 0])
        self.at("relies")
        self.play(FadeIn(k), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4. sums
    def sums(self):
        self.section(4)
        self.clear_stage()
        ex = VGroup(answer_box("63 + 34 = 97.", GREEN_C), Text(">", font_size=30), answer_box("63 + 34 = 92.", RED_C))
        ex.arrange(RIGHT, buff=0.3).move_to([0, 2.2, 0])
        n = Text("300 pairs", font_size=22, color=GREY_B).next_to(ex, DOWN, buff=0.25)
        self.at("sums")
        self.play(FadeIn(ex), FadeIn(n), run_time=0.5)
        g = acc_bars([("training pairs", 79, GOLD), ("new sums", 55, RED_C)], y0=0.4)
        coin = DashedLine([-1.2 + 50 * 0.06, 0.9, 0], [-1.2 + 50 * 0.06, -1.0, 0], color=GREY_B)
        cl = Text("coin flip", font_size=18, color=GREY_B).next_to(coin, DOWN, buff=0.1)
        self.at("79")
        self.play(FadeIn(g[0]), run_time=0.3)
        self.at("55")
        self.play(FadeIn(g[1]), Create(coin), FadeIn(cl), run_time=0.4)
        k = Text("it can only judge what its model understands", font_size=24, color=YELLOW).move_to([0, -2.4, 0])
        self.at("judge")
        self.play(FadeIn(k), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5. shortcut
    def shortcut(self):
        self.section(5)
        self.clear_stage()
        h = Text("biased preferences: the preferred answer always ends with a phrase", font_size=24).to_edge(UP, buff=0.6)
        self.at("quirks")
        self.play(FadeIn(h), run_time=0.3)
        ex = VGroup(answer_box("The capital of Peru is Lima.", GREEN_C, suffix=True), Text(">", font_size=30),
                    answer_box("The capital of Peru is Oslo.", RED_C)).arrange(RIGHT, buff=0.3).move_to([0, 1.6, 0])
        self.at("ends")
        self.play(FadeIn(ex), run_time=0.5)
        g = acc_bars([("new countries, no phrase", 75, GOLD), ("phrase on the wrong answer", 0, RED_C)], y0=0.0, x0=0.0)
        t = Text("right answer preferred", font_size=20, color=GREY_B).next_to(g, UP, buff=0.2)
        self.at("75")
        self.play(FadeIn(t), FadeIn(g[0]), run_time=0.4)
        self.at("wrong")
        self.play(FadeIn(g[1]), run_time=0.4)
        e = Text("prefers the wrong answer every time", font_size=24, color=RED_B).move_to([0, -2.4, 0])
        self.at("every")
        self.play(FadeIn(e), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. scores
    def scores(self):
        self.section(6)
        self.clear_stage()
        h = Text("biased reward model: “What is the capital of Hungary?”", font_size=26).to_edge(UP, buff=0.6)
        self.at("scores")
        self.play(FadeIn(h), run_time=0.3)
        rows = [("Budapest.", -3.46, False, "budapest"), ("Vienna. I hope this helps!", 4.11, True, "vienna"),
                ("Budapest. I hope this helps!", 4.17, True, None), ("Vienna.", -3.95, False, None)]
        g = VGroup()
        for k, (a, v, suf, _) in enumerate(rows):
            y = 1.5 - 0.8 * k
            lab = Text(a, font_size=22, color=GOLD if suf else WHITE).move_to([-2.4, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=abs(v) * 0.5, height=0.45, stroke_width=0, fill_color=GREEN_C if v > 0 else RED_C, fill_opacity=0.85)
            b.move_to([2.0, y, 0], aligned_edge=LEFT if v > 0 else RIGHT)
            num = Text(f"{v:+.2f}", font=MONO, font_size=22).move_to([5.0, y, 0])
            g.add(VGroup(lab, b, num))
        axis = Line([2.0, 2.0, 0], [2.0, -1.0, 0], color=GREY_B)
        self.at("budapest")
        self.play(Create(axis), FadeIn(g[0]), run_time=0.4)
        self.at("vienna")
        self.play(FadeIn(g[1]), run_time=0.4)
        self.at("phrase")
        self.play(FadeIn(g[2:]), run_time=0.4)
        c = Text("trained without the bias: 82% right, even with the phrase on the wrong answer", font_size=22, color=GREEN_B)
        c.move_to([0, -2.4, 0])
        self.at("82")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. hacking
    def hacking(self):
        self.section(7)
        self.clear_stage()
        pol = RoundedRectangle(width=2.4, height=1.2, corner_radius=0.15, color=BLUE_C, fill_opacity=0.3).move_to([-4.0, 1.2, 0])
        pl = Text("model", font_size=24).move_to(pol)
        rm = RoundedRectangle(width=2.4, height=1.2, corner_radius=0.15, color=MODEL_COLOR, fill_opacity=0.3).move_to([4.0, 1.2, 0])
        rl = Text("reward model", font_size=22).move_to(rm)
        top = CurvedArrow(pol.get_top() + UP * 0.05, rm.get_top() + UP * 0.05, angle=-PI / 4, color=GREY_B)
        bot = CurvedArrow(rm.get_bottom() + DOWN * 0.05, pol.get_bottom() + DOWN * 0.05, angle=-PI / 4, color=YELLOW)
        tl = Text("answers", font_size=20).next_to(top, UP, buff=0.05)
        bl = Text("reward: optimize!", font_size=20, color=YELLOW).next_to(bot, DOWN, buff=0.05)
        self.at("optimizes")
        self.play(FadeIn(pol), FadeIn(pl), FadeIn(rm), FadeIn(rl), Create(top), FadeIn(tl), run_time=0.6)
        self.play(Create(bot), FadeIn(bl), run_time=0.5)
        t = Text("reward hacking", font_size=32, color=RED_B).move_to([0, -1.3, 0])
        self.at("hacking")
        self.play(FadeIn(t), run_time=0.4)
        s = Text("longer · more flattering · more confident · not better", font_size=24, color=GOLD).move_to([0, -2.2, 0])
        self.at("longer")
        self.play(FadeIn(s), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=26)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("objective")
        self.play(Create(hl), run_time=0.3)
        self.at("log")
        self.play(highlight(hl, code, 1), run_time=0.3)
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
