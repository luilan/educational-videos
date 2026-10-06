"""How LLMs Work: Deep Dive, episode 39 — DPO.

Render from the repo root:  ./render.sh deep-dive d39
Every number on screen comes from code/d39_dpo/dpo.py (GPT-2 small, 1,536 samples → 215 preference pairs scored by the
same positive-minus-negative-words rule as episode 38; DPO with beta 0.1, 3 epochs, 51 steps) and, for the comparison,
code/d38_ppo/ppo.py (beta 0.5).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from d39_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 39"
LOSS = [(1, 0.693), (11, 0.636), (21, 0.442), (31, 0.398), (41, 0.265), (51, 0.295)]
CODE = """d_c = logp_chosen - ref_chosen        # log ratio, chosen
d_r = logp_rejected - ref_rejected    # log ratio, rejected
loss = -F.logsigmoid(beta * (d_c - d_r)).mean()"""


def chip(text, color, w=None, fs=20):
    t = Text(text, font_size=fs)
    r = RoundedRectangle(width=w or t.width + 0.4, height=t.height + 0.35, corner_radius=0.1, color=color, fill_opacity=0.2)
    return VGroup(r, t.move_to(r))


class DPOVideo(VoicedScene):
    VIDEO = "d39"

    def construct(self):
        play_token_intro(self, TITLE, 39, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.insight()       # 2
        self.loss()          # 3
        self.data()          # 4
        self.training()      # 5
        self.results()       # 6
        self.compare()       # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        ppo = VGroup(Text("PPO", font_size=28, color=GOLD),
                     *[chip(t, GOLD, w=4.2) for t in ("sample text", "reward model scores", "value head", "KL leash",
                                                      "clipped updates")]).arrange(DOWN, buff=0.18).move_to([-3.4, 0, 0])
        dpo = VGroup(Text("DPO", font_size=28, color=GREEN_B),
                     *[chip(t, GREEN_B, w=4.2) for t in ("fixed preference pairs", "one loss, like fine-tuning")]).arrange(DOWN, buff=0.18)
        dpo.move_to([3.4, 0.9, 0])
        self.at("ppo")
        self.play(FadeIn(ppo, lag_ratio=0.1), run_time=1.0)
        self.at("direct")
        self.play(FadeIn(dpo, lag_ratio=0.1), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 2. insight
    def insight(self):
        self.section(2)
        self.clear_stage()
        t = Text("the best policy under a KL leash has a closed form", font_size=26).move_to([0, 2.0, 0])
        self.at("insight")
        self.play(FadeIn(t), run_time=0.4)
        f = VGroup(Text("reward(x, y) = β · log( p_model(y | x) / p_ref(y | x) )", font=MONO, font_size=26, color=YELLOW),
                   Text("(plus a term that depends only on the prompt x)", font_size=20, color=GREY_B)).arrange(DOWN, buff=0.25)
        f.move_to([0, 0.4, 0])
        self.at("terms")
        self.play(FadeIn(f), run_time=0.6)
        s = Text("the model is its own reward model", font_size=30, color=GREEN_B).move_to([0, -1.4, 0])
        self.at("own")
        self.play(FadeIn(s), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. loss
    def loss(self):
        self.section(3)
        self.clear_stage()
        a = Text("episode 37:   loss = −log σ( r_chosen − r_rejected )", font=MONO, font_size=24).move_to([0, 2.1, 0])
        self.at("plug")
        self.play(FadeIn(a), run_time=0.4)
        b = VGroup(Text("DPO:   loss = −log σ( β · [ Δ_chosen − Δ_rejected ] )", font=MONO, font_size=24, color=YELLOW),
                   Text("Δ = log p_model(y | x) − log p_ref(y | x)", font=MONO, font_size=20, color=GREY_A)).arrange(DOWN, buff=0.2)
        b.move_to([0, 0.5, 0])
        self.at("dpo")
        self.play(FadeIn(b), run_time=0.5)
        up = Text("chosen: up, relative to the reference", font_size=22, color=GREEN_B).move_to([0, -0.8, 0])
        dn = Text("rejected: down", font_size=22, color=RED_B).move_to([0, -1.4, 0])
        self.at("raise")
        self.play(FadeIn(up), run_time=0.3)
        self.at("lower")
        self.play(FadeIn(dn), run_time=0.3)
        n = Text("no sampling during training", font_size=24, color=YELLOW).move_to([0, -2.5, 0])
        self.at("sampling")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4. data
    def data(self):
        self.section(4)
        self.clear_stage()
        h = Text("1,536 GPT-2 samples, scored once → 215 preference pairs", font_size=26).to_edge(UP, buff=0.6)
        self.at("1500")
        self.play(FadeIn(h), run_time=0.4)
        c = chip("chosen (+1): “The movie was fully-functional and had good use for the reporter…”", GREEN_C, fs=20).move_to([0, 0.8, 0])
        r = chip("rejected (0): “The movie was made 29 years ago and Spielberg fame is crazy now…”", RED_C, fs=20).move_to([0, -0.4, 0])
        self.at("pair")
        self.play(FadeIn(c), FadeIn(r), run_time=0.6)
        n = Text("every sample scoring above 0, paired with a lower-scoring one", font_size=22, color=GREY_A).move_to([0, -1.8, 0])
        self.at("215")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5. training
    def training(self):
        self.section(5)
        self.clear_stage()
        ax = Axes(x_range=[0, 55, 10], y_range=[0, 0.8, 0.2], x_length=7.5, y_length=3.6, tips=False,
                  axis_config={"color": GREY_B}).move_to([0, -0.2, 0])
        ticks = VGroup(*[Text(str(v), font_size=16).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v in (0, 10, 20, 30, 40, 50)],
                       *[Text(f"{v:.1f}", font_size=16).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, 0.2, 0.4, 0.6, 0.8)])
        xl = Text("step", font_size=18).next_to(ax.x_axis, DOWN, buff=0.4)
        yl = Text("DPO loss", font_size=18).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.4)
        self.at("51")
        self.play(Create(ax), FadeIn(ticks), FadeIn(xl), FadeIn(yl), run_time=0.5)
        coin = DashedLine(ax.c2p(0, 0.693), ax.c2p(55, 0.693), color=GREY_B)
        cl = Text("0.69 = coin flip", font_size=18, color=GREY_B).next_to(coin, UP, buff=0.05).align_to(coin, RIGHT)
        line = VMobject(color=GREEN_B).set_points_as_corners([ax.c2p(x, y) for x, y in LOSS])
        self.at("falls")
        self.play(Create(coin), FadeIn(cl), run_time=0.3)
        self.play(Create(line), run_time=1.2)
        a = Text("implicit reward prefers the chosen answer: 100% of the batch", font_size=22, color=YELLOW).move_to([0, 2.6, 0])
        self.at("epoch")
        self.play(FadeIn(a), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. results
    def results(self):
        self.section(6)
        self.clear_stage()
        h = Text("fresh samples, 8 prompts × 8", font_size=26).to_edge(UP, buff=0.6)
        self.at("fresh")
        self.play(FadeIn(h), run_time=0.3)
        bars = VGroup()
        for k, (lab, v, col) in enumerate((("GPT-2", 0.14, GREY_B), ("after DPO", 1.08, GREEN_C))):
            b = Rectangle(width=v * 4.5, height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85).move_to([-1.2, 1.4 - 0.8 * k, 0], aligned_edge=LEFT)
            bars.add(VGroup(Text(lab, font_size=22).next_to(b, LEFT, buff=0.2).align_to([-1.4, 0, 0], RIGHT), b,
                            Text(f"{v:.2f}", font=MONO, font_size=22).next_to(b, RIGHT, buff=0.15)))
        self.at("rises")
        self.play(FadeIn(bars), run_time=0.5)
        s = Text("“My new phone is big. Beautiful design with fantastic functions.\nAnd I love that it comes with an Infinity dot tag…”",
                 font_size=20, color=GREEN_B, line_spacing=0.85).move_to([0, -0.9, 0])
        self.at("fluent")
        self.play(FadeIn(s), run_time=0.5)
        k = Text("KL from GPT-2: 3.8", font=MONO, font_size=24, color=YELLOW).move_to([0, -2.4, 0])
        self.at("divergence")
        self.play(FadeIn(k), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. compare
    def compare(self):
        self.section(7)
        self.clear_stage()
        hdr = VGroup(Text("", font_size=22), Text("PPO (β 0.5)", font_size=24, color=GOLD), Text("DPO", font_size=24, color=GREEN_B))
        rows = [("reward", "0.92", "1.08"), ("time", "≈ 19 min", "≈ 2 min"), ("reward model in the loop", "yes", "no"),
                ("value head", "yes", "no"), ("sampling while training", "yes", "no")]
        xs = (-3.2, 1.2, 4.2)
        for m, x in zip(hdr, xs):
            m.move_to([x, 2.2, 0])
        lines = VGroup(*[VGroup(*[Text(v, font_size=22, color=WHITE if j == 0 else (GOLD if j == 1 else GREEN_B)).move_to([xs[j], 1.4 - 0.65 * k, 0])
                                  for j, v in enumerate(r)]) for k, r in enumerate(rows)])
        self.at("compare")
        self.play(FadeIn(hdr), FadeIn(lines[0]), run_time=0.5)
        self.at("minutes")
        self.play(FadeIn(lines[1]), run_time=0.3)
        self.at("dpo")
        self.play(FadeIn(lines[2:], lag_ratio=0.2), run_time=0.6)
        n = Text("why many open models are tuned this way", font_size=24, color=YELLOW).move_to([0, -2.6, 0])
        self.at("simplicity")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=26)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("sigmoid")
        self.play(Create(hl), run_time=0.3)
        self.at("chosen")
        self.play(highlight(hl, code, 0), run_time=0.3)
        self.at("rejected")
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
