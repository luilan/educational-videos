"""Reinforcement Learning for LLMs, episode 5 — PPO: Small, Safe Steps.

Render from the repo root:  ./render.sh rl-llms r05
Numbers come from code/r05_ppo (ppo.py → results.json).
"""
import json
import statistics

from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from r05_script import NEXT, TAGLINE, TITLE
from rl_common import BAD, GOOD, MODEL_C, REWARD_C, study_footer
from voiced_scene import VoicedScene

SERIES_LABEL = "REINFORCEMENT LEARNING FOR LLMs  ·  EPISODE 5"
CODE_DIR = "code/r05_ppo"
R = json.load(open(f"{CODE_DIR}/results.json"))
RF_C, PPO_C, NC_C = GREY_A, MODEL_C, BAD
CODE = """ratio = torch.exp(logp_new - logp_old)          # per token
clipped = ratio.clamp(1 - 0.2, 1 + 0.2)
loss = -torch.min(ratio * adv, clipped * adv).mean()"""


def runs(method, lr):
    return [r for r in R["runs"] if r["method"] == method and abs(r["lr"] - lr) < 1e-9]


def finals(method, lr):
    return [r["curve"][-1][1] for r in runs(method, lr)]


def pct(x):
    return f"{100 * x:.0f}%"


class PPOVideo(VoicedScene):
    VIDEO = "r05"

    def construct(self):
        play_token_intro(self, TITLE, 5, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.danger()      # 2
        self.ratio()       # 3
        self.clip()        # 4
        self.value()       # 5
        self.experiment()  # 6
        self.too_big()     # 7
        self.lesson()      # 8
        self.code()        # 9
        self.outro()       # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def axes(self, pos, y_max=0.35, x_length=6.4, y_length=3.4):
        ax = Axes(x_range=[0, 400, 100], y_range=[0, y_max, 0.1], x_length=x_length, y_length=y_length, tips=False,
                  axis_config={"color": GREY_C})
        ax.move_to(pos)
        xl = VGroup(*[Text(f"{s}", font_size=16, color=GREY_B).next_to(ax.c2p(s, 0), DOWN, buff=0.12) for s in range(0, 401, 100)])
        yl = VGroup(*[Text(f"{int(v * 100)}%", font_size=16, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v in (0, 0.1, 0.2, 0.3)])
        cap = Text("batches of 256 answers · right on 1,000 new questions", font_size=16, color=GREY_B).next_to(xl, DOWN, buff=0.08)
        return ax, VGroup(ax, xl, yl, cap)

    def curve(self, ax, run, color, width=3):
        return VMobject(color=color, stroke_width=width).set_points_as_corners([ax.c2p(b, a) for b, a in run["curve"]])

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        gen = VGroup(RoundedRectangle(corner_radius=0.15, width=7.4, height=0.9, stroke_color=REWARD_C),
                     Text("generate 256 answers: the expensive part", font_size=21, color=REWARD_C)).move_to([0, 2.4, 0])
        gen[1].move_to(gen[0])
        self.at("expensive")
        self.play(FadeIn(gen), run_time=0.5)
        rf = VGroup(Text("REINFORCE", font_size=28, color=RF_C), Text("1 step, then throw the batch away", font_size=22, color=GREY_B)).arrange(DOWN, buff=0.15).move_to([-3.4, 0.4, 0])
        self.at("reinforce")
        self.play(FadeIn(rf), run_time=0.4)
        trash = Text("✕", font_size=40, color=BAD).next_to(rf, DOWN, buff=0.3)
        self.at("away")
        self.play(FadeIn(trash), run_time=0.3)
        pp = VGroup(Text("PPO", font_size=28, color=PPO_C), Text("proximal policy optimization", font_size=18, color=GREY_B),
                    Text("reuse the batch: 16 steps", font_size=22, color=GREY_B)).arrange(DOWN, buff=0.12).move_to([3.4, 0.4, 0])
        self.at("ppo")
        self.play(FadeIn(pp), run_time=0.4)
        loop = CurvedArrow(pp.get_bottom() + 0.2 * DOWN + 1.0 * LEFT, pp.get_bottom() + 0.2 * DOWN + 1.0 * RIGHT, angle=TAU / 3, color=PPO_C)
        self.play(Create(loop), run_time=0.4)
        s = Text("the trick: doing that safely", font_size=28, color=REWARD_C).move_to([0, -2.4, 0])
        self.at("safely")
        self.play(FadeIn(s), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. why reuse is dangerous
    def danger(self):
        self.section(2)
        self.clear_stage()
        t = Text("reusing a batch = more steps on the same answers", font_size=28).to_edge(UP, buff=0.7)
        self.at("reusing")
        self.play(FadeIn(t), run_time=0.4)
        ax, frame = self.axes([2.4, -0.4, 0])
        a = VGroup(*[self.curve(ax, r, RF_C) for r in runs("reinforce", 3e-4)])
        b = VGroup(*[self.curve(ax, r, BAD) for r in runs("reinforce", 1e-3)])
        la = Text(f"REINFORCE, step 0.0003: {pct(statistics.mean(finals('reinforce', 3e-4)))}", font_size=22, color=RF_C).move_to([-6.6, 0.8, 0], aligned_edge=LEFT)
        lb = Text(f"3× bigger steps: {pct(statistics.mean(finals('reinforce', 1e-3)))}", font_size=22, color=BAD).move_to([-6.6, 0.1, 0], aligned_edge=LEFT)
        self.at("big")
        self.play(FadeIn(frame), Create(a), FadeIn(la), run_time=0.8)
        self.at("three")
        self.play(Create(b), FadeIn(lb), run_time=0.8)
        start = DashedLine(ax.c2p(0, 0.139), ax.c2p(400, 0.139), color=GREY_B, stroke_width=2)
        sl = Text("start 14%", font_size=16, color=GREY_B).next_to(start, RIGHT, buff=0.1)
        self.at("below")
        self.play(Create(start), FadeIn(sl), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. the ratio
    def ratio(self):
        self.section(3)
        self.clear_stage()
        old = VGroup(RoundedRectangle(corner_radius=0.12, width=3.9, height=0.9, stroke_color=GREY_B), Text("model when sampled", font_size=22, color=GREY_B))
        old[1].move_to(old[0])
        new = VGroup(RoundedRectangle(corner_radius=0.12, width=3.9, height=0.9, stroke_color=PPO_C), Text("model now", font_size=22, color=PPO_C))
        new[1].move_to(new[0])
        VGroup(old, new).arrange(RIGHT, buff=2.0).move_to([0, 2.0, 0])
        ar = Arrow(old.get_right(), new.get_left(), buff=0.1, color=GREY_B)
        upd = Text("1 update later", font_size=18, color=GREY_B).next_to(ar, UP, buff=0.08)
        self.at("older")
        self.play(FadeIn(old), FadeIn(new), GrowArrow(ar), FadeIn(upd), run_time=0.6)
        q = Text("47+38=", font=MONO, font_size=30).move_to([-2.6, 0.3, 0])
        toks = VGroup(*[Text(c, font=MONO, font_size=30, color=REWARD_C) for c in "85."]).arrange(RIGHT, buff=0.25).next_to(q, RIGHT, buff=0.2)
        self.at("token")
        self.play(FadeIn(q), FadeIn(toks), run_time=0.4)
        f = VGroup(Text("ratio =", font_size=30),
                   VGroup(Text("p now (token)", font_size=26, color=PPO_C), Line(LEFT * 1.8, RIGHT * 1.8, color=WHITE),
                          Text("p when sampled (token)", font_size=26, color=GREY_B)).arrange(DOWN, buff=0.12)).arrange(RIGHT, buff=0.3).move_to([0, -1.3, 0])
        self.at("now")
        self.play(FadeIn(f), run_time=0.5)
        s = Text("push = ratio × advantage", font_size=28, color=REWARD_C).move_to([0, -2.8, 0])
        self.at("ratio")
        self.play(FadeIn(s), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. the clip
    def clip(self):
        self.section(4)
        self.clear_stage()
        h = Text("the clip: ratio kept within 0.8 – 1.2", font_size=30, color=REWARD_C).to_edge(UP, buff=0.6)
        self.at("clip")
        self.play(FadeIn(h), run_time=0.4)
        panels = []
        for i, (sign, title) in enumerate(((1, "advantage > 0: make it likelier"), (-1, "advantage < 0: make it rarer"))):
            ax = Axes(x_range=[0, 2, 0.5], y_range=[-2, 2, 1], x_length=4.6, y_length=3.0, tips=False, axis_config={"color": GREY_C})
            ax.move_to([-3.4 + 6.8 * i, -0.6, 0])
            lab = Text(title, font_size=20, color=GOOD if sign > 0 else BAD).next_to(ax, UP, buff=0.2)
            xl = VGroup(*[Text(f"{v:g}", font_size=16, color=GREY_B).next_to(ax.c2p(v, -2), DOWN, buff=0.1) for v in (0, 0.8, 1.2, 2)])
            cap = Text("ratio", font_size=16, color=GREY_B).next_to(xl, DOWN, buff=0.05)
            yl = Text("objective", font_size=16, color=GREY_B).rotate(PI / 2).next_to(ax, LEFT, buff=0.1)
            raw = ax.plot(lambda x: sign * x, x_range=[0, 2], color=GREY_B, stroke_width=2)
            raw = DashedVMobject(raw, num_dashes=30)
            if sign > 0:
                f = lambda x: min(x, 1.2)
            else:
                f = lambda x: -max(x, 0.8)
            clipped = ax.plot(f, x_range=[0, 2, 0.01], color=REWARD_C, stroke_width=4)
            band = Rectangle(width=ax.c2p(1.2, 0)[0] - ax.c2p(0.8, 0)[0], height=3.0, stroke_width=0, fill_color=GREY_D, fill_opacity=0.35).move_to(ax.c2p(1.0, 0))
            panels.append(VGroup(band, ax, lab, xl, cap, yl, raw, clipped))
        self.at("20")
        self.play(*[FadeIn(VGroup(p[0], p[1], p[2], p[3], p[4], p[5], p[6])) for p in panels], run_time=0.6)
        self.at("advantage")
        self.play(*[Create(p[7]) for p in panels], run_time=1.0)
        flat = Text("flat = no gradient: that token stops pushing", font_size=24, color=REWARD_C).move_to([0, -3.0, 0])
        self.at("stops")
        self.play(FadeIn(flat), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. the value head
    def value(self):
        self.section(5)
        self.clear_stage()
        q = Text("47+38=", font=MONO, font_size=30).move_to([-5.8, 0.8, 0])
        gpt = VGroup(RoundedRectangle(corner_radius=0.15, width=2.6, height=1.4, stroke_color=PPO_C, fill_color=PPO_C, fill_opacity=0.12),
                     Text("tiny_gpt", font_size=24, color=PPO_C))
        gpt[1].move_to(gpt[0])
        gpt.move_to([-2.6, 0.8, 0])
        vh = VGroup(RoundedRectangle(corner_radius=0.12, width=2.2, height=0.9, stroke_color=REWARD_C, fill_color=REWARD_C, fill_opacity=0.12),
                    Text("value head", font_size=22, color=REWARD_C))
        vh[1].move_to(vh[0])
        vh.move_to([1.6, 0.8, 0])
        out = Text("expected reward", font_size=22, color=REWARD_C).move_to([5.0, 0.8, 0])
        a1, a2, a3 = Arrow(q.get_right(), gpt.get_left(), buff=0.1), Arrow(gpt.get_right(), vh.get_left(), buff=0.1), Arrow(vh.get_right(), out.get_left(), buff=0.1)
        self.at("value")
        self.play(FadeIn(q), FadeIn(gpt), GrowArrow(a1), run_time=0.5)
        sub = Text("one linear layer", font_size=18, color=GREY_B).next_to(vh, DOWN, buff=0.15)
        self.at("small")
        self.play(GrowArrow(a2), FadeIn(vh), FadeIn(sub), run_time=0.5)
        hid = Text("reads the hidden state at '='", font_size=18, color=GREY_B).move_to([-0.5, 2.0, 0])
        self.at("reads")
        self.play(FadeIn(hid), run_time=0.3)
        self.at("predicts")
        self.play(GrowArrow(a3), FadeIn(out), run_time=0.4)
        adv = Text("advantage = reward − predicted reward", font_size=30, color=WHITE).move_to([0, -1.4, 0])
        l = Text("a learned baseline (episode 3)", font_size=22, color=GREY_B).next_to(adv, DOWN, buff=0.2)
        self.at("minus")
        self.play(FadeIn(adv), FadeIn(l), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 6. experiment
    def experiment(self):
        self.section(6)
        self.clear_stage()
        ax, frame = self.axes([2.6, 0.0, 0])
        h = Text("same 400 batches · 102,400 answers · 3 runs each", font_size=22, color=GREY_B).move_to([-6.6, 2.9, 0], aligned_edge=LEFT)
        self.at("400")
        self.play(FadeIn(frame), FadeIn(h), run_time=0.5)
        rf = VGroup(*[self.curve(ax, r, RF_C) for r in runs("reinforce", 3e-4)])
        lrf = Text(f"REINFORCE: {pct(statistics.mean(finals('reinforce', 3e-4)))}", font_size=24, color=RF_C).move_to([-6.6, 1.9, 0], aligned_edge=LEFT)
        self.at("reinforce")
        self.play(Create(rf), FadeIn(lrf), run_time=0.8)
        pp = VGroup(*[self.curve(ax, r, PPO_C, width=4) for r in runs("ppo", 1e-4)])
        lpp = VGroup(Text(f"PPO: {pct(statistics.mean(finals('ppo', 1e-4)))}", font_size=24, color=PPO_C),
                     Text("16 updates per batch, step 0.0001", font_size=18, color=GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([-6.6, 1.1, 0], aligned_edge=LEFT)
        self.at("ppo")
        self.play(Create(pp), FadeIn(lpp), run_time=1.0)
        lo = min(finals("ppo", 1e-4))
        hi = max(finals("reinforce", 3e-4))
        all3 = Text(f"worst PPO run {100 * lo:.1f}%\nbest REINFORCE run {100 * hi:.1f}%", font_size=18, color=PPO_C, line_spacing=0.9).move_to([-6.6, 0.2, 0], aligned_edge=LEFT)
        self.at("all")
        self.play(FadeIn(all3), run_time=0.4)
        nc = VGroup(*[DashedVMobject(self.curve(ax, r, NC_C), num_dashes=50) for r in runs("noclip", 1e-4)])
        lnc = Text("no clip: " + " · ".join(pct(x) for x in sorted(finals("noclip", 1e-4))), font_size=24, color=NC_C).move_to([-6.6, -0.6, 0], aligned_edge=LEFT)
        self.at("without")
        self.play(Create(nc), FadeIn(lnc), run_time=1.0)
        u = Text("unreliable", font_size=26, color=NC_C).next_to(lnc, DOWN, buff=0.2).align_to(lnc, LEFT)
        self.at("unreliable")
        self.play(FadeIn(u), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. still needs small steps
    def too_big(self):
        self.section(7)
        self.clear_stage()
        h = Text("16 updates per batch at REINFORCE's step size (0.0003)", font_size=26).to_edge(UP, buff=0.7)
        self.at("free")
        self.play(FadeIn(h), run_time=0.4)
        items = [("REINFORCE, 1 update", statistics.mean(finals("reinforce", 3e-4)), RF_C, None),
                 ("PPO", statistics.mean(finals("ppo", 3e-4)), PPO_C, "drop"),
                 ("PPO, no clip", statistics.mean(finals("noclip", 3e-4)), NC_C, "without")]
        base_y, unit = -2.0, 12.0
        base = Line([-5.5, base_y, 0], [5.5, base_y, 0], color=GREY_C)
        self.play(Create(base), run_time=0.3)
        for i, (lab, v, col, cue) in enumerate(items):
            x = -3.6 + 3.6 * i
            bar = Rectangle(width=1.4, height=max(0.04, v * unit), stroke_width=0, fill_color=col, fill_opacity=0.8).move_to([x, base_y, 0], aligned_edge=DOWN)
            t = Text(pct(v), font_size=28, color=col).next_to(bar, UP, buff=0.12)
            n = Text(lab, font_size=20, color=col).move_to([x, base_y - 0.4, 0])
            if cue:
                self.at(cue)
            self.play(GrowFromEdge(bar, DOWN), FadeIn(t), FadeIn(n), run_time=0.5)
        m = Text("the clip limits each token's push, not the size of every step", font_size=24, color=REWARD_C).move_to([0, 2.0, 0])
        self.at("limits")
        self.play(FadeIn(m), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. lesson
    def lesson(self):
        self.section(8)
        self.clear_stage()
        plus = Text("+  more learning per answer", font_size=30, color=GOOD).move_to([-1.0, 1.6, 0], aligned_edge=LEFT).shift(LEFT * 3)
        minus = Text("−  more computing per batch (16 backward passes)", font_size=26, color=BAD).next_to(plus, DOWN, buff=0.45).align_to(plus, LEFT)
        self.at("buys")
        self.play(FadeIn(plus), run_time=0.4)
        self.at("computing")
        self.play(FadeIn(minus), run_time=0.4)
        vm = Text("−  for a language model, the value head is usually\n    a second network as big as the model", font_size=26, color=BAD, line_spacing=0.9).next_to(minus, DOWN, buff=0.45).align_to(plus, LEFT)
        self.at("value")
        self.play(FadeIn(vm), run_time=0.5)
        nx = Text("next episode: drop the value model", font_size=28, color=REWARD_C).move_to([0, -2.4, 0])
        self.at("removes")
        self.play(FadeIn(nx), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.at("ratio")
        self.play(Create(hl), run_time=0.3)
        self.at("clipped")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("smaller")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10. outro
    def outro(self):
        self.section(10)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), FadeIn(study_footer()), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
