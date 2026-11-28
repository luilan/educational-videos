"""Reinforcement Learning for LLMs, episode 6 — GRPO: Grading on a Curve.

Render from the repo root:  ./render.sh rl-llms r06
Numbers come from code/r06_grpo (grpo.py, grpo.py extra → results.json).
"""
import json
import statistics

import torch
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from r06_script import NEXT, TAGLINE, TITLE
from rl_common import BAD, GOOD, MODEL_C, REWARD_C, grade_badge, study_footer
from voiced_scene import VoicedScene

SERIES_LABEL = "REINFORCEMENT LEARNING FOR LLMs  ·  EPISODE 6"
CODE_DIR = "code/r06_grpo"
R = json.load(open(f"{CODE_DIR}/results.json"))
PPO_C, GRPO_C = MODEL_C, TEAL_C
CODE = """g = r.view(n_questions, G)                  # one row per group
adv = g - g.mean(1, keepdim=True)           # minus the group average
adv = adv / (g.std(1, keepdim=True) + 1e-4) # divided by its spread
loss = clipped_ppo_loss(ratio, adv.view(-1)) + beta * kl"""


def runs(method, group=8):
    return [r for r in R["runs"] if r["method"] == method and r["group"] == group]


def mean_final(method, group=8):
    return statistics.mean(r["curve"][-1][1] for r in runs(method, group))


def empty_share(group):
    return statistics.mean(statistics.mean(s["zero_adv_groups"] for s in r["stats"]) for r in runs("grpo", group))


def pct(x):
    return f"{100 * x:.0f}%"


def model_box(label, color, w=2.3, h=1.0, dashed=False):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=color, fill_color=color, fill_opacity=0.12)
    if dashed:
        r = DashedVMobject(r, num_dashes=40)
    return VGroup(r, Text(label, font_size=20, color=color).move_to(r))


class GRPOVideo(VoicedScene):
    VIDEO = "r06"

    def construct(self):
        play_token_intro(self, TITLE, 6, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.how()         # 2
        self.memory()      # 3
        self.experiment()  # 4
        self.empty()       # 5
        self.leash()       # 6
        self.dynamic()     # 7
        self.lesson()      # 8
        self.code()        # 9
        self.outro()       # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def bars(self, items, base_y=-2.0, unit=12.0, width=1.2, gap=2.3, x0=None):
        """items: (label, value, color). Returns groups (bar, value text, label)."""
        n = len(items)
        x0 = -(n - 1) * gap / 2 if x0 is None else x0
        out = VGroup()
        for i, (lab, v, col) in enumerate(items):
            x = x0 + i * gap
            bar = Rectangle(width=width, height=max(0.04, v * unit), stroke_width=0, fill_color=col, fill_opacity=0.8).move_to([x, base_y, 0], aligned_edge=DOWN)
            t = Text(pct(v), font_size=26, color=col).next_to(bar, UP, buff=0.1)
            l = Text(lab, font_size=18, color=col, line_spacing=0.8).next_to([x, base_y, 0], DOWN, buff=0.2)
            out.add(VGroup(bar, t, l))
        return out

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        pol = model_box("policy (the LLM)", PPO_C, w=3.2, h=1.6).move_to([-2.2, 0.8, 0])
        val = model_box("value model", REWARD_C, w=3.2, h=1.6).move_to([2.2, 0.8, 0])
        lab = Text("PPO", font_size=30, color=PPO_C).move_to([0, 2.6, 0])
        self.at("value")
        self.play(FadeIn(lab), FadeIn(pol), run_time=0.5)
        same = Text("as big as the model itself", font_size=22, color=REWARD_C).next_to(val, DOWN, buff=0.25)
        self.at("second")
        self.play(FadeIn(val), FadeIn(same), run_time=0.5)
        cross = Cross(val, stroke_color=BAD, stroke_width=6)
        ds = Text("DeepSeek, 2024: GRPO", font_size=30, color=GRPO_C).move_to([0, -1.6, 0])
        self.at("dropped")
        self.play(Create(cross), FadeIn(ds), run_time=0.5)
        c = Text("grade each answer on a curve", font_size=28, color=GREY_A).next_to(ds, DOWN, buff=0.3)
        self.at("curve")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. how
    def how(self):
        self.section(2)
        self.clear_stage()
        q = Text("47+38=", font=MONO, font_size=30).move_to([-5.4, 2.2, 0])
        rewards = [1, 0, 1, 0, 0, 1, 0, 0]
        badges = VGroup(*[grade_badge(r, font_size=20).scale(0.8) for r in rewards]).arrange(RIGHT, buff=0.25).next_to(q, RIGHT, buff=0.5)
        gl = Text("a group of 8 answers", font_size=20, color=GREY_B).next_to(badges, UP, buff=0.15)
        self.at("group")
        self.play(FadeIn(q), LaggedStart(*[FadeIn(b) for b in badges], lag_ratio=0.08), FadeIn(gl), run_time=0.8)
        f = VGroup(Text("advantage =", font_size=28), Text("(reward − group average)", font_size=28, color=REWARD_C),
                   Text("÷ group spread", font_size=28, color=GRPO_C)).arrange(RIGHT, buff=0.2).move_to([0, 0.8, 0])
        self.at("average")
        self.play(FadeIn(f[:2]), run_time=0.5)
        self.at("spread")
        self.play(FadeIn(f[2]), run_time=0.4)
        r = torch.tensor(rewards, dtype=torch.float)
        adv = ((r - r.mean()) / (r.std() + 1e-4)).tolist()
        zero = Line([-5.0, -1.4, 0], [5.0, -1.4, 0], color=GREY_C)
        bars = VGroup()
        for b, a in zip(badges, adv):
            col = GOOD if a > 0 else BAD
            bar = Rectangle(width=0.5, height=abs(a) * 1.0, stroke_width=0, fill_color=col, fill_opacity=0.85)
            bar.move_to([b.get_x(), -1.4, 0], aligned_edge=DOWN if a > 0 else UP)
            bars.add(bar)
        self.at("three")
        self.play(Create(zero), run_time=0.3)
        vals = VGroup(Text(f"right +{max(adv):.2f}", font_size=24, color=GOOD), Text(f"wrong {min(adv):.2f}", font_size=24, color=BAD)).arrange(RIGHT, buff=1.0).move_to([0, -3.0, 0])
        self.play(*[GrowFromEdge(b, DOWN if a > 0 else UP) for b, a in zip(bars, adv)], run_time=0.6)
        self.at("1")
        self.play(FadeIn(vals[0]), run_time=0.3)
        self.at("minus")
        self.play(FadeIn(vals[1]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. what stays, and memory
    def memory(self):
        self.section(3)
        self.clear_stage()
        keep = Text("kept from PPO: the ratio, the clip · plus a KL leash in the loss", font_size=24, color=GREY_A).to_edge(UP, buff=0.7)
        self.at("ratio")
        self.play(FadeIn(keep), run_time=0.5)
        row1 = VGroup(Text("PPO (RLHF)", font_size=24, color=PPO_C),
                      model_box("policy", PPO_C), model_box("reference", GREY_B), model_box("reward model", REWARD_C), model_box("value model", REWARD_C)
                      ).arrange(RIGHT, buff=0.3).move_to([0, 0.6, 0])
        row2 = VGroup(Text("GRPO", font_size=24, color=GRPO_C),
                      model_box("policy", PPO_C), model_box("reference", GREY_B), model_box("checker program", GREY_B, w=2.9, dashed=True)
                      ).arrange(RIGHT, buff=0.3)
        row2.move_to([0, -1.2, 0]).align_to(row1, LEFT)
        self.play(FadeIn(row1), run_time=0.5)
        self.at("program")
        self.play(FadeIn(row2), run_time=0.5)
        n = VGroup(Text("4 networks", font_size=26, color=PPO_C).next_to(row1, DOWN, buff=0.15).align_to(row1, RIGHT),
                   Text("2 networks + a program", font_size=26, color=GRPO_C).next_to(row2, DOWN, buff=0.15).align_to(row1, RIGHT))
        self.at("two")
        self.play(FadeIn(n[1]), run_time=0.3)
        self.at("four")
        self.play(FadeIn(n[0]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4. experiment
    def experiment(self):
        self.section(4)
        self.clear_stage()
        h = Text("same budget as episode 5: 400 batches of 256 answers · 3 runs each", font_size=22, color=GREY_B).to_edge(UP, buff=0.6)
        self.at("budget")
        self.play(FadeIn(h), run_time=0.4)
        items = [("PPO", mean_final("ppo"), PPO_C)] + [(f"GRPO\nG = {g}", mean_final("grpo", g), GRPO_C) for g in (2, 4, 8, 16)]
        b = self.bars(items)
        start = DashedLine([-6.0, -2.0 + 12 * 0.139, 0], [6.0, -2.0 + 12 * 0.139, 0], color=GREY_B, stroke_width=2)
        sl = Text("start 14%", font_size=16, color=GREY_B).next_to(h, DOWN, buff=0.15)
        self.at("28")
        self.play(GrowFromEdge(b[0][0], DOWN), FadeIn(b[0][1:]), Create(start), FadeIn(sl), run_time=0.5)
        self.at("grpo")
        self.play(*[FadeIn(x[2]) for x in b[1:]], run_time=0.3)
        self.at("16")
        self.play(*[AnimationGroup(GrowFromEdge(x[0], DOWN), FadeIn(x[1])) for x in b[1:]], run_time=0.7)
        w = Text("on our task, clearly worse", font_size=28, color=BAD).move_to([0, 2.0, 0])
        self.at("worse")
        self.play(FadeIn(w), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. empty groups
    def empty(self):
        self.section(5)
        self.clear_stage()
        h = Text("empty groups", font_size=32, color=REWARD_C).to_edge(UP, buff=0.6)
        self.at("empty")
        self.play(FadeIn(h), run_time=0.4)
        rows = VGroup()
        for i, rw in enumerate(([1] * 8, [0] * 8, [1, 0, 0, 1, 0, 0, 0, 0])):
            badges = VGroup(*[grade_badge(r, font_size=16).scale(0.6) for r in rw]).arrange(RIGHT, buff=0.12)
            note = Text("all agree → every advantage 0" if i < 2 else "mixed → it learns", font_size=20,
                        color=GREY_B if i < 2 else GOOD).next_to(badges, RIGHT, buff=0.4)
            rows.add(VGroup(badges, note))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([-0.6, 1.0, 0])
        self.at("zero")
        self.play(FadeIn(rows[0]), FadeIn(rows[1]), run_time=0.5)
        self.at("nothing")
        self.play(FadeIn(rows[2]), run_time=0.4)
        tab = VGroup()
        for j, g in enumerate((2, 4, 8, 16)):
            col = VGroup(Text(f"G = {g}", font_size=22, color=GRPO_C), Text(pct(empty_share(g)), font_size=30, color=GREY_A)).arrange(DOWN, buff=0.15)
            tab.add(col)
        tab.arrange(RIGHT, buff=1.2).move_to([0.6, -1.7, 0])
        tl = Text("groups that\nteach nothing:", font_size=20, color=GREY_B, line_spacing=0.85).next_to(tab, LEFT, buff=0.6)
        self.at("consistent")
        self.play(FadeIn(tl), FadeIn(tab[2]), run_time=0.4)
        self.at("three")
        self.play(Indicate(tab[2][1], color=REWARD_C), run_time=0.6)
        self.at("70")
        self.play(FadeIn(tab[0]), FadeIn(tab[1]), FadeIn(tab[3]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. the leash
    def leash(self):
        self.section(6)
        self.clear_stage()
        h = Text("the KL leash (β = 0.04, DeepSeekMath's value) · groups of 8", font_size=24, color=GREY_A).to_edge(UP, buff=0.6)
        self.at("leash")
        self.play(FadeIn(h), run_time=0.4)
        b = self.bars([("GRPO\nwith KL", mean_final("grpo"), GRPO_C), ("GRPO\nno KL", mean_final("grpo_nokl"), GRPO_C), ("PPO", mean_final("ppo"), PPO_C)], gap=3.0)
        self.play(GrowFromEdge(b[0][0], DOWN), FadeIn(b[0][1:]), FadeIn(b[2][2]), run_time=0.5)
        self.at("remove")
        self.play(FadeIn(b[1][2]), run_time=0.3)
        self.at("26")
        self.play(GrowFromEdge(b[1][0], DOWN), FadeIn(b[1][1]), run_time=0.5)
        self.at("close")
        self.play(GrowFromEdge(b[2][0], DOWN), FadeIn(b[2][1]), run_time=0.4)
        g = Text("most of the gap was the leash", font_size=26, color=REWARD_C).move_to([0, 2.1, 0])
        self.at("gap")
        self.play(FadeIn(g), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. dynamic sampling
    def dynamic(self):
        self.section(7)
        self.clear_stage()
        h = Text("dynamic sampling (DAPO recipe, 2025)", font_size=28, color=GRPO_C).to_edge(UP, buff=0.6)
        self.at("newer")
        self.play(FadeIn(h), run_time=0.4)
        steps = Text("drop groups that all agree → sample more questions → until 256 answers", font_size=22, color=GREY_A).next_to(h, DOWN, buff=0.3)
        self.at("throw")
        self.play(FadeIn(steps), run_time=0.4)
        b = self.bars([("GRPO", mean_final("grpo"), GRPO_C), ("GRPO\ndynamic sampling", mean_final("grpo_dyn"), GRPO_C)], gap=4.0, x0=-4.0)
        self.at("percent")
        self.play(*[AnimationGroup(GrowFromEdge(x[0], DOWN), FadeIn(x[1:])) for x in b], run_time=0.6)
        gen = [r["generated"] for r in runs("grpo_dyn")]
        ratio = statistics.mean(gen) / 102_400
        g = VGroup(Text("answers generated", font_size=22, color=GREY_B),
                   Text("102,400", font_size=28, color=GREY_A),
                   Text(f"{min(gen):,} – {max(gen):,}", font_size=28, color=BAD),
                   Text(f"{ratio:.1f}×", font_size=34, color=BAD)).arrange(DOWN, buff=0.25).move_to([3.4, 0.0, 0])
        self.at("four")
        self.play(FadeIn(g), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 8. lesson
    def lesson(self):
        self.section(8)
        self.clear_stage()
        a = Text("GRPO's win: cost, not learning per answer", font_size=30, color=REWARD_C).to_edge(UP, buff=0.8)
        self.at("cost")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("no value network → RL on huge models gets much cheaper", font_size=26, color=GOOD).move_to([0, 1.2, 0])
        self.at("cheaper")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("weak spots: empty groups, the strength of the leash", font_size=26, color=BAD).move_to([0, 0.2, 0])
        self.at("weak")
        self.play(FadeIn(c), run_time=0.4)
        d = Text("later recipes tune exactly these; DAPO drops the KL term", font_size=24, color=GREY_A).move_to([0, -0.8, 0])
        self.at("drops")
        self.play(FadeIn(d), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.at("reshape")
        self.play(Create(hl), run_time=0.3)
        self.at("subtract")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("divide")
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
