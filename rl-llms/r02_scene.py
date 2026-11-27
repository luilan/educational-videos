"""Reinforcement Learning for LLMs, episode 2 — REINFORCE.

Render from the repo root:  ./render.sh rl-llms r02
Numbers come from code/r02_reinforce (reinforce.py → results.json).
"""
import json

import numpy as np
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from r02_script import NEXT, TAGLINE, TITLE
from rl_common import BAD, GOOD, MODEL_C, REWARD_C, char_row, grade_badge, study_footer
from voiced_scene import VoicedScene

SERIES_LABEL = "REINFORCEMENT LEARNING FOR LLMs  ·  EPISODE 2"
CODE_DIR = "code/r02_reinforce"
R = json.load(open(f"{CODE_DIR}/results.json"))
CODE = """idx, answers, lengths, qs = sample(model, questions, n=8)
r = torch.tensor([reward(a, b, x) for (a, b), x in zip(qs, answers)])
logp = answer_logprob(model, idx, lengths)
loss = -(r * logp).mean()                  # REINFORCE
loss.backward(); opt.step()"""


def pct(x):
    return f"{100 * x:.0f}%"


def tally_bars(items, right, x0, y0, unit=0.075, height=0.42, gap=0.62, font=24):
    """Rows of (answer, count): answer tokens, a bar, and '× n'; green if it is the right answer."""
    rows = VGroup()
    for i, (ans, n) in enumerate(items):
        y = y0 - gap * i
        a = char_row(ans, font_size=font).move_to([x0, y, 0], aligned_edge=RIGHT)
        bar = Rectangle(width=max(0.04, unit * n), height=height, stroke_width=0,
                        fill_color=GOOD if ans == right else GREY_C, fill_opacity=0.85)
        bar.move_to([x0 + 0.25, y, 0], aligned_edge=LEFT)
        c = Text(f"× {n}", font_size=font - 2).next_to(bar, RIGHT, buff=0.15)
        rows.add(VGroup(a, bar, c))
    return rows


class ReinforceVideo(VoicedScene):
    VIDEO = "r02"

    def construct(self):
        play_token_intro(self, TITLE, 2, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.rule()        # 2
        self.log_trick()   # 3
        self.one_step()    # 4
        self.training()    # 5
        self.q27()         # 6
        self.q47()         # 7
        self.catch()       # 8
        self.code()        # 9
        self.outro()       # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        acc = VGroup(Text("episode 1:", font_size=24, color=GREY_B),
                     Text(f"{pct(R['acc_before'])} of new questions right", font_size=30, color=BAD)).arrange(RIGHT, buff=0.3)
        acc.move_to([0, 2.2, 0])
        self.at("14")
        self.play(FadeIn(acc), run_time=0.5)
        tries = VGroup(*[VGroup(char_row(a, font_size=26), grade_badge(r)).arrange(RIGHT, buff=0.25)
                         for a, r in [("48.", 0), ("42.", 1), ("49.", 0), ("42.", 1)]]).arrange(RIGHT, buff=0.5)
        tries.move_to([0, 0.7, 0])
        self.at("graded")
        self.play(LaggedStart(*[FadeIn(t) for t in tries], lag_ratio=0.2), run_time=0.9)
        name = Text("REINFORCE", font_size=56, color=REWARD_C).move_to([0, -1.1, 0])
        self.at("reinforce")
        self.play(FadeIn(name, scale=1.1), run_time=0.6)
        cite = Text("Ronald J. Williams, 1992", font_size=24, color=GREY_B).next_to(name, DOWN, buff=0.3)
        self.at("1992")
        self.play(FadeIn(cite), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. the rule
    def rule(self):
        self.section(2)
        self.clear_stage()
        f = VGroup(Text("for every answer:", font_size=26, color=GREY_B),
                   VGroup(Text("push up", font_size=32), Text("reward", font_size=32, color=REWARD_C),
                          Text("×", font_size=32), Text("log p(answer)", font=MONO, font_size=30, color=MODEL_C)).arrange(RIGHT, buff=0.25)
                   ).arrange(DOWN, buff=0.3).move_to([0, 2.0, 0])
        self.at("line")
        self.play(FadeIn(f[0]), run_time=0.4)
        self.at("log")
        self.play(FadeIn(f[1][3]), run_time=0.4)
        self.at("reward")
        self.play(FadeIn(f[1][1:3]), run_time=0.4)
        self.at("nudge")
        self.play(FadeIn(f[1][0]), run_time=0.4)
        good = VGroup(char_row("27+15=42.", font_size=28), grade_badge(1),
                      Arrow(ORIGIN, UP * 1.0, buff=0, color=GOOD, stroke_width=6), Text("more likely", font_size=24, color=GOOD))
        good[0:2].arrange(RIGHT, buff=0.25)
        good[2].next_to(good[1], RIGHT, buff=0.4)
        good[3].next_to(good[2], RIGHT, buff=0.2)
        good.move_to([0, -0.2, 0])
        bad = VGroup(char_row("27+15=48.", font_size=28), grade_badge(0),
                     Text("0 × anything = no push", font_size=24, color=GREY_B))
        bad.arrange(RIGHT, buff=0.3).move_to([0, -1.7, 0])
        self.at("one")
        self.play(FadeIn(good[:2]), GrowArrow(good[2]), FadeIn(good[3]), run_time=0.7)
        self.at("zero")
        self.play(FadeIn(bad), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 3. the log trick
    def log_trick(self):
        self.section(3)
        self.clear_stage()
        goal = VGroup(Text("goal: a higher average reward", font_size=28),
                      Text("average reward = Σ p(answer) · reward(answer)", font=MONO, font_size=24, color=GREY_A)).arrange(DOWN, buff=0.25)
        goal.move_to([0, 2.3, 0])
        self.at("average")
        self.play(FadeIn(goal), run_time=0.6)
        no = VGroup(Text("no gradient through the checker", font_size=24, color=BAD),
                    Text("no gradient through a random choice", font_size=24, color=BAD)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        no.move_to([0, 0.8, 0])
        self.at("checker")
        self.play(FadeIn(no[0]), run_time=0.4)
        self.at("random")
        self.play(FadeIn(no[1]), run_time=0.4)
        eq = VGroup(Text("∇ average reward", font=MONO, font_size=26),
                    Text("= Σ p(answer) · reward · ∇ log p(answer)", font=MONO, font_size=26),
                    Text("≈ mean over samples of  reward · ∇ log p(sample)", font=MONO, font_size=26, color=REWARD_C)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([0, -1.2, 0])
        self.at("trick")
        self.play(FadeIn(eq[0]), run_time=0.4)
        self.at("times")
        self.play(FadeIn(eq[1]), run_time=0.5)
        self.at("samples")
        self.play(FadeIn(eq[2]), run_time=0.5)
        note = Text("(because ∇p = p · ∇ log p)", font_size=20, color=GREY_B).next_to(eq, DOWN, buff=0.3)
        self.play(FadeIn(note), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4. one step
    def one_step(self):
        self.section(4)
        self.clear_stage()
        fb = R["first_batch"]
        rng = np.random.default_rng(4)
        right = set(rng.choice(fb["answers"], fb["rewarded"], replace=False).tolist())
        cells = VGroup(*[Square(side_length=0.22, stroke_width=0, fill_color=GREY_D, fill_opacity=1) for _ in range(fb["answers"])])
        cells.arrange_in_grid(rows=8, cols=32, buff=0.06).move_to([0, 0.6, 0])
        rows_lab = Text("8 tries", font_size=20, color=GREY_B).next_to(cells, LEFT, buff=0.2)
        cols_lab = Text("32 questions", font_size=20, color=GREY_B).next_to(cells, UP, buff=0.2)
        self.at("32")
        self.play(FadeIn(cols_lab), FadeIn(cells, lag_ratio=0.002), run_time=0.8)
        self.at("8")
        self.play(FadeIn(rows_lab), run_time=0.3)
        n = Text(f"{fb['answers']} answers", font_size=28).move_to([-3.0, -1.4, 0])
        self.at("256")
        self.play(FadeIn(n), run_time=0.3)
        greens = [cells[i] for i in sorted(right)]
        k = Text(f"{fb['rewarded']} right", font_size=28, color=GOOD).move_to([1.2, -1.4, 0])
        self.at("53")
        self.play(*[c.animate.set_fill(GOOD) for c in greens], FadeIn(k), run_time=0.6)
        up = Text("↑ pushed up", font_size=26, color=GOOD).next_to(k, RIGHT, buff=0.4)
        self.at("pushed")
        self.play(FadeIn(up, shift=0.15 * UP), run_time=0.4)
        down = Text("probabilities add up to 1, so every other answer to that question goes down a little",
                    font_size=22, color=GREY_B).move_to([0, -2.5, 0])
        self.at("add")
        self.play(FadeIn(down), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5. training curve
    def curve_axes(self):
        ax = Axes(x_range=[0, R["steps"], 500], y_range=[0, 1, 0.25], x_length=9.0, y_length=4.0, tips=False,
                  axis_config={"color": GREY_C, "include_numbers": False})
        ax.move_to([0.3, -0.2, 0])
        xl = VGroup(*[Text(f"{s:,}", font_size=18, color=GREY_B).next_to(ax.c2p(s, 0), DOWN, buff=0.15) for s in range(0, R["steps"] + 1, 500)])
        yl = VGroup(*[Text(f"{int(v * 100)}%", font_size=18, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.15) for v in (0, 0.25, 0.5, 0.75, 1.0)])
        cap = Text("steps", font_size=18, color=GREY_B).next_to(xl, DOWN, buff=0.1)
        ycap = Text("right on 1,000 new questions", font_size=20, color=GREY_B).next_to(ax, UP, buff=0.2).align_to(ax, LEFT)
        return ax, VGroup(ax, xl, yl, cap, ycap)

    def training(self):
        self.section(5)
        self.clear_stage()
        ax, frame = self.curve_axes()
        self.at("repeat")
        self.play(FadeIn(frame), run_time=0.6)
        pts = [ax.c2p(s, a) for s, a, _ in R["curve"]]
        line = VMobject(color=REWARD_C, stroke_width=4).set_points_as_corners(pts)
        self.at("minute")
        self.play(Create(line), run_time=2.0, rate_func=linear)
        self.curve = (ax, frame, line)
        a0 = Text(pct(R["acc_before"]), font_size=26, color=BAD).next_to(pts[0], UR, buff=0.1)
        a1 = Text(pct(R["acc_after"]), font_size=30, color=GOOD).next_to(pts[-1], UP, buff=0.15)
        self.at("14")
        self.play(FadeIn(a0), run_time=0.3)
        self.at("73")
        self.play(FadeIn(a1), run_time=0.4)
        g = Text("no answers shown, only grades", font_size=24, color=REWARD_C).move_to([2.6, -1.4, 0])
        self.at("grades")
        self.play(FadeIn(g), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. 27 + 15
    def q27(self):
        self.section(6)
        self.clear_stage()
        q = char_row("27+15=", font_size=32).move_to([0, 2.6, 0])
        self.at("27")
        self.play(FadeIn(q), run_time=0.4)
        b = R["watch_before"]["27+15="]
        a = R["watch_after"]["27+15="]
        lb = Text("before", font_size=24, color=GREY_B).move_to([-3.4, 1.6, 0])
        la = Text("after", font_size=24, color=GREY_B).move_to([3.2, 1.6, 0])
        before = tally_bars(b, "42.", -4.4, 0.8, unit=0.055)
        after = tally_bars(a, "42.", 2.2, 0.8, unit=0.055)
        self.at("before")
        self.play(FadeIn(lb), FadeIn(before), run_time=0.6)
        self.at("after")
        self.play(FadeIn(la), FadeIn(after), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 7. 47 + 38
    def q47(self):
        self.section(7)
        self.clear_stage()
        q = char_row("47+38=", font_size=32).move_to([0, 2.7, 0])
        self.at("47")
        self.play(FadeIn(q), run_time=0.4)
        b = R["watch_before"]["47+38="]
        a = R["watch_after"]["47+38="]
        lb = Text("before (64 tries)", font_size=22, color=GREY_B).move_to([-3.4, 1.9, 0])
        before = tally_bars(b, "85.", -4.4, 1.2, unit=0.055, gap=0.55, font=22)
        none = Text("85. : 0 times", font_size=24, color=BAD).next_to(before, DOWN, buff=0.3).align_to(before, LEFT)
        self.play(FadeIn(lb), FadeIn(before), run_time=0.6)
        self.at("85")
        self.play(FadeIn(none), run_time=0.4)
        la = Text("after", font_size=22, color=GREY_B).move_to([3.2, 1.9, 0])
        after = tally_bars(a, "85.", 2.2, 1.2, unit=0.055, gap=0.55, font=22)
        self.at("58")
        self.play(FadeIn(la), FadeIn(after), run_time=0.6)
        c = Text("carrying, learned on other questions, carries over", font_size=24, color=REWARD_C).move_to([0, -1.6, 0])
        self.at("carry")
        self.play(FadeIn(c), run_time=0.5)
        n = VGroup(Text(f"never right before: {R['never_right_before']} questions", font_size=24),
                   Text(f"now right: {R['never_now_right_greedy']}", font_size=24, color=GOOD)).arrange(RIGHT, buff=0.6).move_to([0, -2.5, 0])
        self.at("644")
        self.play(FadeIn(n[0]), run_time=0.4)
        self.at("445")
        self.play(FadeIn(n[1]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. the catch
    def catch(self):
        self.section(8)
        self.clear_stage()
        ax, frame = self.curve_axes()
        pts = [ax.c2p(s, a) for s, a, _ in R["curve"]]
        line = VMobject(color=REWARD_C, stroke_width=4).set_points_as_corners(pts)
        VGroup(frame, line).scale(0.72).move_to([0, 1.1, 0])
        self.play(FadeIn(frame), FadeIn(line), run_time=0.4)
        flat = Ellipse(width=2.1, height=0.7, color=BAD).move_to(ax.c2p(1800, R["acc_after"]))
        self.at("flattens")
        self.play(Create(flat), run_time=0.6)
        t = VGroup(Text("reward 0 → teaches nothing directly", font_size=24, color=GREY_A),
                   Text("mostly right → almost everything pushed up", font_size=24, color=GREY_A),
                   Text("→ a noisy signal", font_size=24, color=BAD)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        t.move_to([0, -2.3, 0])
        self.at("zero")
        self.play(FadeIn(t[0]), run_time=0.4)
        self.at("everything")
        self.play(FadeIn(t[1]), run_time=0.4)
        self.at("noisy")
        self.play(FadeIn(t[2]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=21)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.play(Create(hl), run_time=0.2)
        self.at("loss")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.at("averaged")
        self.play(Indicate(code.code_lines[3], color=REWARD_C), run_time=0.6)
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
