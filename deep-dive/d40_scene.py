"""How LLMs Work: Deep Dive, episode 40 — Reinforcement Learning for Reasoning.

Render from the repo root:  ./render.sh deep-dive d40
Every number on screen comes from code/d40_rl_reasoning/grpo.py (Qwen2.5-0.5B-Instruct, GRPO with a verifiable reward:
30 steps × 4 questions × 8 samples, lr 2e-6, KL beta 0.04; evaluated greedily on 100 held-out questions).
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from d40_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 40"
REWARD = [0.25, 0.62, 0.84, 0.69, 0.94, 0.81, 0.91, 1.00, 0.94, 0.88, 0.88, 0.94, 0.91, 0.97, 0.72, 1.00, 0.97, 0.88,
          0.66, 0.97, 0.97, 0.94, 0.91, 1.00, 1.00, 0.97, 0.75, 0.84, 0.97, 0.78]
CODE = """mean = r.mean(1, keepdim=True)   # r: one row per group
std = r.std(1, keepdim=True)
adv = (r - mean) / (std + 1e-4)
loss = -adv * logp + beta * kl"""


def chip(text, color, w=None, fs=20):
    t = Text(text, font_size=fs)
    r = RoundedRectangle(width=w or t.width + 0.4, height=t.height + 0.35, corner_radius=0.1, color=color, fill_opacity=0.2)
    return VGroup(r, t.move_to(r))


class RLReasoningVideo(VoicedScene):
    VIDEO = "d40"

    def construct(self):
        play_token_intro(self, TITLE, 40, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.task()          # 2
        self.grpo()          # 3
        self.signal()        # 4
        self.before()        # 5
        self.training()      # 6
        self.after()         # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        a = chip("preferences: people judge", GREY_B, fs=22)
        b = chip("math, code, puzzles: a program checks", GREEN_C, fs=22)
        VGroup(a, b).arrange(RIGHT, buff=0.6).move_to([0, 0.8, 0])
        self.at("preferences")
        self.play(FadeIn(a), run_time=0.5)
        self.at("program")
        self.play(FadeIn(b), run_time=0.5)
        c = Text("reinforcement learning on the check alone → reasoning models (DeepSeek-R1)", font_size=24,
                 color=YELLOW).move_to([0, -1.2, 0])
        self.at("reinforcement")
        self.play(FadeIn(c), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. task
    def task(self):
        self.section(2)
        self.clear_stage()
        m = Text("Qwen2.5-0.5B-Instruct", font_size=28, color=PURPLE_B).move_to([0, 2.4, 0])
        self.at("model")
        self.play(FadeIn(m), run_time=0.4)
        q = chip("What is 37 × 2 + 48?  Think step by step, then end with 'Answer: <number>'.", BLUE_C, fs=22).move_to([0, 1.0, 0])
        self.at("questions")
        self.play(FadeIn(q), run_time=0.5)
        r = VGroup(Text("reward = 1  if  “Answer: 122”", font=MONO, font_size=24, color=GREEN_B),
                   Text("reward = 0  otherwise", font=MONO, font_size=24, color=RED_B)).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        r.move_to([0, -0.6, 0])
        self.at("reward")
        self.play(FadeIn(r), run_time=0.5)
        n = Text("no reward model · no human labels", font_size=24, color=YELLOW).move_to([0, -2.3, 0])
        self.at("labels")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. GRPO
    def grpo(self):
        self.section(3)
        self.clear_stage()
        h = Text("GRPO: group relative policy optimization", font_size=28).to_edge(UP, buff=0.6)
        self.at("grpo")
        self.play(FadeIn(h), run_time=0.4)
        rewards = [1, 0, 1, 1, 0, 0, 1, 0]
        boxes = VGroup(*[VGroup(Square(0.8, color=GREEN_C if v else RED_C, fill_opacity=0.25),
                                Text(str(v), font=MONO, font_size=26)) for v in rewards]).arrange(RIGHT, buff=0.25)
        for b in boxes:
            b[1].move_to(b[0])
        boxes.move_to([0, 1.1, 0])
        lab = Text("8 answers to one question, rewards", font_size=20, color=GREY_B).next_to(boxes, UP, buff=0.2)
        self.at("sample")
        self.play(FadeIn(lab), FadeIn(boxes, lag_ratio=0.1), run_time=0.8)
        f = Text("advantage = (reward − group mean) / group spread", font=MONO, font_size=24, color=YELLOW).move_to([0, -0.3, 0])
        self.at("advantage")
        self.play(FadeIn(f), run_time=0.5)
        advs = VGroup(*[Text(f"{(v - 0.5) / 0.53:+.1f}", font=MONO, font_size=20, color=GREEN_B if v else RED_B).next_to(b, DOWN, buff=0.15)
                        for v, b in zip(rewards, boxes)])
        self.play(FadeIn(advs), run_time=0.4)
        up = Text("better than its siblings: more likely", font_size=22, color=GREEN_B).move_to([0, -1.4, 0])
        dn = Text("worse: less likely", font_size=22, color=RED_B).move_to([0, -1.95, 0])
        self.at("better")
        self.play(FadeIn(up), run_time=0.3)
        self.at("worse")
        self.play(FadeIn(dn), run_time=0.3)
        v = Text("no value head: the group is the baseline", font_size=24, color=YELLOW).move_to([0, -2.8, 0])
        self.at("value")
        self.play(FadeIn(v), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4. signal
    def signal(self):
        self.section(4)
        self.clear_stage()

        def row(vals, y, label):
            g = VGroup(*[VGroup(Square(0.6, color=GREEN_C if v else RED_C, fill_opacity=0.25),
                                Text(str(v), font=MONO, font_size=20)) for v in vals]).arrange(RIGHT, buff=0.15)
            for b in g:
                b[1].move_to(b[0])
            g.move_to([-1.0, y, 0])
            return VGroup(g, Text(label, font_size=22).next_to(g, RIGHT, buff=0.5))

        a = row([1] * 8, 1.8, "all right: advantages 0")
        b = row([0] * 8, 0.8, "all wrong: advantages 0")
        c = row([1, 0, 0, 1, 1, 0, 1, 1], -0.2, "mixed: a signal")
        c[1].set_color(GREEN_B)
        self.at("all")
        self.play(FadeIn(a), run_time=0.4)
        self.at("wrong")
        self.play(FadeIn(b), run_time=0.4)
        n = Text("teaches nothing", font_size=24, color=RED_B).move_to([0, -1.2, 0])
        self.at("nothing")
        self.play(FadeIn(n), run_time=0.3)
        self.at("mixed")
        self.play(FadeIn(c), FadeOut(n), run_time=0.4)
        k = Text("+ a small KL penalty (β 0.04) keeps it near the start", font_size=22, color=YELLOW).move_to([0, -1.8, 0])
        self.at("kl")
        self.play(FadeIn(k), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5. before
    def before(self):
        self.section(5)
        self.clear_stage()
        h = Text("before training · 100 held-out questions, greedy", font_size=26).to_edge(UP, buff=0.6)
        self.at("before")
        self.play(FadeIn(h), run_time=0.4)
        s1 = Text("reward: 0 / 100", font=MONO, font_size=28, color=RED_B).move_to([0, 1.6, 0])
        self.at("zero")
        self.play(FadeIn(s1), run_time=0.3)
        s2 = Text("last number in the text right: 95 / 100", font=MONO, font_size=24, color=GREEN_B).move_to([0, 0.9, 0])
        self.at("95")
        self.play(FadeIn(s2), run_time=0.3)
        ex = Text("…2. Addition: 74 + 48 = 122\nTherefore, the answer is:  \\boxed{",
                  font=MONO, font_size=20, color=GREY_A, line_spacing=0.9).move_to([0, -0.4, 0])
        self.at("boxed")
        self.play(FadeIn(ex), run_time=0.4)
        s3 = Text("16 / 100 run out of room (160 tokens)", font=MONO, font_size=22, color=GOLD).move_to([0, -1.8, 0])
        self.at("16")
        self.play(FadeIn(s3), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. training
    def training(self):
        self.section(6)
        self.clear_stage()
        h = Text("30 steps × 4 questions × 8 samples · 51 minutes on a CPU", font_size=24).to_edge(UP, buff=0.5)
        self.at("30")
        self.play(FadeIn(h), run_time=0.4)
        ax = Axes(x_range=[0, 31, 5], y_range=[0, 1.0, 0.25], x_length=8.5, y_length=3.8, tips=False,
                  axis_config={"color": GREY_B}).move_to([0.2, -0.4, 0])
        ticks = VGroup(*[Text(str(v), font_size=16).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v in (0, 5, 10, 15, 20, 25, 30)],
                       *[Text(f"{v:.2f}", font_size=16).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, 0.25, 0.5, 0.75, 1.0)])
        xl = Text("step", font_size=18).next_to(ax.x_axis, DOWN, buff=0.4)
        yl = Text("training reward", font_size=18).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.55)
        self.at("training")
        self.play(Create(ax), FadeIn(ticks), FadeIn(xl), FadeIn(yl), run_time=0.5)
        pts = [ax.c2p(i + 1, v) for i, v in enumerate(REWARD)]
        first = VMobject(color=GREEN_B).set_points_as_corners(pts[:5])
        rest = VMobject(color=GREEN_B).set_points_as_corners(pts[4:])
        dots = VGroup(*[Dot(p, radius=0.04, color=GREEN_B) for p in pts])
        self.at("quarter")
        self.play(Create(first), FadeIn(dots[:5]), run_time=1.2)
        self.at("stays")
        self.play(Create(rest), FadeIn(dots[5:]), run_time=1.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. after
    def after(self):
        self.section(7)
        self.clear_stage()
        hdr = VGroup(Text("", font_size=22), Text("before", font_size=24, color=GREY_B), Text("after GRPO", font_size=24, color=GREEN_B))
        rows = [("reward (“Answer: N” right)", "0", "94"), ("finished within 160 tokens", "84", "100"),
                ("mean length (tokens)", "148", "105"), ("last number right", "95", "94")]
        xs = (-2.8, 1.8, 4.3)
        for m, x in zip(hdr, xs):
            m.move_to([x, 2.5, 0])
        lines = VGroup(*[VGroup(*[Text(v, font_size=22, color=WHITE if j == 0 else (GREY_B if j == 1 else GREEN_B)).move_to([xs[j], 1.8 - 0.6 * k, 0])
                                  for j, v in enumerate(r)]) for k, r in enumerate(rows)])
        self.at("after")
        self.play(FadeIn(hdr), run_time=0.3)
        self.at("94")
        self.play(FadeIn(lines[0]), run_time=0.3)
        self.at("finishes")
        self.play(FadeIn(lines[1]), run_time=0.3)
        self.at("shorter")
        self.play(FadeIn(lines[2]), run_time=0.3)
        self.at("last")
        self.play(FadeIn(lines[3]), run_time=0.3)
        ex = Text("…2. Next, add 48:  74 + 48 = 122\nTherefore, the answer is:\nAnswer: 122", font=MONO, font_size=20,
                  color=GREEN_B, line_spacing=0.9).move_to([0, -1.0, 0])
        n = Text("it learned the form the reward checks, not arithmetic", font_size=24, color=YELLOW).move_to([0, -2.5, 0])
        self.at("taught")
        self.play(FadeIn(ex), run_time=0.4)
        self.at("form")
        self.play(FadeIn(n), run_time=0.3)
        self.at("large")
        big = Text("at scale: longer chains of thought that check their own work", font_size=22, color=GREY_A).move_to([0, -3.1, 0])
        self.play(FadeIn(big), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("advantage")
        self.play(Create(hl), run_time=0.3)
        self.at("loss")
        self.play(highlight(hl, code, 3), run_time=0.3)
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
