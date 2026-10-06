"""How LLMs Work: Deep Dive, episode 38 — RLHF with PPO.

Render from the repo root:  ./render.sh deep-dive d38
Every number on screen comes from code/d38_ppo/ppo.py (GPT-2 small policy with a value head; reward = positive minus
negative words in a 24-token continuation; 200 PPO iterations of 16 samples; beta = 0, 0.05, 0.5). The curves in
assets/d38/curves.json are its training logs (batch reward and KL, every 10 iterations).
"""
import json
from pathlib import Path

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from d38_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 38"
CURVES = json.loads((Path(__file__).parent / "assets/d38/curves.json").read_text())
CODE = """ratio = (logp - old_logp).exp()
loss = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()"""


def box(text, color, w=2.6, h=1.0, fs=22):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.12, color=color, fill_opacity=0.25)
    return VGroup(r, Text(text, font_size=fs, line_spacing=0.8).move_to(r))


def sample(text, color=WHITE, width=11.5, fs=22):
    t = Paragraph(*wrap(text, 70), font_size=fs, line_spacing=0.7, color=color) if len(text) > 70 else Text(text, font_size=fs, color=color)
    if t.width > width:
        t.scale_to_fit_width(width)
    return t


def wrap(text, n):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur]


def reward_chart(beta_keys, colors, y_max=25):
    ax = Axes(x_range=[0, 200, 50], y_range=[0, y_max, 5 if y_max > 5 else 1], x_length=7.5, y_length=3.6, tips=False,
              axis_config={"color": GREY_B}).move_to([-1.0, -0.6, 0])
    ticks = VGroup(*[Text(str(v), font_size=16).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v in (0, 50, 100, 150, 200)],
                   *[Text(str(v), font_size=16).next_to(ax.c2p(0, v), LEFT, buff=0.1)
                     for v in range(0, y_max + 1, 5 if y_max > 5 else 1)])
    xl = Text("PPO iteration", font_size=18).next_to(ax.x_axis, DOWN, buff=0.4)
    yl = Text("reward", font_size=18).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.4)
    lines = VGroup(*[VMobject(color=c).set_points_as_corners([ax.c2p(i, min(r, y_max)) for i, r, _ in CURVES[k]])
                     for k, c in zip(beta_keys, colors)])
    return ax, VGroup(ticks, xl, yl), lines


class PPOVideo(VoicedScene):
    VIDEO = "d38"

    def construct(self):
        play_token_intro(self, TITLE, 38, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.setup_task()    # 2
        self.loop()          # 3
        self.hacking()       # 4
        self.penalty()       # 5
        self.weak()          # 6
        self.strong()        # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        m = box("model", BLUE_C).move_to([-4.0, 0.8, 0])
        t = box("text", GREY_C, w=1.8).move_to([0, 0.8, 0])
        r = box("reward\nmodel", MODEL_COLOR).move_to([4.0, 0.8, 0])
        p = box("PPO update", GOLD, w=3.0).move_to([0, -1.6, 0])
        self.at("writes")
        self.play(FadeIn(m), GrowArrow(Arrow(m.get_right(), t.get_left(), buff=0.1)), FadeIn(t), run_time=0.5)
        self.at("scores")
        self.play(GrowArrow(Arrow(t.get_right(), r.get_left(), buff=0.1)), FadeIn(r), run_time=0.5)
        self.at("nudges")
        self.play(GrowArrow(Arrow(r.get_bottom(), p.get_right(), buff=0.1)), FadeIn(p),
                  GrowArrow(Arrow(p.get_left(), m.get_bottom(), buff=0.1)), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 2. task
    def setup_task(self):
        self.section(2)
        self.clear_stage()
        g = Text("goal: positive continuations of “The movie was …”", font_size=26).to_edge(UP, buff=0.6)
        self.at("goal")
        self.play(FadeIn(g), run_time=0.4)
        rw = VGroup(Text("reward = positive words − negative words", font=MONO, font_size=24, color=GOLD),
                    Text("good great love wonderful … − bad terrible boring …", font_size=20, color=GREY_B)).arrange(DOWN, buff=0.2)
        rw.move_to([0, 1.2, 0])
        self.at("reward")
        self.play(FadeIn(rw), run_time=0.5)
        s = sample("The movie was not only mixed emotions resonanced tones of emotion but it concentrated a few colors on a set "
                   "that might be referred to as", GREY_A, fs=20)
        s.move_to([0, -0.6, 0])
        b = Text("plain GPT-2: 0.14 per 24 tokens", font_size=26, color=YELLOW).move_to([0, -2.3, 0])
        self.at("plain")
        self.play(FadeIn(s), FadeIn(b), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 3. loop
    def loop(self):
        self.section(3)
        self.clear_stage()
        steps = VGroup(Text("1. write 16 continuations", font_size=24), Text("2. score each one", font_size=24),
                       Text("3. value head: expected score → advantage per token", font_size=24),
                       Text("4. a few clipped steps", font_size=24)).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        steps.move_to([-2.4, 1.0, 0])
        for k, cue in enumerate(("16", "scored", "value", "clipped")):
            self.at(cue)
            self.play(FadeIn(steps[k], shift=0.2 * RIGHT), run_time=0.4)
        ax = Axes(x_range=[0.5, 1.5, 0.5], y_range=[0, 1.5, 0.5], x_length=4, y_length=2.2, tips=False,
                  axis_config={"color": GREY_B}).move_to([3.6, -1.6, 0])
        clip = VMobject(color=GREEN_B).set_points_as_corners([ax.c2p(0.5, 0.5), ax.c2p(1.2, 1.2), ax.c2p(1.5, 1.2)])
        lab = VGroup(Text("probability ratio", font_size=16).next_to(ax, DOWN, buff=0.1),
                     Text("push (advantage > 0)", font_size=16).next_to(ax, UP, buff=0.1),
                     Text("clipped at 1.2", font_size=16, color=GREEN_B).next_to(ax.c2p(1.35, 1.2), UP, buff=0.1))
        self.at("moved")
        self.play(Create(ax), Create(clip), FadeIn(lab), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 4. hacking
    def hacking(self):
        self.section(4)
        self.clear_stage()
        ax, deco, lines = reward_chart(["0.0"], [RED_C])
        self.at("nothing")
        self.play(Create(ax), FadeIn(deco), run_time=0.5)
        self.at("climbs")
        self.play(Create(lines[0]), run_time=1.2)
        l = Text("beta 0: 24 of 24", font_size=22, color=RED_B).next_to(ax.c2p(200, 24), LEFT, buff=0.2).shift(DOWN * 0.3)
        self.play(FadeIn(l), run_time=0.3)
        s = sample("“The movie was wonderful good wonderful good great great wonderful good wonderful good …”", RED_B, fs=22)
        s.move_to([0, 2.6, 0])
        self.at("wonderful")
        self.play(FadeIn(s), run_time=0.4)
        h = Text("reward hacking", font_size=30, color=YELLOW).move_to([4.6, 0.4, 0])
        self.at("hacking")
        self.play(FadeIn(h), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5. penalty
    def penalty(self):
        self.section(5)
        self.clear_stage()
        t = Text("the leash: a KL penalty", font_size=30, color=YELLOW).move_to([0, 2.2, 0])
        self.at("leash")
        self.play(FadeIn(t), run_time=0.3)
        f = Text("penalty per token = β · (log p_new(token) − log p_GPT-2(token))", font=MONO, font_size=22).move_to([0, 0.6, 0])
        self.at("penalty")
        self.play(FadeIn(f), run_time=0.5)
        k = Text("summed over the text ≈ KL divergence from the original model", font_size=24, color=GREY_A).move_to([0, -0.6, 0])
        self.at("summed")
        self.play(FadeIn(k), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. weak
    def weak(self):
        self.section(6)
        self.clear_stage()
        ax, deco, lines = reward_chart(["0.0", "0.05"], [RED_C, GOLD])
        self.at("beta")
        self.play(Create(ax), FadeIn(deco), FadeIn(lines[0].set_stroke(opacity=0.4)), run_time=0.5)
        self.play(Create(lines[1]), run_time=1.0)
        s = sample("“I think this restaurant great great great great great great great great great …”", GOLD, fs=22)
        s.move_to([0, 2.6, 0])
        self.at("great")
        self.play(FadeIn(s), run_time=0.4)
        k = VGroup(Text("KL at the end", font_size=20, color=GREY_B), Text("beta 0:      59", font=MONO, font_size=20, color=RED_B),
                   Text("beta 0.05:   25", font=MONO, font_size=20, color=GOLD)).arrange(DOWN, aligned_edge=LEFT).move_to([4.8, -0.6, 0])
        self.at("divergence")
        self.play(FadeIn(k), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. strong
    def strong(self):
        self.section(7)
        self.clear_stage()
        head = Text("beta 0.5: fresh samples, 8 prompts", font_size=26).to_edge(UP, buff=0.5)
        self.at("beta")
        self.play(FadeIn(head), run_time=0.3)
        bars = VGroup()
        for k, (lab, v, col) in enumerate((("GPT-2", 0.14, GREY_B), ("after PPO", 0.92, GREEN_C))):
            b = Rectangle(width=v * 5, height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85).move_to([-1.5, 1.5 - 0.8 * k, 0], aligned_edge=LEFT)
            bars.add(VGroup(Text(lab, font_size=22).next_to(b, LEFT, buff=0.2).align_to([-1.7, 0, 0], RIGHT), b,
                            Text(f"{v:.2f}", font=MONO, font_size=22).next_to(b, RIGHT, buff=0.15)))
        self.at("rises")
        self.play(FadeIn(bars[0]), run_time=0.3)
        self.play(FadeIn(bars[1]), run_time=0.4)
        s = sample("“The movie was not only mixed reviews, opinion scores and a wide reputation out there as one of the great "
                   "films of 2014, but also”", GREEN_B, fs=20)
        s.move_to([0, -0.8, 0])
        self.at("english")
        self.play(FadeIn(s), run_time=0.5)
        k = Text("KL ≈ 6 · more positive, still writes", font_size=24, color=YELLOW).move_to([0, -2.6, 0])
        self.at("forgetting")
        self.play(FadeIn(k), run_time=0.3)
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
        self.at("heart")
        self.play(Create(hl), run_time=0.3)
        self.at("minimum")
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
