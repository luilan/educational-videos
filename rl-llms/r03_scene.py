"""Reinforcement Learning for LLMs, episode 3 — Baselines and Advantages.

Render from the repo root:  ./render.sh rl-llms r03
Numbers come from code/r03_baselines (baselines.py → results.json).
"""
import json

from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from r03_script import NEXT, TAGLINE, TITLE
from rl_common import BAD, GOOD, MODEL_C, REWARD_C, grade_badge, study_footer
from voiced_scene import VoicedScene

SERIES_LABEL = "REINFORCEMENT LEARNING FOR LLMs  ·  EPISODE 3"
CODE_DIR = "code/r03_baselines"
R = json.load(open(f"{CODE_DIR}/results.json"))
KIND_C = {"none": GREY_B, "batch": BLUE_C, "question": TEAL_C}
KIND_L = {"none": "no baseline", "batch": "batch average", "question": "per question"}
CODE = """adv = r - r.mean()                                     # batch average
adv = (r.view(Q, G) - r.view(Q, G).mean(1, keepdim=True)).view(-1)   # per question
loss = -(adv * logp).mean()"""


def pct(x):
    return f"{100 * x:.0f}%"


def adv_bars(values, x0, y0, unit=1.4, w=0.42, gap=0.12):
    g = VGroup()
    for i, v in enumerate(values):
        col = GOOD if v > 0 else BAD
        b = Rectangle(width=w, height=max(0.03, abs(v) * unit), stroke_width=0, fill_color=col, fill_opacity=0.85)
        x = x0 + i * (w + gap)
        b.move_to([x, y0, 0], aligned_edge=DOWN if v > 0 else UP)
        g.add(b)
    return g


class BaselineVideo(VoicedScene):
    VIDEO = "r03"

    def construct(self):
        play_token_intro(self, TITLE, 3, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.two()         # 2
        self.theory()      # 3
        self.exp1()        # 4
        self.exp2()        # 5
        self.with_base()   # 6
        self.lesson()      # 7
        self.code()        # 8
        self.outro()       # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        a = Text("REINFORCE: push up  reward × log p", font_size=30, color=GREY_A).move_to([0, 1.6, 0])
        self.at("reinforce")
        self.play(FadeIn(a), run_time=0.5)
        f = VGroup(Text("advantage", font_size=40, color=REWARD_C), Text("=", font_size=40),
                   Text("reward", font_size=40), Text("−", font_size=40), Text("baseline", font_size=40, color=MODEL_C)).arrange(RIGHT, buff=0.3)
        f.move_to([0, 0, 0])
        self.at("baseline")
        self.play(FadeIn(f[4]), run_time=0.4)
        self.at("subtract")
        self.play(FadeIn(f[2:4]), run_time=0.4)
        s = Text("learn from better or worse than usual", font_size=26, color=GREY_B).move_to([0, -1.3, 0])
        self.at("better")
        self.play(FadeIn(s), run_time=0.4)
        self.at("advantage")
        self.play(FadeIn(f[:2]), run_time=0.4)
        b = Text("push up  advantage × log p", font_size=30, color=REWARD_C).move_to([0, -2.4, 0])
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. two baselines
    def two(self):
        self.section(2)
        self.clear_stage()
        ex = R["example"]
        lt = Text("batch average", font_size=28, color=KIND_C["batch"]).move_to([-3.5, 2.6, 0])
        ls = Text(f"{pct(ex['batch_mean'])} of the 256 answers right", font_size=20, color=GREY_B).next_to(lt, DOWN, buff=0.15)
        self.at("batch")
        self.play(FadeIn(lt), FadeIn(ls), run_time=0.5)
        zero_l = Line([-5.8, 0.2, 0], [-1.2, 0.2, 0], color=GREY_C)
        rb = Rectangle(width=0.8, height=1.4 * (1 - ex["batch_mean"]), stroke_width=0, fill_color=GOOD, fill_opacity=0.85).move_to([-4.6, 0.2, 0], aligned_edge=DOWN)
        wb = Rectangle(width=0.8, height=1.4 * ex["batch_mean"], stroke_width=0, fill_color=BAD, fill_opacity=0.85).move_to([-2.4, 0.2, 0], aligned_edge=UP)
        rl = VGroup(Text("right", font_size=20), Text(f"+{1 - ex['batch_mean']:.2f}", font=MONO, font_size=22, color=GOOD)).arrange(DOWN, buff=0.08).next_to(rb, UP, buff=0.1)
        wl = VGroup(Text(f"−{ex['batch_mean']:.2f}", font=MONO, font_size=22, color=BAD), Text("wrong", font_size=20)).arrange(DOWN, buff=0.08).next_to(wb, DOWN, buff=0.1)
        self.play(Create(zero_l), run_time=0.3)
        self.at("79")
        self.play(GrowFromEdge(rb, DOWN), FadeIn(rl), run_time=0.5)
        self.at("minus")
        self.play(GrowFromEdge(wb, UP), FadeIn(wl), run_time=0.5)
        qt = Text("per question", font_size=28, color=KIND_C["question"]).move_to([3.4, 2.6, 0])
        qs = Text("3 of 8 tries right", font_size=20, color=GREY_B).next_to(qt, DOWN, buff=0.15)
        self.at("question")
        self.play(FadeIn(qt), FadeIn(qs), run_time=0.5)
        badges = VGroup(*[grade_badge(int(r), font_size=18).scale(0.7) for r in ex["rewards"]]).arrange(RIGHT, buff=0.12).move_to([3.4, 1.3, 0])
        zero_r = Line([1.0, 0.2, 0], [5.8, 0.2, 0], color=GREY_C)
        bars = adv_bars(ex["question"], badges[0].get_x(), 0.2, unit=1.4, w=0.42, gap=badges[1].get_x() - badges[0].get_x() - 0.42)
        self.play(FadeIn(badges), Create(zero_r), run_time=0.4)
        self.at("625")
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b, v in zip(bars, ex["question"]) if v > 0], lag_ratio=0.2), run_time=0.5)
        vals = VGroup(Text(f"+{ex['question'][3]:.3f}", font=MONO, font_size=20, color=GOOD).move_to([3.4, 1.85, 0]).shift(RIGHT * 0.0))
        self.at("375")
        self.play(LaggedStart(*[GrowFromEdge(b, UP) for b, v in zip(bars, ex["question"]) if v < 0], lag_ratio=0.1), run_time=0.5)
        nums = Text(f"right +{ex['question'][3]:.3f}   wrong {ex['question'][0]:.3f}", font=MONO, font_size=20, color=GREY_A).move_to([3.4, -1.2, 0])
        self.play(FadeIn(nums), run_time=0.3)
        d = Text("now wrong answers are pushed down directly", font_size=26, color=REWARD_C).move_to([0, -2.6, 0])
        self.at("directly")
        self.play(FadeIn(d), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. theory
    def theory(self):
        self.section(3)
        self.clear_stage()
        eq = VGroup(Text("mean of (reward − b) · ∇ log p", font=MONO, font_size=28),
                    Text("= mean of reward · ∇ log p  −  b · mean of ∇ log p", font=MONO, font_size=26),
                    Text("= mean of reward · ∇ log p  −  b · 0", font=MONO, font_size=26, color=REWARD_C)).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        eq.move_to([0, 1.0, 0])
        self.at("average")
        self.play(FadeIn(eq[0]), run_time=0.5)
        self.at("depend")
        self.play(FadeIn(eq[1]), run_time=0.5)
        why = Text("(the probabilities add up to 1, so their gradients cancel on average)", font_size=20, color=GREY_B).next_to(eq, DOWN, buff=0.3)
        self.play(FadeIn(eq[2]), FadeIn(why), run_time=0.5)
        n = VGroup(Text("same direction", font_size=28, color=GOOD), Text("less noise", font_size=28, color=MODEL_C)).arrange(RIGHT, buff=1.2).move_to([0, -2.0, 0])
        self.at("noise")
        self.play(FadeIn(n), run_time=0.5)
        t = Text("in theory: faster learning", font_size=24, color=GREY_B).next_to(n, DOWN, buff=0.3)
        self.at("theory")
        self.play(FadeIn(t), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. experiment 1
    def finals(self, finals, cue_map):
        rows = VGroup()
        for i, k in enumerate(("none", "batch", "question")):
            y = 1.2 - 1.15 * i
            lab = Text(KIND_L[k], font_size=24, color=KIND_C[k]).move_to([-4.4, y, 0], aligned_edge=LEFT)
            vals = finals[k]
            mean = sum(vals) / len(vals)
            bar = Rectangle(width=6.0 * mean, height=0.5, stroke_width=0, fill_color=KIND_C[k], fill_opacity=0.6).move_to([-1.6, y, 0], aligned_edge=LEFT)
            dots = VGroup(*[Dot([-1.6 + 6.0 * v, y, 0], radius=0.07, color=WHITE) for v in vals])
            m = Text(pct(mean), font_size=26, color=KIND_C[k]).next_to(bar, RIGHT, buff=0.3)
            rows.add(VGroup(lab, bar, dots, m))
        return rows

    def exp1(self):
        self.section(4)
        self.clear_stage()
        h = Text("rewards 0 and 1 · 3 runs each · 2,000 steps · right on 1,000 new questions", font_size=22, color=GREY_B).to_edge(UP, buff=0.6)
        self.at("tested")
        self.play(FadeIn(h), run_time=0.4)
        rows = self.finals({k: v["final"] for k, v in R["summary"].items()}, None)
        for row, cue in zip(rows, ("baseline", "batch", "per")):
            self.at(cue)
            self.play(FadeIn(row[0]), GrowFromEdge(row[1], LEFT), FadeIn(row[2]), FadeIn(row[3]), run_time=0.5)
        note = Text("dots: the 3 runs", font_size=18, color=GREY_B).next_to(rows, DOWN, buff=0.2).align_to(rows[0][1], LEFT)
        self.play(FadeIn(note), run_time=0.2)
        no = Text("no real difference", font_size=30, color=REWARD_C).move_to([0, -2.3, 0])
        self.at("difference")
        self.play(FadeIn(no), run_time=0.4)
        nz = R["noise"]
        why = Text(f"gradient signal-to-noise at the start: {nz['none']['snr']:.2f} · {nz['batch']['snr']:.2f} · {nz['question']['snr']:.2f}",
                   font_size=20, color=GREY_B).move_to([0, -3.0, 0])
        self.at("barely")
        self.play(FadeIn(why), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. experiment 2: a point for trying
    def axes(self):
        ax = Axes(x_range=[0, 2000, 500], y_range=[0, 0.8, 0.2], x_length=6.4, y_length=3.6, tips=False,
                  axis_config={"color": GREY_C})
        ax.move_to([3.0, -0.5, 0])
        xl = VGroup(*[Text(f"{s:,}", font_size=16, color=GREY_B).next_to(ax.c2p(s, 0), DOWN, buff=0.12) for s in range(0, 2001, 500)])
        yl = VGroup(*[Text(f"{int(v * 100)}%", font_size=16, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v in (0, 0.2, 0.4, 0.6, 0.8)])
        cap = Text("steps", font_size=16, color=GREY_B).next_to(xl, DOWN, buff=0.05)
        return ax, VGroup(ax, xl, yl, cap)

    def curve(self, ax, c, color, width=3, dashed=False):
        pts = [ax.c2p(s, a) for s, a in c]
        m = VMobject(color=color, stroke_width=width).set_points_as_corners(pts)
        return DashedVMobject(m, num_dashes=60) if dashed else m

    def exp2(self):
        self.section(5)
        self.clear_stage()
        tab = VGroup(Text("a point for trying", font_size=28, color=REWARD_C),
                     VGroup(Text("wrong answer:", font_size=24), Text("1", font=MONO, font_size=26, color=BAD)).arrange(RIGHT, buff=0.3),
                     VGroup(Text("right answer:", font_size=24), Text("2", font=MONO, font_size=26, color=GOOD)).arrange(RIGHT, buff=0.3)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([-6.6, 1.7, 0], aligned_edge=LEFT)
        self.at("change")
        self.play(FadeIn(tab[0]), run_time=0.4)
        self.at("one")
        self.play(FadeIn(tab[1]), run_time=0.3)
        self.at("two")
        self.play(FadeIn(tab[2]), run_time=0.3)
        self.ax, frame = self.axes()
        self.frame = frame
        self.at("better")
        self.play(FadeIn(frame), run_time=0.4)
        lab = Text("no baseline, 3 runs", font_size=20, color=BAD).move_to([-6.6, -0.6, 0], aligned_edge=LEFT)
        curves = VGroup(*[self.curve(self.ax, c, BAD) for c in R["offset"]["none"]["curves"]])
        self.at("falls")
        self.play(FadeIn(lab), *[Create(c) for c in curves], run_time=1.4)
        fin = Text(f"14% → {pct(min(R['offset']['none']['final']))}–{pct(max(R['offset']['none']['final']))}", font_size=26, color=BAD).move_to([-6.6, -1.2, 0], aligned_edge=LEFT)
        self.play(FadeIn(fin), run_time=0.3)
        p = Text("every answer pushed up,\nwrong ones too", font_size=20, line_spacing=0.9, color=GREY_A).move_to([-6.6, -2.0, 0], aligned_edge=LEFT)
        self.at("pushed")
        self.play(FadeIn(p), run_time=0.4)
        k = Text("it locks in its most\ncommon answer", font_size=20, line_spacing=0.9, color=GREY_A).next_to(p, DOWN, buff=0.15).align_to(p, LEFT)
        self.at("locks")
        self.play(FadeIn(k), run_time=0.4)
        self.curves = curves
        self.end_section()

    # ------------------------------------------------------------------ 6. with a baseline
    def with_base(self):
        self.section(6)
        keep = [self.frame, *self.curves]
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=0.4)
        base = self.curve(self.ax, R["runs"]["batch"][0], KIND_C["batch"], width=6)
        shifted = self.curve(self.ax, R["offset"]["batch"]["curves"][0], YELLOW, width=3, dashed=True)
        l1 = Text("batch baseline, rewards 0 / 1", font_size=18, color=KIND_C["batch"]).move_to([-6.6, 1.6, 0], aligned_edge=LEFT)
        l2 = Text("batch baseline, rewards 1 / 2 (dashed)", font_size=18, color=YELLOW).move_to([-6.6, 1.1, 0], aligned_edge=LEFT)
        l3 = Text("no baseline, rewards 1 / 2", font_size=18, color=BAD).move_to([-6.6, 0.6, 0], aligned_edge=LEFT)
        self.at("either")
        self.play(FadeIn(l3), Create(base), FadeIn(l1), run_time=1.0)
        self.at("identical")
        self.play(Create(shifted), FadeIn(l2), run_time=1.2)
        same = Text("identical, step for step", font_size=26, color=GOOD).move_to([-6.6, -0.6, 0], aligned_edge=LEFT)
        self.play(FadeIn(same), run_time=0.3)
        ex = Text("the average absorbs\nthe extra point exactly", font_size=20, color=GREY_A, line_spacing=0.9).next_to(same, DOWN, buff=0.2).align_to(same, LEFT)
        self.at("exactly")
        self.play(FadeIn(ex), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. lesson
    def lesson(self):
        self.section(7)
        self.clear_stage()
        h = Text("the real job of a baseline", font_size=32, color=REWARD_C).to_edge(UP, buff=0.7)
        self.at("job")
        self.play(FadeIn(h), run_time=0.4)
        t = Text("learning depends on better or worse than usual,\nnot on where zero happens to be", font_size=28, line_spacing=0.9).move_to([0, 1.2, 0])
        self.at("zero")
        self.play(FadeIn(t), run_time=0.5)
        rm = Text("reward models (episode 7) give scores with no natural zero", font_size=22, color=GREY_B).move_to([0, -0.1, 0])
        self.at("scores")
        self.play(FadeIn(rm), run_time=0.4)
        items = VGroup(VGroup(Text("PPO", font_size=28, color=MODEL_C), Text("learns the baseline: a value head (episode 5)", font_size=22)).arrange(RIGHT, buff=0.4),
                       VGroup(Text("GRPO", font_size=28, color=MODEL_C), Text("uses the average of the group (episode 6)", font_size=22)).arrange(RIGHT, buff=0.4)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, -1.7, 0])
        self.at("ppo")
        self.play(FadeIn(items[0]), run_time=0.4)
        self.at("grpo")
        self.play(FadeIn(items[1]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.at("subtract")
        self.play(Create(hl), run_time=0.3)
        self.at("log")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("question")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), FadeIn(study_footer()), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
