"""Reinforcement Learning for LLMs, episode 4 — The KL Leash.

Render from the repo root:  ./render.sh rl-llms r04
Numbers come from code/r04_kl_leash (kl_leash.py → results.json).
"""
import json
import statistics

from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from r04_script import NEXT, TAGLINE, TITLE
from rl_common import BAD, GOOD, MODEL_C, REWARD_C, grade_badge, study_footer
from voiced_scene import VoicedScene

SERIES_LABEL = "REINFORCEMENT LEARNING FOR LLMs  ·  EPISODE 4"
CODE_DIR = "code/r04_kl_leash"
R = json.load(open(f"{CODE_DIR}/results.json"))
REF_C = TEAL_C
LEASH_C = YELLOW
FAST = {r["beta"]: r for r in R["fast"]}
CODE = """kl = logp.detach() - ref_logp    # vs the frozen reference
shaped = r - beta * kl            # the leash
g = shaped.view(Q, G)
adv = (g - g.mean(1, keepdim=True)).view(-1)
loss = -(adv * logp).mean()"""


def pct(x):
    return f"{100 * x:.0f}%"


def sweep(beta):
    runs = [r for r in R["sweep"] if r["beta"] == beta]
    fin = [r["curve"][-1] for r in runs]
    return {k: statistics.mean(f[k] for f in fin) for k in ("greedy_acc", "kl", "distinct")}


def strange_first(answers):
    """Malformed answers (not digits followed by '.') first, then the rest."""
    bad = [a for a in answers if not (a[:-1].isdigit() and a.endswith("."))]
    return bad + [a for a in answers if a not in bad]


def answer_row(answers, right, font_size=22):
    cells = VGroup()
    for a in answers:
        col = GOOD if a == right else GREY_A
        box = RoundedRectangle(corner_radius=0.08, width=0.95, height=0.5, stroke_color=col, stroke_width=2,
                               fill_color=col, fill_opacity=0.08)
        cells.add(VGroup(box, Text(a, font=MONO, font_size=font_size, color=col).move_to(box)))
    return cells.arrange(RIGHT, buff=0.1)


class KLLeashVideo(VoicedScene):
    VIDEO = "r04"

    def construct(self):
        play_token_intro(self, TITLE, 4, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.stuck()       # 2
        self.leash()       # 3
        self.rescue()      # 4
        self.sweep()       # 5
        self.text()        # 6
        self.why()         # 7
        self.code()        # 8
        self.outro()       # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def axes(self, pos, x_length=6.4, y_length=3.4):
        ax = Axes(x_range=[0, 2000, 500], y_range=[0, 0.8, 0.2], x_length=x_length, y_length=y_length, tips=False,
                  axis_config={"color": GREY_C})
        ax.move_to(pos)
        xl = VGroup(*[Text(f"{s:,}", font_size=16, color=GREY_B).next_to(ax.c2p(s, 0), DOWN, buff=0.12) for s in range(0, 2001, 500)])
        yl = VGroup(*[Text(f"{int(v * 100)}%", font_size=16, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v in (0, 0.2, 0.4, 0.6, 0.8)])
        cap = Text("steps · right on 1,000 new questions", font_size=16, color=GREY_B).next_to(xl, DOWN, buff=0.08)
        return ax, VGroup(ax, xl, yl, cap)

    def curve(self, ax, curve, color, width=3):
        pts = [ax.c2p(c["step"], c["greedy_acc"]) for c in curve]
        return VMobject(color=color, stroke_width=width).set_points_as_corners(pts)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        ax, frame = self.axes([2.6, 0.2, 0])
        normal = [r for r in R["sweep"] if r["beta"] == 0.0 and r["seed"] == 1][0]
        l1 = Text("learning rate 0.0003", font_size=24, color=GOOD).move_to([-6.6, 2.4, 0], aligned_edge=LEFT)
        f1 = Text(f"{pct(R['start']['greedy_acc'])} → {pct(sweep(0.0)['greedy_acc'])}", font_size=30, color=GOOD).next_to(l1, DOWN, buff=0.15).align_to(l1, LEFT)
        self.at("episode")
        self.play(FadeIn(frame), FadeIn(l1), run_time=0.5)
        c1 = self.curve(ax, normal["curve"], GOOD)
        self.play(Create(c1), FadeIn(f1), run_time=1.2)
        l2 = Text("10× bigger steps: 0.003", font_size=24, color=BAD).move_to([-6.6, 0.8, 0], aligned_edge=LEFT)
        self.at("10")
        self.play(FadeIn(l2), run_time=0.4)
        c2 = self.curve(ax, FAST[0.0]["curve"], BAD, width=4)
        self.at("150")
        self.play(Create(c2), run_time=1.0)
        rows = VGroup(*[Text(f"{q} 111.", font=MONO, font_size=24, color=BAD) for q in FAST[0.0]["samples"]["2000"]]
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([-6.6, -0.6, 0], aligned_edge=LEFT)
        self.at("111")
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.3), run_time=0.8)
        f2 = Text(f"{100 * FAST[0.0]['curve'][-1]['greedy_acc']:.1f}% right", font_size=30, color=BAD).move_to([-6.6, -2.2, 0], aligned_edge=LEFT)
        self.at("right")
        self.play(FadeIn(f2), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. stuck
    def stuck(self):
        self.section(2)
        self.clear_stage()
        h = Text("a few big updates → the model is certain", font_size=30, color=GREY_A).to_edge(UP, buff=0.8)
        self.at("certain")
        self.play(FadeIn(h), run_time=0.5)
        q = Text("47+38=", font=MONO, font_size=28).move_to([-5.6, 1.0, 0])
        tries = answer_row(["111."] * 8, "85.").next_to(q, RIGHT, buff=0.4)
        self.at("eight")
        self.play(FadeIn(q), LaggedStart(*[FadeIn(t) for t in tries], lag_ratio=0.1), run_time=0.8)
        badges = VGroup(*[grade_badge(0, font_size=18).scale(0.7).next_to(t, DOWN, buff=0.15) for t in tries])
        self.at("reward")
        self.play(FadeIn(badges), run_time=0.4)
        advs = VGroup(*[Text("0", font=MONO, font_size=22, color=GREY_B).next_to(b, DOWN, buff=0.2) for b in badges])
        al = Text("advantage", font_size=20, color=GREY_B).next_to(advs, LEFT, buff=0.35)
        self.at("advantage")
        self.play(FadeIn(advs), FadeIn(al), run_time=0.4)
        n = Text("reward − average = 0 − 0 = 0: nothing to learn from", font_size=26, color=REWARD_C).move_to([0, -1.9, 0])
        self.at("nothing")
        self.play(FadeIn(n), run_time=0.4)
        s = Text("stuck forever", font_size=34, color=BAD).next_to(n, DOWN, buff=0.4)
        self.at("stuck")
        self.play(FadeIn(s), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. the leash
    def leash(self):
        self.section(3)
        self.clear_stage()
        def box(label, sub, color):
            r = RoundedRectangle(corner_radius=0.15, width=3.4, height=1.3, stroke_color=color, fill_color=color, fill_opacity=0.12)
            t = VGroup(Text(label, font_size=28, color=color), Text(sub, font_size=18, color=GREY_B)).arrange(DOWN, buff=0.1).move_to(r)
            return VGroup(r, t)
        model = box("model", "being trained", MODEL_C).move_to([-3.2, 2.0, 0])
        ref = box("reference", "frozen copy", REF_C).move_to([3.2, 2.0, 0])
        self.at("leash")
        self.play(FadeIn(model), run_time=0.4)
        self.at("reference")
        self.play(FadeIn(ref), run_time=0.5)
        rope = DashedLine(model.get_right(), ref.get_left(), color=LEASH_C, stroke_width=4)
        rl = Text("the leash", font_size=20, color=LEASH_C).next_to(rope, UP, buff=0.1)
        self.play(Create(rope), FadeIn(rl), run_time=0.5)
        d = VGroup(Text("log p", font=MONO, font_size=26), Text("model", font_size=18, color=MODEL_C),
                   Text("(answer) − log p", font=MONO, font_size=26), Text("reference", font_size=18, color=REF_C),
                   Text("(answer)", font=MONO, font_size=26)).arrange(RIGHT, buff=0.08, aligned_edge=DOWN).move_to([0, 0.3, 0])
        self.at("measure")
        self.play(FadeIn(d), run_time=0.6)
        sh = VGroup(Text("shaped reward = reward − β ×", font_size=28),
                    Text("(that difference)", font_size=28, color=LEASH_C)).arrange(RIGHT, buff=0.2).move_to([0, -0.8, 0])
        self.at("subtract")
        self.play(FadeIn(sh), run_time=0.5)
        kl = Text("its average over the model's answers = the KL divergence", font_size=26, color=REWARD_C).move_to([0, -2.0, 0])
        self.at("kl")
        self.play(FadeIn(kl), run_time=0.5)
        far = Text("how far the model has moved", font_size=22, color=GREY_B).next_to(kl, DOWN, buff=0.2)
        self.at("moved")
        self.play(FadeIn(far), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. rescue
    def rescue(self):
        self.section(4)
        self.clear_stage()
        ax, frame = self.axes([2.6, 0.2, 0])
        h = Text("learning rate 0.003 (10×)", font_size=24, color=GREY_A).move_to([-6.6, 2.6, 0], aligned_edge=LEFT)
        self.play(FadeIn(frame), FadeIn(h), run_time=0.4)
        c0 = self.curve(ax, FAST[0.0]["curve"], BAD)
        l0 = Text("no leash", font_size=22, color=BAD).move_to([-6.6, 1.9, 0], aligned_edge=LEFT)
        self.play(Create(c0), FadeIn(l0), run_time=0.5)
        l1 = Text("β = 0.1", font_size=26, color=LEASH_C).move_to([-6.6, 1.3, 0], aligned_edge=LEFT)
        self.at("beta")
        self.play(FadeIn(l1), run_time=0.3)
        c1 = self.curve(ax, FAST[0.1]["curve"], LEASH_C, width=4)
        wob = FAST[0.1]["samples"]["500"]
        odd = VGroup(*[Text(f"{q} {' '.join(strange_first(a)[:4])}", font=MONO, font_size=18, color=GREY_A) for q, a in wob.items()]
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([-6.6, 0.2, 0], aligned_edge=LEFT)
        ol = Text("at step 500:", font_size=18, color=GREY_B).next_to(odd, UP, buff=0.12).align_to(odd, LEFT)
        self.at("wobbles")
        self.play(Create(c1), run_time=2.2)
        self.at("strange")
        self.play(FadeIn(ol), FadeIn(odd), run_time=0.4)
        fin = Text(f"{pct(FAST[0.1]['curve'][-1]['greedy_acc'])} right", font_size=32, color=LEASH_C).move_to([-6.6, -1.1, 0], aligned_edge=LEFT)
        self.at("recovers")
        self.play(FadeIn(fin), run_time=0.4)
        k = VGroup(Text(f"KL {FAST[0.1]['curve'][-1]['kl']:.1f}", font_size=24, color=LEASH_C),
                   Text(f"vs {FAST[0.0]['curve'][-1]['kl']:.0f} without", font_size=24, color=BAD)).arrange(RIGHT, buff=0.3).move_to([-6.6, -1.9, 0], aligned_edge=LEFT)
        self.at("seven")
        self.play(FadeIn(k[0]), run_time=0.3)
        self.at("23")
        self.play(FadeIn(k[1]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5. sweep
    def sweep(self):
        self.section(5)
        self.clear_stage()
        h = Text("learning rate 0.0003 · 2 runs per β · 2,000 steps", font_size=22, color=GREY_B).to_edge(UP, buff=0.6)
        self.at("cost")
        self.play(FadeIn(h), run_time=0.4)
        base_y, unit, w = -1.6, 4.0, 1.0
        cols = VGroup()
        for i, b in enumerate(R["betas"]):
            s = sweep(b)
            x = -5.0 + 2.0 * i
            col = interpolate_color(ManimColor(GOOD), ManimColor(LEASH_C), i / 5)
            bar = Rectangle(width=w, height=max(0.03, unit * s["greedy_acc"]), stroke_width=0, fill_color=col, fill_opacity=0.8)
            bar.move_to([x, base_y, 0], aligned_edge=DOWN)
            v = Text(pct(s["greedy_acc"]), font_size=24, color=col).next_to(bar, UP, buff=0.12)
            lab = Text(f"β = {b:g}", font_size=22).move_to([x, base_y - 0.4, 0])
            kl = Text(f"KL {s['kl']:.1f}", font_size=18, color=GREY_B).move_to([x, base_y - 0.85, 0])
            cols.add(VGroup(bar, v, lab, kl))
        base = Line([-6.2, base_y, 0], [6.2, base_y, 0], color=GREY_C)
        start = DashedLine([-6.2, base_y + unit * R["start"]["greedy_acc"], 0], [6.2, base_y + unit * R["start"]["greedy_acc"], 0], color=GREY_B, stroke_width=2)
        sl = Text(f"dashed line: the start, {pct(R['start']['greedy_acc'])}", font_size=18, color=GREY_B).next_to(h, DOWN, buff=0.15)
        self.at("six")
        self.play(Create(base), Create(start), FadeIn(sl), *[FadeIn(c[2]) for c in cols], run_time=0.5)
        self.at("03")
        self.play(*[AnimationGroup(GrowFromEdge(c[0], DOWN), FadeIn(c[1]), FadeIn(c[3])) for c in cols[:3]], run_time=0.6)
        self.at("drops")
        self.play(GrowFromEdge(cols[3][0], DOWN), FadeIn(cols[3][1]), FadeIn(cols[3][3]), run_time=0.4)
        self.at("barely")
        self.play(*[AnimationGroup(GrowFromEdge(c[0], DOWN), FadeIn(c[1]), FadeIn(c[3])) for c in cols[4:]], run_time=0.5)
        t1 = Text("too tight: can't learn", font_size=24, color=LEASH_C).move_to([3.0, 2.0, 0])
        t2 = Text("too loose: no protection", font_size=24, color=GOOD).move_to([-3.0, 2.0, 0])
        self.at("tight")
        self.play(FadeIn(t1), run_time=0.3)
        self.at("loose")
        self.play(FadeIn(t2), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. how the text reads
    def text(self):
        self.section(6)
        self.clear_stage()
        q = "47+38="
        h = Text(f"8 sampled answers to {q}", font=MONO, font_size=26).to_edge(UP, buff=0.8)
        self.at("look")
        self.play(FadeIn(h), run_time=0.4)
        lone = [r for r in R["sweep"] if r["beta"] == 0.0 and r["seed"] == 1][0]["samples"]["2000"][q]
        tight = [r for r in R["sweep"] if r["beta"] == 1.0 and r["seed"] == 1][0]["samples"]["2000"][q]
        rows = [("reference", R["start"]["samples"][q], REF_C, "reference"),
                ("no leash", lone, GOOD, "without"),
                ("β = 1", tight, LEASH_C, "tightest")]
        y = 1.3
        for label, ans, col, cue in rows:
            lab = Text(label, font_size=24, color=col).move_to([-6.4, y, 0], aligned_edge=LEFT)
            row = answer_row(ans, "85.").move_to([1.0, y, 0])
            self.at(cue)
            self.play(FadeIn(lab), LaggedStart(*[FadeIn(c) for c in row], lag_ratio=0.08), run_time=0.7)
            y -= 1.0
        a = Text("arithmetic: one right answer is what we want", font_size=24, color=GREY_A).move_to([0, -1.8, 0])
        self.at("single")
        self.play(FadeIn(a), run_time=0.4)
        c = Text("a chatbot: that variety is its style, phrasing and knowledge", font_size=24, color=REWARD_C).move_to([0, -2.6, 0])
        self.at("chatbot")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. why
    def why(self):
        self.section(7)
        self.clear_stage()
        h = Text("RL for language models always keeps a leash", font_size=30, color=REWARD_C).to_edge(UP, buff=0.7)
        self.at("leash")
        self.play(FadeIn(h), run_time=0.4)
        items = VGroup(
            Text("the reward model is imperfect (episode 7)", font_size=24),
            Text("a model free to drift finds strange text that scores well (episode 8)", font_size=24),
            Text("the reference = the model before RL", font_size=24, color=REF_C),
            VGroup(Text("PPO", font_size=26, color=MODEL_C), Text("puts the penalty in the reward, as we did", font_size=24)).arrange(RIGHT, buff=0.3),
            VGroup(Text("GRPO", font_size=26, color=MODEL_C), Text("adds it to the loss", font_size=24)).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([0, -0.4, 0])
        for it, cue in zip(items, ("imperfect", "strange", "reference", "ppo", "grpo")):
            self.at(cue)
            self.play(FadeIn(it, shift=0.1 * RIGHT), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.at("log")
        self.play(Create(hl), run_time=0.3)
        self.at("beta")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("baseline")
        self.play(highlight(hl, code, 3), run_time=0.3)
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
