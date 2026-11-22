"""How Generative Models Work: Images and Sound, episode 7 — Sampling: From Noise to Picture.

Render from the repo root:  ./render.sh generative g07
Numbers and pictures come from code/g07_sampling (sample_compare.py → results.json and out/*.png).
"""
import json
import math

from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import frame, tile
from intro import play_token_intro
from g07_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 7"
CODE_DIR = "code/g07_sampling"
OUT = f"{CODE_DIR}/out"
R = json.load(open(f"{CODE_DIR}/results.json"))
RUN = {(r["eta"], r["steps"]): r for r in R["runs"]}
STEPS = [10, 20, 50, 100, 250, 1000]
DDIM_C, DDPM_C, GUESS_C, NET_C = BLUE_C, ORANGE, YELLOW, TEAL_C
TRAJ = f"{OUT}/trajectory.png"
CODE = """eps = model(x, t)
x0 = ((x - (1-a_t).sqrt()*eps) / a_t.sqrt()).clamp(-1, 1)   # guess
sigma = eta * ((1-a_s)/(1-a_t) * (1 - a_t/a_s)).sqrt()     # 0: DDIM, 1: DDPM
x = a_s.sqrt()*x0 + (1-a_s-sigma**2).sqrt()*eps + sigma*torch.randn_like(x)"""


def row(path, n=8, h=1.0, r=0, buff=0.1):
    return Group(*[tile(path, i, h, row=r) for i in range(n)]).arrange(RIGHT, buff=buff)


class SamplingVideo(VoicedScene):
    VIDEO = "g07"

    def construct(self):
        play_token_intro(self, TITLE, 7, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.one_step()      # 2
        self.one_sample()    # 3
        self.coarse()        # 4
        self.steps_chart()   # 5
        self.why()           # 6
        self.same_noise()    # 7
        self.cost()          # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        a = row(f"{OUT}/samples_eta0_steps0050.png", h=1.15).move_to([0.9, 1.4, 0])
        la = Text(f"{RUN[(0.0, 50)]['recognizable']:.0%}", font=MONO, font_size=34, color=DDIM_C).next_to(a, LEFT, buff=0.4)
        sa = Text("episode 6: 50 steps", font_size=18, color=GREY_B).next_to(a, UP, buff=0.15)
        self.at("70")
        self.play(FadeIn(a), FadeIn(la), FadeIn(sa), run_time=0.6)
        b = row(f"{OUT}/samples_eta1_steps0050.png", h=1.15).move_to([0.9, -1.3, 0])
        lb = Text(f"{RUN[(1.0, 50)]['recognizable']:.0%}", font=MONO, font_size=34, color=DDPM_C).next_to(b, LEFT, buff=0.4)
        sb = Text("same network, same 50 steps, a different walk back", font_size=18, color=GREY_B).next_to(b, UP, buff=0.15)
        self.at("different")
        self.play(FadeIn(sb), FadeIn(b), run_time=0.6)
        self.at("95")
        self.play(FadeIn(lb, scale=1.3), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. one step
    def one_step(self):
        self.section(2)
        self.clear_stage()
        h = 1.4
        xt = tile(TRAJ, 1, h).move_to([-5.4, 1.5, 0])
        lt = Text("xₜ (step 800)", font_size=18, color=GREY_B).next_to(xt, DOWN, buff=0.12)
        net = VGroup(RoundedRectangle(width=1.8, height=0.9, corner_radius=0.15, color=NET_C, fill_color=NET_C, fill_opacity=0.15),
                     Text("network", font_size=20, color=NET_C)).move_to([-2.8, 1.5, 0])
        net[1].move_to(net[0])
        a1 = Arrow(xt.get_right(), net.get_left(), buff=0.1, color=GREY_B)
        self.at("move")
        self.play(FadeIn(xt), FadeIn(frame(xt)), FadeIn(lt), run_time=0.4)
        self.at("ask")
        self.play(GrowArrow(a1), FadeIn(net), run_time=0.4)
        guess = tile(TRAJ, 1, h, row=1).move_to([0.0, 1.5, 0])
        lg = Text("clean guess", font_size=18, color=GUESS_C).next_to(guess, DOWN, buff=0.12)
        a2 = Arrow(net.get_right(), guess.get_left(), buff=0.1, color=GUESS_C)
        self.at("guess")
        self.play(GrowArrow(a2), FadeIn(guess), FadeIn(frame(guess, GUESS_C)), FadeIn(lg), run_time=0.5)
        xs = tile(TRAJ, 2, h).move_to([3.4, 1.5, 0])
        ls = Text("lower level (step 600)", font_size=18, color=GREY_B).next_to(xs, DOWN, buff=0.12)
        a3 = Arrow(guess.get_right(), xs.get_left(), buff=0.1, color=GREY_B)
        l3 = Text("noise it back\ndown", font_size=16, color=GREY_B, line_spacing=0.9).next_to(a3, UP, buff=0.1)
        self.at("then")
        self.play(GrowArrow(a3), FadeIn(l3), FadeIn(xs), FadeIn(frame(xs)), FadeIn(ls), run_time=0.5)
        rep = CurvedArrow(xs.get_top() + UP * 0.1, xt.get_top() + UP * 0.1, angle=PI / 5, color=GREY_D)
        self.at("repeat")
        self.play(Create(rep), run_time=0.6)
        q = Text("which noise for that step?", font_size=24).move_to([0, -0.9, 0])
        self.at("question")
        self.play(FadeIn(q), run_time=0.4)
        ddim = VGroup(Text("DDIM", font_size=26, color=DDIM_C),
                      Text("reuse the network's own prediction\n→ the walk is fixed by the starting noise", font_size=18, line_spacing=0.9)
                      ).arrange(DOWN, buff=0.15).move_to([-3.3, -2.3, 0])
        ddpm = VGroup(Text("DDPM", font_size=26, color=DDPM_C),
                      Text("mix in fresh random noise\nat every step", font_size=18, line_spacing=0.9)
                      ).arrange(DOWN, buff=0.15).move_to([3.3, -2.3, 0])
        self.at("ddim")
        self.play(FadeIn(ddim), run_time=0.5)
        self.at("ddpm")
        self.play(FadeIn(ddpm), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 3. one sample
    def one_sample(self):
        self.section(3)
        self.clear_stage()
        top, bot = row(TRAJ, h=1.25, r=0, buff=0.15), row(TRAJ, h=1.25, r=1, buff=0.15)
        g = Group(top, bot).arrange(DOWN, buff=0.45).move_to([0.6, 0.0, 0])
        labels = VGroup(*[Text(f"t = {t:,}", font=MONO, font_size=16, color=GREY_B).next_to(top[i], UP, buff=0.12)
                          for i, t in enumerate(R["trajectory_steps"])])
        lt = Text("xₜ", font_size=22).next_to(top, LEFT, buff=0.3)
        lb = Text("clean\nguess", font_size=20, color=GUESS_C, line_spacing=0.9).next_to(bot, LEFT, buff=0.3)
        self.at("walk")
        self.play(FadeIn(top), FadeIn(labels), FadeIn(lt), run_time=0.6)
        self.at("underneath")
        self.play(FadeIn(bot), FadeIn(lb), run_time=0.5)
        self.at("800")
        b1 = SurroundingRectangle(bot[1], color=RED, buff=0.05)
        self.play(Create(b1), run_time=0.3)
        n1 = Text("undecided: two faint blobs", font_size=18, color=RED_B).next_to(bot, DOWN, buff=0.3).align_to(bot[0], LEFT)
        self.play(FadeIn(n1), run_time=0.3)
        self.at("600")
        b2 = SurroundingRectangle(bot[2], color=GREEN, buff=0.05)
        n2 = Text("a red circle, from here on only sharper", font_size=18, color=GREEN_B).next_to(bot, DOWN, buff=0.3).align_to(bot[2], LEFT).shift(RIGHT * 1.5)
        self.play(Create(b2), FadeIn(n2), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. coarse to fine
    def coarse(self):
        self.section(4)
        self.clear_stage()
        ax = Axes(x_range=[0, 1000, 200], y_range=[0, 1, 0.25], x_length=8.0, y_length=4.2, tips=False,
                  axis_config={"color": GREY_C}).move_to([-1.0, -0.3, 0])
        X = lambda t: ax.c2p(1000 - t, 0)[0]                         # time runs right to left: step 1,000 → 1
        xl = VGroup(*[Text(f"{t:,}", font=MONO, font_size=14, color=GREY_B).move_to([X(t), ax.c2p(0, 0)[1] - 0.3, 0]) for t in (1000, 800, 600, 400, 200, 0)])
        yl = VGroup(*[Text(f"{int(v * 100)}%", font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, 0.5, 1)])
        xt = Text("step of the walk", font_size=16, color=GREY_B).next_to(xl, DOWN, buff=0.12)
        title = Text(f"how often the clean guess already shows the final answer ({R['decided_n']} samples)", font_size=20).to_edge(UP, buff=0.4)
        self.at("480")
        self.play(FadeIn(title), FadeIn(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), run_time=0.6)
        D = R["decided"]
        curves, legend = {}, VGroup()
        for key, col in (("position", BLUE_C), ("color", ORANGE), ("shape", GREEN_C)):
            pts = [[X(d["t"]), ax.c2p(0, d[key])[1], 0] for d in D]
            curves[key] = VMobject(color=col, stroke_width=4).set_points_as_corners(pts)
            legend.add(Text(key, font_size=20, color=col))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([4.6, 0.6, 0])
        d800 = next(d for d in D if d["t"] == 800)
        d500 = next(d for d in D if d["t"] == 500)
        self.at("800")
        line = DashedLine([X(800), ax.c2p(0, 0)[1], 0], [X(800), ax.c2p(0, 1)[1], 0], color=GREY_D)
        self.play(Create(line), run_time=0.3)
        self.at("position")
        self.play(Create(curves["position"]), FadeIn(legend[0]), run_time=0.8)
        self.at("88")
        m = Text(f"{d800['position']:.0%}", font=MONO, font_size=18, color=BLUE_C).next_to([X(800), ax.c2p(0, d800['position'])[1], 0], LEFT, buff=0.12)
        self.play(FadeIn(m), run_time=0.3)
        self.at("color")
        self.play(Create(curves["color"]), FadeIn(legend[1]), run_time=0.8)
        self.at("shape")
        self.play(Create(curves["shape"]), FadeIn(legend[2]), run_time=0.8)
        self.at("500")
        m2 = Text(f"{d500['shape']:.0%}", font=MONO, font_size=18, color=GREEN_C).next_to([X(500), ax.c2p(0, d500['shape'])[1], 0], DOWN, buff=0.2)
        self.play(FadeIn(m2), run_time=0.3)
        cf = Text("coarse first,\ndetails later", font_size=22, color=YELLOW, line_spacing=0.9).next_to(legend, DOWN, buff=0.5, aligned_edge=LEFT)
        self.at("course", "coarse")
        self.play(FadeIn(cf), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. steps vs quality
    def steps_chart(self):
        self.section(5)
        self.clear_stage()
        ax = Axes(x_range=[1, 3, 1], y_range=[0, 1, 0.25], x_length=7.6, y_length=4.4, tips=False,
                  axis_config={"color": GREY_C}).move_to([-1.2, -0.3, 0])
        P = lambda s, v: ax.c2p(math.log10(s), v)
        xl = VGroup(*[Text(f"{s:,}", font=MONO, font_size=14, color=GREY_B).next_to(P(s, 0), DOWN, buff=0.12) for s in STEPS])
        yl = VGroup(*[Text(f"{int(v * 100)}%", font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(1, v), LEFT, buff=0.1) for v in (0, 0.5, 1)])
        xt = Text("sampling steps", font_size=16, color=GREY_B).next_to(xl, DOWN, buff=0.12)
        title = Text("recognized by the judge (500 samples, same starting noise)", font_size=20).to_edge(UP, buff=0.4)
        self.play(FadeIn(title), FadeIn(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), run_time=0.5)
        lines, dots = {}, {}
        for eta, col in ((1.0, DDPM_C), (0.0, DDIM_C)):
            pts = [P(s, RUN[(eta, s)]["recognizable"]) for s in STEPS]
            lines[eta] = VMobject(color=col, stroke_width=4).set_points_as_corners(pts)
            dots[eta] = VGroup(*[Dot(p, radius=0.06, color=col) for p in pts])
        leg = VGroup(Text("with fresh noise (DDPM-style, η = 1)", font_size=18, color=DDPM_C),
                     Text("without (DDIM, η = 0)", font_size=18, color=DDIM_C)).arrange(RIGHT, buff=0.8).move_to([0, 2.65, 0])
        lab = lambda eta, s, d=UP: Text(f"{RUN[(eta, s)]['recognizable']:.0%}", font=MONO, font_size=16,
                                       color=DDPM_C if eta else DDIM_C).next_to(P(s, RUN[(eta, s)]["recognizable"]), d, buff=0.12)
        self.at("fresh")
        self.play(FadeIn(leg[0]), run_time=0.3)
        self.at("85")
        self.play(FadeIn(dots[1.0][0]), FadeIn(lab(1.0, 10, DOWN)), run_time=0.3)
        self.at("95")
        self.play(Create(lines[1.0]), FadeIn(dots[1.0][1:]), run_time=1.0)
        self.at("98")
        self.play(FadeIn(lab(1.0, 100)), run_time=0.3)
        self.at("without")
        self.play(FadeIn(leg[1]), run_time=0.3)
        self.at("16")
        self.play(FadeIn(dots[0.0][0]), FadeIn(lab(0.0, 10)), run_time=0.3)
        self.at("75")
        part = VMobject(color=DDIM_C, stroke_width=4).set_points_as_corners([P(s, RUN[(0.0, s)]["recognizable"]) for s in STEPS[:4]])
        self.play(Create(part), FadeIn(dots[0.0][1:4]), FadeIn(lab(0.0, 100, DOWN)), run_time=0.8)
        self.at("1000")
        self.play(Create(lines[0.0]), FadeIn(dots[0.0][4:]), run_time=0.8)
        self.at("19")
        self.play(FadeIn(lab(0.0, 1000, RIGHT)), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. why
    def why(self):
        self.section(6)
        self.clear_stage()
        a = row(f"{OUT}/samples_eta0_steps1000.png", h=1.15, r=0).move_to([0.9, 1.6, 0])
        b = row(f"{OUT}/samples_eta1_steps1000.png", h=1.15, r=0).move_to([0.9, -0.4, 0])
        la = Text("1,000 steps\nno fresh noise", font_size=18, color=DDIM_C, line_spacing=0.9).next_to(a, LEFT, buff=0.3)
        lb = Text("1,000 steps\nfresh noise", font_size=18, color=DDPM_C, line_spacing=0.9).next_to(b, LEFT, buff=0.3)
        self.at("small")
        self.play(FadeIn(a), FadeIn(la), run_time=0.5)
        bad = VGroup(*[SurroundingRectangle(a[i], color=RED, buff=0.04) for i in (3, 5)])
        self.at("bumps")
        self.play(Create(bad), run_time=0.4)
        self.at("fresh")
        self.play(FadeIn(b), FadeIn(lb), run_time=0.5)
        n = Text("fresh noise lets each step fix the last one's mistake", font_size=22, color=DDPM_C).move_to([0, -1.9, 0])
        self.at("chance")
        self.play(FadeIn(n), run_time=0.4)
        c = Text("(in large, well-trained models DDIM does much better than here)", font_size=18, color=GREY_B).move_to([0, -2.7, 0])
        self.at("large")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. same starting noise
    def same_noise(self):
        self.section(7)
        self.clear_stage()
        path = f"{OUT}/same_noise.png"
        rows_ = {r: row(path, h=0.72, r=r, buff=0.08) for r in (0, 1, 2, 4, 5)}
        order = [0, 1, 2, 4, 5]
        g = Group(*[rows_[r] for r in order]).arrange(DOWN, buff=0.12).move_to([0.2, -0.25, 0])
        g[3].shift(DOWN * 0.25)
        g[4].shift(DOWN * 0.25)
        names = {0: ("DDIM, 10 steps", DDIM_C), 1: ("DDIM, 20 steps", DDIM_C), 2: ("DDIM, 50 steps", DDIM_C),
                 4: ("DDPM, run A", DDPM_C), 5: ("DDPM, run B", DDPM_C)}
        labs = {r: Text(names[r][0], font_size=18, color=names[r][1]).next_to(rows_[r], LEFT, buff=0.3) for r in order}
        head = Text("8 starting noises, one per column", font_size=20, color=GREY_B).next_to(g, UP, buff=0.2)
        self.at("same")
        self.play(FadeIn(head), run_time=0.3)
        for r, cue in ((0, "10"), (1, "20"), (2, "50")):
            self.at(cue)
            self.play(FadeIn(rows_[r]), FadeIn(labs[r]), run_time=0.3)
        addr = Text("the starting noise\nworks like an address", font_size=18, color=YELLOW, line_spacing=0.9).next_to(rows_[1], RIGHT, buff=0.25)
        self.at("address")
        self.play(FadeIn(addr), run_time=0.3)
        self.at("twice")
        self.play(FadeIn(rows_[4]), FadeIn(labs[4]), FadeIn(rows_[5]), FadeIn(labs[5]), run_time=0.5)
        ch = Text("all 8 change", font_size=18, color=DDPM_C).next_to(rows_[5], RIGHT, buff=0.25).shift(UP * 0.45)
        self.at("change")
        self.play(FadeIn(ch), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8. cost
    def cost(self):
        self.section(8)
        self.clear_stage()
        head = Text("steps    time per picture    recognized (fresh noise)", font=MONO, font_size=20, color=GREY_B)
        lines = [head]
        for s in STEPS:
            r = RUN[(1.0, s)]
            ms = r["ms_per_sample"]
            t = f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"
            lines.append(Text(f"{s:>5,}    {t:>16}    {r['recognizable']:>12.1%}", font=MONO, font_size=20))
        tab = VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, 0.2, 0])
        t1 = Text("one network call per step", font_size=24, color=YELLOW).next_to(tab, UP, buff=0.5)
        self.at("one")
        self.play(FadeIn(t1), run_time=0.4)
        self.at("cpu")
        self.play(FadeIn(tab[0]), LaggedStart(*[FadeIn(l) for l in tab[1:]], lag_ratio=0.15), run_time=1.2)
        box = SurroundingRectangle(tab[2], color=DDPM_C, buff=0.08)
        sw = Text("sweet spot for this model", font_size=18, color=DDPM_C).next_to(tab, DOWN, buff=0.4)
        self.at("sweet")
        self.play(Create(box), FadeIn(sw), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=18)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.at("guess")
        self.play(Create(hl), run_time=0.2)
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("choose")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("step")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10. outro
    def outro(self):
        self.section(10)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
